"""Image edits preserve source bytes and enforce connected split boundaries."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.dataset import DatasetError, load_dataset
from ipde.dataset_edit import apply_dataset_edits, edit_dataset
from ipde.dataset_review import _array_records, _publish_new_directory, review_dataset
from ipde.resource_lock import resource_lock
from ipde.trainer_cli import main, workspace_report


def _dataset(root: Path, *, offset: int = 0, compressed: bool = False) -> tuple[Path, list[dict]]:
    root.mkdir()
    samples = []
    for index in range(4):
        rgb = array_record(root, root / f"rgb-{index}.npy", np.full((2, 3, 3), offset + index, np.uint16), compressed=False)
        depth = np.full((2, 3), 1.234567, ">f4")
        depth.view(">u4")[0, :2] = [0x7fc01234, 0x80000000]
        target = array_record(root, root / f"target-{index}{'.npz' if compressed else '.npy'}", depth, compressed=compressed)
        mask = array_record(root, root / f"mask-{index}.npy", np.isfinite(depth), compressed=False)
        aux = array_record(root, root / f"aux-{index}.npy", np.array([0x7e55, 0x8000, 0x3555], np.uint16).view(np.float16), compressed=False)
        samples.append({"id": f"photo-{index}", "source_path": f"IMG_{offset + index}.HEIC", "source_sha256": f"source-{offset + index}",
                        "group_id": f"group-{index}", "split": "validation" if index == 3 else "train",
                        "rgb": rgb, "right_rgb": rgb, "raw_assets": [{"storage": aux}],
                        "teacher": {"units": "meters", "target": target, "native_target": target, "valid_mask": mask,
                                    "metadata": {"checkpoint_sha256": "a" * 64, "model_id": "Teacher"}},
                        "photo_metadata": {"category": "Architecture"}, "custom_scientific_metadata": {"untouched": True}})
    (root / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1", "name": "Original", "samples": samples,
                                                  "custom_dataset_metadata": {"precision": "raw"}}))
    return root, samples


def _rewrite(root: Path, mutate) -> None:
    manifest = json.loads((root / "dataset.json").read_text())
    mutate(manifest)
    (root / "dataset.json").write_text(json.dumps(manifest))


def _bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class DatasetEditTests(unittest.TestCase):
    def test_removal_preserves_exact_files_arrays_and_metadata_without_array_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root / "source")
            before = _bytes(source)
            output, events = root / "edited", []
            with patch("ipde.dataset_collection._clone_array_file", return_value=False), \
                 patch("ipde.array_storage.array_record", side_effect=AssertionError("Unexpected array writer")), \
                 patch("shutil.copyfile", side_effect=AssertionError("Unexpected payload copy")):
                report = edit_dataset(source, {"keep": ["photo-0", "photo-3"]}, output, workers=2, progress_callback=events.append)
            saved = load_dataset(output)
            self.assertEqual([sample["id"] for sample in saved["samples"]], ["photo-0", "photo-3"])
            self.assertEqual(saved["custom_dataset_metadata"], {"precision": "raw"})
            self.assertEqual(saved["name"], "edited")
            for sample in saved["samples"]:
                original = next(item for item in samples if item["id"] == sample["id"])
                self.assertEqual(sample["custom_scientific_metadata"], original["custom_scientific_metadata"])
                self.assertEqual(sample["split"], original["split"])
                for old, new in zip(_array_records(original), _array_records(sample)):
                    self.assertEqual({key: old[key] for key in old if key != "path"}, {key: new[key] for key in new if key != "path"})
                    self.assertEqual((output / new["path"]).read_bytes(), before[old["path"]])
                    self.assertEqual(read_array(output / new["path"]).tobytes(), read_array(source / old["path"]).tobytes())
                    representatives = {path.stat().st_ino for path in source.iterdir() if path.is_file() and path.read_bytes() == before[old["path"]]}
                    self.assertIn((output / new["path"]).stat().st_ino, representatives)
            self.assertEqual(before, _bytes(source))
            self.assertEqual(report["dataset_edit"]["excluded_sample_ids"], ["photo-1", "photo-2"])
            self.assertEqual(report["collection"]["added_array_storage_bytes"], 0)
            self.assertEqual(report["collection"]["added_storage_bytes"], (output / "dataset.json").stat().st_size)
            self.assertEqual([event["phase"] for event in events[-2:]], ["verifying_output", "publishing_dataset"])
            shutil.rmtree(source)
            self.assertEqual(load_dataset(output)["summary"]["samples"], 2)

    def test_additions_namespace_colliding_ids_and_preserve_all_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = _dataset(root / "base")
            added, added_samples = _dataset(root / "added", offset=100)
            _rewrite(added, lambda data: data.update(custom_dataset_metadata={"camera": "second"}))
            originals = _bytes(base), _bytes(added)
            output = root / "edited"
            report = edit_dataset(base, {"keep": ["photo-0", "photo-3"]}, output, add_datasets=[added], workers=2)
            saved = load_dataset(output)
            self.assertEqual(report["summary"]["samples"], 6)
            self.assertEqual(report["dataset_edit"]["added_samples"], 4)
            imported = saved["samples"][2:]
            self.assertEqual(len({sample["id"] for sample in saved["samples"]}), 6)
            self.assertTrue(all(sample["id"].startswith("added-") for sample in imported))
            self.assertEqual([sample["edit_provenance"]["source_sample_id"] for sample in imported], [sample["id"] for sample in added_samples])
            self.assertEqual(report["dataset_edit"]["sources"][1]["metadata"]["custom_dataset_metadata"], {"camera": "second"})
            self.assertEqual(originals, (_bytes(base), _bytes(added)))

    def test_duplicate_addition_split_conflict_requires_and_accepts_explicit_group_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = _dataset(root / "base")
            added, _ = _dataset(root / "added")
            _rewrite(added, lambda data: data["samples"][0].update(split="validation"))
            with self.assertRaisesRegex(DatasetError, "different existing splits.*Train or Validation"):
                edit_dataset(base, {}, root / "conflict", add_datasets=[added])
            output = root / "resolved"
            edit_dataset(base, {"splits": {"photo-0": "validation"}}, output, add_datasets=[added])
            saved = load_dataset(output)
            duplicate = [sample for sample in saved["samples"] if sample["source_sha256"] == "source-0"]
            self.assertEqual(len(duplicate), 2)
            self.assertEqual({sample["split"] for sample in duplicate}, {"validation"})
            self.assertEqual(len({sample["group_id"] for sample in duplicate}), 1)

    def test_overrides_move_transitively_linked_teacher_burst_and_scene_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            def link(data):
                entries = data["samples"]
                entries[1]["source_sha256"] = entries[0]["source_sha256"]
                entries[1]["teacher_id"] = "different-teacher"
                entries[1]["burst_ids"] = ["burst"]
                entries[2]["burst_ids"] = ["burst"]
                entries[2]["requested_group"] = "one-scene"
                entries[3]["requested_group"] = "one-scene"
            _rewrite(source, link)
            original = (source / "dataset.json").read_bytes()
            review = review_dataset(source)
            self.assertEqual(len({sample["management_group_id"] for sample in review["samples"]}), 1)
            self.assertEqual((source / "dataset.json").read_bytes(), original)
            output = root / "edited"
            edit_dataset(source, {"splits": {"photo-0": "validation"}}, output)
            saved = load_dataset(output)
            self.assertEqual({sample["split"] for sample in saved["samples"]}, {"validation"})
            self.assertEqual(len({sample["group_id"] for sample in saved["samples"]}), 1)
            self.assertTrue(any("before training" in message for message in saved["warnings"]))
            with self.assertRaisesRegex(DatasetError, "Conflicting split choices.*same split"):
                edit_dataset(source, {"splits": {"photo-0": "train", "photo-3": "validation"}}, root / "conflict")

    def test_existing_group_survives_removed_connector_and_categories_do_not_join_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            _rewrite(source, lambda data: data["samples"][1].update(group_id="group-0"))
            output = root / "edited"
            edit_dataset(source, {"keep": ["photo-0", "photo-1", "photo-3"], "splits": {"photo-0": "validation"}}, output)
            samples = load_dataset(output)["samples"]
            self.assertEqual(samples[0]["group_id"], samples[1]["group_id"])
            self.assertNotEqual(samples[0]["group_id"], samples[2]["group_id"])
            self.assertEqual(samples[1]["split"], "validation")

    def test_scene_namespace_is_preserved_across_additions_and_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = _dataset(root / "base")
            added, _ = _dataset(root / "added", offset=100)
            for source in (base, added):
                _rewrite(source, lambda data: data["samples"][0].update(requested_group="scene-A"))
            first, second = root / "first", root / "second"
            edit_dataset(base, {"splits": {"photo-0": "validation"}}, first, add_datasets=[added])
            originals = load_dataset(first)["samples"]
            self.assertNotEqual(originals[0]["group_id"], originals[4]["group_id"])
            edit_dataset(first, {}, second)
            saved = load_dataset(second)["samples"]
            self.assertNotEqual(saved[0]["group_id"], saved[4]["group_id"])
            self.assertEqual(saved[4]["edit_provenance"]["group_namespace"], str(added.resolve()))

    def test_identical_array_values_in_different_containers_retain_each_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, base_samples = _dataset(root / "base")
            added, added_samples = _dataset(root / "added", offset=100, compressed=True)
            output = root / "edited"
            edit_dataset(base, {}, output, add_datasets=[added])
            saved = load_dataset(output)["samples"]
            self.assertEqual(saved[0]["teacher"]["target"]["array_sha256"], saved[4]["teacher"]["target"]["array_sha256"])
            self.assertNotEqual(saved[0]["teacher"]["target"]["file_sha256"], saved[4]["teacher"]["target"]["file_sha256"])
            self.assertEqual(saved[0]["teacher"]["target"]["file_sha256"], base_samples[0]["teacher"]["target"]["file_sha256"])
            self.assertEqual(saved[4]["teacher"]["target"]["file_sha256"], added_samples[0]["teacher"]["target"]["file_sha256"])

    def test_ancestral_authored_group_survives_merging_independently_edited_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            _rewrite(source, lambda data: data["samples"][1].update(group_id="group-0"))
            left, right = root / "left", root / "right"
            edit_dataset(source, {"keep": ["photo-0", "photo-3"]}, left)
            edit_dataset(source, {"keep": ["photo-1", "photo-2", "photo-3"]}, right)
            output = root / "merged"
            edit_dataset(left, {"splits": {"photo-0": "validation"}}, output, add_datasets=[right])
            samples = load_dataset(output)["samples"]
            authored = [sample for sample in samples if sample["source_sha256"] in {"source-0", "source-1"}]
            self.assertEqual(len(authored), 2)
            self.assertEqual(len({sample["group_id"] for sample in authored}), 1)
            self.assertEqual({sample["split"] for sample in authored}, {"validation"})
            review = review_dataset(output)
            self.assertEqual(len({sample["management_group_id"] for sample in review["samples"] if sample["source_id"] in {"source-0", "source-1"}}), 1)

    def test_invalid_edits_and_destinations_are_actionable_and_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            before = _bytes(source)
            cases = [(None, "object"), ({"other": 1}, "only keep and splits"), ({"keep": "photo-0"}, "list"),
                     ({"keep": ["photo-0", "photo-0"]}, "Duplicate"), ({"keep": ["absent"]}, "Unknown base"),
                     ({"keep": []}, "cannot be empty"), ({"splits": {"photo-0": []}}, "map sample IDs"),
                     ({"splits": {"absent": "train"}}, "removed or unknown"),
                     ({"keep": ["photo-1"], "splits": {"photo-0": "train"}}, "removed or unknown")]
            for index, (edits, message) in enumerate(cases):
                with self.subTest(edits=edits), self.assertRaisesRegex(DatasetError, message):
                    edit_dataset(source, edits, root / f"failed-{index}")
            with self.assertRaisesRegex(DatasetError, "outside every source"):
                edit_dataset(source, {}, source / "inside")
            with self.assertRaisesRegex(DatasetError, "already exists"):
                edit_dataset(source, {}, source)
            with self.assertRaisesRegex(DatasetError, "already the dataset"):
                edit_dataset(source, {}, root / "duplicate", add_datasets=[source])
            with self.assertRaisesRegex(DatasetError, "workers"):
                edit_dataset(source, {}, root / "worker", workers=-1)
            self.assertEqual(list(root.iterdir()), [source])
            self.assertEqual(before, _bytes(source))

    def test_removed_all_base_images_is_allowed_when_adding_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = _dataset(root / "base")
            added, _ = _dataset(root / "added", offset=100)
            report = edit_dataset(base, {"keep": []}, root / "edited", add_datasets=[added])
            self.assertEqual(report["summary"]["samples"], 4)
            self.assertEqual(report["dataset_edit"]["kept_sample_ids"], [])

    def test_corrupt_future_array_is_rejected_even_when_its_sample_would_be_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root / "source")
            _rewrite(source, lambda data: data["samples"][2].update(future_scientific_plane=copy.deepcopy(samples[2]["rgb"])))
            (source / samples[2]["rgb"]["path"]).write_bytes(b"corrupt")
            with self.assertRaises(DatasetError):
                edit_dataset(source, {"keep": ["photo-0"]}, root / "edited")
            self.assertFalse((root / "edited").exists())

    def test_cancellation_and_parallel_transfer_failures_remove_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            before = _bytes(source)
            def cancel(event):
                if event["phase"] == "editing_dataset":
                    raise KeyboardInterrupt("Cancelled")
            with self.assertRaises(KeyboardInterrupt):
                edit_dataset(source, {}, root / "cancelled", progress_callback=cancel, workers=3)
            with patch("ipde.dataset_edit._share_array_file", side_effect=OSError("Disk full")):
                with self.assertRaisesRegex(OSError, "Disk full"):
                    edit_dataset(source, {}, root / "failed", workers=3)
            self.assertEqual(list(root.iterdir()), [source])
            self.assertEqual(before, _bytes(source))

    def test_changed_source_manifest_and_raced_in_destination_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            original = (source / "dataset.json").read_bytes()
            changed = False
            def change(event):
                nonlocal changed
                if event["phase"] == "editing_dataset" and not changed:
                    changed = True
                    _rewrite(source, lambda data: data.update(name="Changed"))
            with self.assertRaisesRegex(DatasetError, "changed during editing"):
                edit_dataset(source, {}, root / "changed", progress_callback=change)
            (source / "dataset.json").write_bytes(original)
            output = root / "raced"
            def race(temporary, destination):
                destination.mkdir()
                (destination / "unrelated").write_text("preserve")
                _publish_new_directory(temporary, destination)
            with patch("ipde.dataset_edit._publish_new_directory", side_effect=race):
                with self.assertRaisesRegex(DatasetError, "already exists"):
                    edit_dataset(source, {}, output)
            self.assertEqual((output / "unrelated").read_text(), "preserve")
            self.assertFalse(list(root.glob(".raced-*")))
            self.assertFalse(list(root.glob(".changed-*")))

    def test_cli_has_structured_success_errors_and_all_addition_source_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            added, _ = _dataset(root / "added", offset=100)
            edits = root / "edits.json"
            edits.write_text(json.dumps({"keep": ["photo-0", "photo-3"], "splits": {"photo-0": "validation"}}))
            output, stdout, stderr = root / "edited", io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(["--json", "edit-dataset", str(source), "--edits-json", str(edits), "--add-dataset", str(added),
                               "--output-dir", str(output), "--workers", "2"])
            self.assertEqual(result, 0, stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["summary"]["samples"], 6)
            self.assertIn("IPDE_EVENT", stderr.getvalue())
            stdout = io.StringIO()
            edits.write_text(json.dumps({"keep": ["missing"]}))
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                result = main(["--json", "edit-dataset", str(source), "--edits-json", str(edits), "--output-dir", str(root / "error")])
            self.assertEqual(result, 1)
            self.assertIn("Unknown base", json.loads(stdout.getvalue())["error"])

    def test_cli_termination_unwinds_staging_and_restores_signal_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root / "source")
            edits = root / "edits.json"
            edits.write_text("{}")
            stdout = io.StringIO()
            previous = signal.getsignal(signal.SIGTERM)
            def terminate(event):
                if event["phase"] == "editing_dataset":
                    signal.raise_signal(signal.SIGTERM)
            with patch("ipde.trainer_cli._progress", side_effect=terminate), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                result = main(["--json", "edit-dataset", str(source), "--edits-json", str(edits), "--output-dir", str(root / "cancelled"), "--workers", "1"])
            self.assertEqual(result, 1)
            self.assertIn("cancelled", json.loads(stdout.getvalue())["error"])
            self.assertFalse((root / "cancelled").exists())
            self.assertFalse(list(root.glob(".cancelled-*")))
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


class FastDatasetUpdateTests(unittest.TestCase):
    def test_workspace_hides_internal_added_photo_staging_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "datasets").mkdir()
            base, _ = _dataset(workspace / "datasets" / "dataset")
            _dataset(workspace / "datasets" / ".added-20261004-123456", offset=100)
            report = workspace_report(workspace)
            self.assertEqual([entry["path"] for entry in report["datasets"]], [str(base.resolve())])

    def test_workspace_refresh_uses_active_counts_and_cached_size_without_payload_walk(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "datasets").mkdir()
            root, _ = _dataset(workspace / "datasets" / "dataset")
            _rewrite(root, lambda data: data.update(summary={"array_storage_bytes": 32 * 1024 ** 3}))
            apply_dataset_edits(root, {"keep": ["photo-0", "photo-3"]})
            real_rglob = Path.rglob
            def checkpoint_metadata_only(path, pattern, **kwargs):
                if pattern != "*.pth.json":
                    raise AssertionError("Workspace refresh walked scientific payload files")
                return real_rglob(path, pattern, **kwargs)
            with patch.object(Path, "rglob", checkpoint_metadata_only):
                report = workspace_report(workspace)
            entry = report["datasets"][0]
            self.assertEqual((entry["sample_count"], entry["source_count"], entry["excluded_count"]), (2, 2, 2))
            self.assertEqual((entry["train_count"], entry["validation_count"]), (1, 1))
            self.assertTrue(entry["storage_bytes_known"])
            self.assertGreater(entry["storage_bytes"], 32 * 1024 ** 3)

    def test_remove_restore_and_split_touch_only_the_same_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            payloads = {path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
                        for path in root.iterdir() if path.name != "dataset.json"}
            original_open = Path.open

            def manifest_only(path, *args, **kwargs):
                if path.suffix in {".npy", ".npz"}:
                    raise AssertionError("Metadata edit opened a scientific payload")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", manifest_only), \
                 patch("ipde.dataset_edit.sha256_file", side_effect=AssertionError("Payload hash scan")), \
                 patch("ipde.dataset_edit.load_dataset", side_effect=AssertionError("Payload validation")), \
                 patch("ipde.dataset_edit._verified_array", side_effect=AssertionError("Payload decode")), \
                 patch("ipde.dataset_edit._share_array_file", side_effect=AssertionError("Payload duplication")):
                report = apply_dataset_edits(root, {"keep": ["photo-0", "photo-3"], "splits": {"photo-0": "validation"}})
                self.assertEqual(report["dataset_path"], str(root.resolve()))
                self.assertEqual(report["summary"]["samples"], 2)
                self.assertEqual(report["excluded_samples"], 2)
                review = review_dataset(root)
                self.assertEqual(len(review["samples"]), 4)
                self.assertEqual({sample["id"] for sample in review["samples"] if sample["included"]}, {"photo-0", "photo-3"})
                restored = apply_dataset_edits(root, {"keep": [f"photo-{index}" for index in range(4)]},
                                               expected_manifest_sha256=report["manifest_sha256"])
                self.assertEqual(restored["summary"]["samples"], 4)
                self.assertEqual(restored["edit_revision"], 2)
            self.assertEqual({path.name for path in root.iterdir()}, {"dataset.json", *payloads})
            self.assertEqual(payloads, {path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
                                       for path in root.iterdir() if path.name != "dataset.json"})
            saved = json.loads((root / "dataset.json").read_text())
            self.assertEqual(saved["name"], "Original")
            self.assertEqual(saved["custom_dataset_metadata"], {"precision": "raw"})
            self.assertEqual(saved["samples"][0]["split"], "validation")
            self.assertEqual(restored["manifest_sha256"], hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest())

    def test_all_removed_remains_restorable_after_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            report = apply_dataset_edits(root, {"keep": []})
            self.assertEqual(report["summary"]["samples"], 0)
            reopened = review_dataset(root)
            self.assertEqual(reopened["summary"]["samples"], 0)
            self.assertTrue(all(not entry["included"] for entry in reopened["samples"]))
            apply_dataset_edits(root, {"keep": ["photo-1"]}, expected_manifest_sha256=reopened["manifest_sha256"])
            self.assertEqual(review_dataset(root)["summary"]["samples"], 1)

    def test_automatic_percentage_endpoints_and_related_capture_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            _rewrite(root, lambda data: data["samples"][1].update(source_sha256="source-0"))
            all_validation = apply_dataset_edits(root, {"validation_fraction": 1.0, "seed": 91})
            self.assertEqual(all_validation["summary"]["validation_samples"], 4)
            self.assertEqual(all_validation["summary"]["train_samples"], 0)
            zero_validation = apply_dataset_edits(root, {"validation_fraction": 0.0})
            self.assertEqual(zero_validation["summary"]["train_samples"], 4)
            self.assertEqual(zero_validation["summary"]["validation_samples"], 0)
            apply_dataset_edits(root, {"validation_fraction": 0.05, "seed": 7})
            first = json.loads((root / "dataset.json").read_text())
            self.assertEqual(first["validation_fraction"], 0.05)
            self.assertEqual(first["split_seed"], 7)
            self.assertEqual(first["samples"][0]["split"], first["samples"][1]["split"])
            self.assertEqual(first["samples"][0]["group_id"], first["samples"][1]["group_id"])
            apply_dataset_edits(root, {"validation_fraction": 0.05, "seed": 7})
            second = json.loads((root / "dataset.json").read_text())
            self.assertEqual([sample["split"] for sample in first["samples"]], [sample["split"] for sample in second["samples"]])

    def test_saved_automatic_fraction_applies_to_restored_and_new_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            apply_dataset_edits(root, {"keep": [], "validation_fraction": 1.0, "seed": 29})
            restored = apply_dataset_edits(root, {"keep": [f"photo-{index}" for index in range(4)]})
            self.assertEqual(restored["validation_fraction"], 1.0)
            self.assertEqual(restored["split_seed"], 29)
            self.assertEqual((restored["summary"]["train_samples"], restored["summary"]["validation_samples"]), (0, 4))
            added, _ = _dataset(directory / "new-photos", offset=100)
            imported = apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(imported["validation_fraction"], 1.0)
            self.assertEqual((imported["summary"]["train_samples"], imported["summary"]["validation_samples"]), (0, 8))
            saved = json.loads((root / "dataset.json").read_text())
            all_ids = [sample["id"] for sample in saved["samples"]]
            apply_dataset_edits(root, {"validation_fraction": 0.0, "seed": 31})
            apply_dataset_edits(root, {"keep": []})
            restored = apply_dataset_edits(root, {"keep": all_ids})
            self.assertEqual((restored["summary"]["train_samples"], restored["summary"]["validation_samples"]), (8, 0))
            self.assertEqual(restored["split_seed"], 31)

    def test_automatic_membership_resplit_uses_saved_seed_and_manual_splits_disable_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            apply_dataset_edits(root, {"validation_fraction": 0.5, "seed": 83})
            apply_dataset_edits(root, {"keep": ["photo-0", "photo-1"]})
            membership_split = json.loads((root / "dataset.json").read_text())
            self.assertEqual(membership_split["split_seed"], 83)
            explicit = apply_dataset_edits(root, {"validation_fraction": 0.5, "seed": 83})
            self.assertEqual(explicit["summary"]["train_samples"], 1)
            self.assertEqual(explicit["summary"]["validation_samples"], 1)
            self.assertEqual([sample["split"] for sample in membership_split["samples"]],
                             [sample["split"] for sample in json.loads((root / "dataset.json").read_text())["samples"]])
            manual = apply_dataset_edits(root, {"splits": {"photo-0": "validation", "photo-1": "validation"}})
            self.assertIsNone(manual["validation_fraction"])
            before_restore = json.loads((root / "dataset.json").read_text())
            restored = apply_dataset_edits(root, {"keep": [f"photo-{index}" for index in range(4)]})
            self.assertIsNone(restored["validation_fraction"])
            after_restore = json.loads((root / "dataset.json").read_text())
            self.assertEqual([sample["split"] for sample in before_restore["samples"]],
                             [sample["split"] for sample in after_restore["samples"]])

    def test_stale_snapshot_rejected_without_touching_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            old_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            apply_dataset_edits(root, {"keep": ["photo-0"]}, expected_manifest_sha256=old_hash)
            current = (root / "dataset.json").read_bytes()
            with self.assertRaisesRegex(DatasetError, "changed since.*Reload"):
                apply_dataset_edits(root, {"keep": ["photo-1"]}, expected_manifest_sha256=old_hash)
            self.assertEqual((root / "dataset.json").read_bytes(), current)

    def test_completed_operation_replay_acknowledges_exact_snapshot_without_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            old_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            edits = {"keep": ["photo-0", "photo-3"], "validation_fraction": 1.0, "seed": 41}
            committed = apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="save-1")
            self.assertFalse(committed["already_applied"])
            manifest_before = (root / "dataset.json").read_bytes()
            inode_before = (root / "dataset.json").stat().st_ino
            actual_open = Path.open
            def metadata_only(path, *args, **kwargs):
                if path.suffix in {".npy", ".npz"}:
                    raise AssertionError("Recovery read a scientific payload")
                return actual_open(path, *args, **kwargs)
            with patch.object(Path, "open", metadata_only), \
                 patch("ipde.dataset_edit.os.replace", side_effect=AssertionError("Recovery rewrote manifest")), \
                 patch("ipde.dataset_edit._array_path", side_effect=AssertionError("Recovery inspected arrays")):
                replay = apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="save-1")
            self.assertTrue(replay["already_applied"])
            self.assertEqual(replay["manifest_sha256"], committed["manifest_sha256"])
            self.assertEqual(replay["edit_revision"], committed["edit_revision"])
            self.assertEqual(replay["summary"], committed["summary"])
            self.assertEqual((root / "dataset.json").read_bytes(), manifest_before)
            self.assertEqual((root / "dataset.json").stat().st_ino, inode_before)
            proof = replay["dataset_update"]
            self.assertEqual(proof["previous_manifest_sha256"], old_hash)
            self.assertEqual(proof["operation_id"], "save-1")
            self.assertEqual(len(proof["committed_content_sha256"]), 64)

    def test_crash_after_commit_recovers_addition_even_after_temporary_source_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            added, _ = _dataset(directory / "added", offset=100)
            old_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            actual_replace = os.replace
            def crash_after_commit(source, destination):
                actual_replace(source, destination)
                raise RuntimeError("crash before save acknowledgement")
            with patch("ipde.dataset_edit.os.replace", side_effect=crash_after_commit), self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                apply_dataset_edits(root, {}, add_datasets=[added], expected_manifest_sha256=old_hash, operation_id="save-add")
            committed = (root / "dataset.json").read_bytes()
            shutil.rmtree(added)
            with patch("ipde.dataset_edit._array_path", side_effect=AssertionError("Recovery touched payload")), \
                 patch("ipde.dataset_edit.os.link", side_effect=AssertionError("Recovery duplicated files")):
                report = apply_dataset_edits(root, {}, add_datasets=[added], expected_manifest_sha256=old_hash, operation_id="save-add")
            self.assertTrue(report["already_applied"])
            self.assertEqual(report["summary"]["samples"], 8)
            self.assertEqual(report["added_samples"], 4)
            self.assertEqual(report["edit_revision"], 1)
            self.assertEqual((root / "dataset.json").read_bytes(), committed)
            self.assertEqual(load_dataset(root)["summary"]["samples"], 8)
            # A newer recovered draft rebases to the verified commit hash and
            # receives its own operation ID; it cannot reuse the first ID.
            updated = apply_dataset_edits(root, {"splits": {"photo-0": "validation"}},
                                          expected_manifest_sha256=report["manifest_sha256"], operation_id="save-newer")
            self.assertEqual(updated["edit_revision"], 2)

    def test_operation_recovery_rejects_different_requests_and_intervening_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            old_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            edits = {"keep": ["photo-0"]}
            committed = apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="save-a")
            current = (root / "dataset.json").read_bytes()
            with self.assertRaisesRegex(DatasetError, "different edits"):
                apply_dataset_edits(root, {"keep": ["photo-1"]}, expected_manifest_sha256=old_hash, operation_id="save-a")
            with self.assertRaisesRegex(DatasetError, "different dataset snapshot"):
                apply_dataset_edits(root, edits, expected_manifest_sha256=committed["manifest_sha256"], operation_id="save-a")
            with self.assertRaisesRegex(DatasetError, "changed since"):
                apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="unrelated-save")
            self.assertEqual((root / "dataset.json").read_bytes(), current)
            _rewrite(root, lambda data: data.update(name="External rename retaining audit"))
            changed = (root / "dataset.json").read_bytes()
            with self.assertRaisesRegex(DatasetError, "changed after.*committed"):
                apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="save-a")
            self.assertEqual((root / "dataset.json").read_bytes(), changed)
            # A real subsequent editor replaces the audit. Old requests still
            # fail ordinary optimistic concurrency rather than bypassing it.
            apply_dataset_edits(root, {"keep": ["photo-2"]})
            with self.assertRaisesRegex(DatasetError, "changed since"):
                apply_dataset_edits(root, edits, expected_manifest_sha256=old_hash, operation_id="save-a")

    def test_recoverable_operations_require_a_valid_unique_id_and_base_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            before = (root / "dataset.json").read_bytes()
            old_hash = hashlib.sha256(before).hexdigest()
            for operation in ("", "save with spaces", "bad/slash", "x" * 129, 4):
                with self.subTest(operation=operation), self.assertRaisesRegex(DatasetError, "operation_id"):
                    apply_dataset_edits(root, {}, expected_manifest_sha256=old_hash, operation_id=operation)
            with self.assertRaisesRegex(DatasetError, "requires expected_manifest_sha256"):
                apply_dataset_edits(root, {}, operation_id="without-base-hash")
            self.assertEqual((root / "dataset.json").read_bytes(), before)

    def test_running_training_snapshot_permits_edits_but_cleanup_and_writers_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            with resource_lock(root, shared=True):
                report = apply_dataset_edits(root, {"keep": ["photo-0"]})
            self.assertEqual(report["summary"]["samples"], 1)
            before = (root / "dataset.json").read_bytes()
            with resource_lock(root), self.assertRaisesRegex(RuntimeError, "in use"):
                apply_dataset_edits(root, {"keep": ["photo-1"]})
            with resource_lock(root / "dataset.json"), self.assertRaisesRegex(RuntimeError, "in use"):
                apply_dataset_edits(root, {"keep": ["photo-1"]})
            self.assertEqual((root / "dataset.json").read_bytes(), before)

    def test_new_payloads_link_once_existing_arrays_do_not_move_and_source_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, originals = _dataset(directory / "dataset")
            added, new_samples = _dataset(directory / "new-photos", offset=100)
            _rewrite(added, lambda data: data["samples"][3].update(excluded=True))
            original_records = copy.deepcopy(list(_array_records(originals)))
            original_open = Path.open
            def manifest_only(path, *args, **kwargs):
                if path.suffix in {".npy", ".npz"}:
                    raise AssertionError("Attaching prepared photos read scientific payload bytes")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", manifest_only), patch("shutil.copyfile", side_effect=AssertionError("Payload copy")):
                first = apply_dataset_edits(root, {}, add_datasets=[added])
                again = apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(first["added_samples"], 3)
            self.assertEqual(again["added_samples"], 0)
            saved = json.loads((root / "dataset.json").read_text())
            self.assertEqual(list(_array_records(saved["samples"][:4])), original_records)
            for entry, original in zip(saved["samples"][4:], new_samples[:3]):
                for new, old in zip(_array_records(entry), _array_records(original)):
                    self.assertTrue((root / new["path"]).samefile(added / old["path"]))
            shutil.rmtree(added)
            self.assertEqual(load_dataset(root)["summary"]["samples"], 7)

    def test_connected_additions_inherit_existing_split_and_idempotent_retry_does_not_reset_it(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            added, _ = _dataset(directory / "new-targets")
            _rewrite(root, lambda data: data["samples"][0].update(split="validation"))
            # The duplicate generated photo starts as Train in its independent
            # temporary dataset. It must follow the existing Validation group.
            first = apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(first["added_samples"], 4)
            saved = json.loads((root / "dataset.json").read_text())
            matches = [sample for sample in saved["samples"] if sample["source_sha256"] == "source-0"]
            self.assertEqual(len(matches), 2)
            self.assertEqual({sample["split"] for sample in matches}, {"validation"})
            moved = apply_dataset_edits(root, {"splits": {"photo-0": "train"}})
            self.assertEqual(moved["added_samples"], 0)
            retried = apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(retried["added_samples"], 0)
            final = json.loads((root / "dataset.json").read_text())
            self.assertEqual({sample["split"] for sample in final["samples"] if sample["source_sha256"] == "source-0"}, {"train"})

    def test_added_connector_cannot_silently_merge_conflicting_existing_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            added, _ = _dataset(directory / "connector", offset=100)
            _rewrite(root, lambda data: (data["samples"][0].update(burst_ids=["one"]), data["samples"][3].update(burst_ids=["two"])))
            _rewrite(added, lambda data: data["samples"][0].update(burst_ids=["one", "two"]))
            before = _bytes(root)
            with self.assertRaisesRegex(DatasetError, "different existing splits"):
                apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(before, _bytes(root))
            apply_dataset_edits(root, {"splits": {"photo-0": "validation"}}, add_datasets=[added])
            saved = json.loads((root / "dataset.json").read_text())
            self.assertEqual(saved["samples"][0]["group_id"], saved["samples"][3]["group_id"])
            self.assertEqual(saved["samples"][0]["split"], saved["samples"][3]["split"])

    def test_queued_new_photos_persist_before_inference_and_clear_after_attach(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            pending = str(directory / "not-yet-generated.HEIC")
            queued = apply_dataset_edits(root, {"pending_photos": [pending]})
            self.assertEqual(queued["pending_photos"], [pending])
            self.assertEqual(review_dataset(root)["pending_photos"], [pending])
            added, _ = _dataset(directory / "added", offset=100)
            _rewrite(added, lambda data: data["samples"][0].update(source_path=pending))
            report = apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(report["pending_photos"], [])

    def test_completed_photo_alias_leaves_only_unfinished_pending_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            photos = directory / "photos"
            photos.mkdir()
            photo = photos / "capture.HEIC"
            photo.write_bytes(b"raw source must never be opened by metadata editing")
            alias = directory / "photo-alias"
            alias.symlink_to(photos, target_is_directory=True)
            pending = str(alias / photo.name)
            unfinished = str(alias / "absent-for-recovery.HEIC")
            apply_dataset_edits(root, {"pending_photos": [pending, unfinished]})
            added, _ = _dataset(directory / "added", offset=100)
            _rewrite(added, lambda data: data["samples"][0].update(source_path=str(photo.resolve())))
            before = _bytes(added)
            original_open = Path.open

            def manifest_only(path, *args, **kwargs):
                if path.suffix.lower() in {".npy", ".npz", ".heic"}:
                    raise AssertionError("Pending identity comparison opened a payload")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", manifest_only), \
                 patch("ipde.dataset_edit.sha256_file", side_effect=AssertionError("Payload hash scan")), \
                 patch("ipde.dataset_edit.load_dataset", side_effect=AssertionError("Payload validation")), \
                 patch("ipde.dataset_edit._verified_array", side_effect=AssertionError("Payload decode")):
                report = apply_dataset_edits(root, {}, add_datasets=[added])
                self.assertEqual(report["pending_photos"], [unfinished])
                self.assertEqual(review_dataset(root)["pending_photos"], [unfinished])
                # Reattaching an already imported generation clears the same
                # aliased queue entry without creating additional samples.
                apply_dataset_edits(root, {"pending_photos": [pending, unfinished]})
                retry = apply_dataset_edits(root, {}, add_datasets=[added])
                self.assertEqual(retry["added_samples"], 0)
                self.assertEqual(retry["pending_photos"], [unfinished])
            self.assertEqual(_bytes(added), before)
            self.assertEqual(photo.read_bytes(), b"raw source must never be opened by metadata editing")
            self.assertFalse(Path(unfinished).exists())

    def test_atomic_publish_failure_preserves_manifest_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            before = _bytes(root)
            with patch("ipde.dataset_edit.os.replace", side_effect=OSError("simulated filesystem failure")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    apply_dataset_edits(root, {"keep": ["photo-0"]})
            self.assertEqual(before, _bytes(root))
            self.assertFalse(list(root.glob(".dataset-update-*")))

    def test_failed_addition_rolls_back_links_and_postcommit_cancellation_preserves_them(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, _ = _dataset(directory / "dataset")
            added, _ = _dataset(directory / "added", offset=100)
            before = _bytes(root)
            actual_link = os.link
            link_count = 0
            def fail_after_first(source, destination):
                nonlocal link_count
                link_count += 1
                if link_count == 2:
                    raise OSError("simulated interrupted addition")
                actual_link(source, destination)
            with patch("ipde.dataset_edit.os.link", side_effect=fail_after_first), self.assertRaisesRegex(DatasetError, "without copying"):
                apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(before, _bytes(root))
            self.assertFalse((root / "arrays").exists())
            actual_replace = os.replace
            def cancel_after_commit(source, destination):
                actual_replace(source, destination)
                raise RuntimeError("cancel arrived after commit")
            with patch("ipde.dataset_edit.os.replace", side_effect=cancel_after_commit), self.assertRaisesRegex(RuntimeError, "after commit"):
                apply_dataset_edits(root, {}, add_datasets=[added])
            self.assertEqual(load_dataset(root)["summary"]["samples"], 8)
            self.assertFalse(list(root.glob(".dataset-update-*")))

    def test_invalid_metadata_updates_are_rejected_without_a_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            before = (root / "dataset.json").read_bytes()
            invalid = [None, {"unknown": True}, {"keep": ["absent"]}, {"splits": {"photo-0": []}},
                       {"validation_fraction": 1.1}, {"validation_fraction": True}, {"seed": False},
                       {"pending_photos": ["relative.HEIC"]}, {"pending_photos": ["/photo", "/photo"]},
                       {"splits": {"photo-0": "train"}, "validation_fraction": .2}]
            for edits in invalid:
                with self.subTest(edits=edits), self.assertRaises(DatasetError):
                    apply_dataset_edits(root, edits)
            self.assertEqual(before, (root / "dataset.json").read_bytes())

    def test_cli_same_path_update_has_structured_revision_and_conflict_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            original_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            edits = Path(directory) / "edits.json"
            edits.write_text(json.dumps({"keep": ["photo-0"], "validation_fraction": 1.0}))
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = main(["--json", "update-dataset", str(root), "--edits-json", str(edits),
                             "--expected-manifest-sha256", original_hash])
            self.assertEqual(code, 0, stdout.getvalue())
            response = json.loads(stdout.getvalue())
            self.assertEqual(response["dataset_path"], str(root.resolve()))
            self.assertEqual(response["summary"]["validation_samples"], 1)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = main(["--json", "update-dataset", str(root), "--edits-json", str(edits),
                             "--expected-manifest-sha256", original_hash])
            self.assertEqual(code, 1)
            self.assertIn("changed since", json.loads(stdout.getvalue())["error"])

    def test_cli_operation_id_replays_completed_save_with_structured_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _dataset(Path(directory) / "dataset")
            base_hash = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
            edits = Path(directory) / "edits.json"
            edits.write_text(json.dumps({"validation_fraction": 1.0, "seed": 61}))
            reports = []
            for _ in range(2):
                stdout = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    code = main(["--json", "update-dataset", str(root), "--edits-json", str(edits),
                                 "--expected-manifest-sha256", base_hash, "--operation-id", "gui-save-123"])
                self.assertEqual(code, 0, stdout.getvalue())
                reports.append(json.loads(stdout.getvalue()))
            self.assertFalse(reports[0]["already_applied"])
            self.assertTrue(reports[1]["already_applied"])
            self.assertEqual(reports[0]["manifest_sha256"], reports[1]["manifest_sha256"])
            self.assertEqual(reports[0]["edit_revision"], reports[1]["edit_revision"])


if __name__ == "__main__":
    unittest.main()
