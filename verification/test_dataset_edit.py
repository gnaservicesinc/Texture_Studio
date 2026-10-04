"""Image edits preserve source bytes and enforce connected split boundaries."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.dataset import DatasetError, load_dataset
from ipde.dataset_edit import edit_dataset
from ipde.dataset_review import _array_records, _publish_new_directory, review_dataset
from ipde.trainer_cli import main


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


if __name__ == "__main__":
    unittest.main()
