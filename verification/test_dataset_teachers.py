"""Selected-photo teacher changes preserve data, groups and atomic membership."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.dataset_edit import apply_dataset_edits
from ipde.dataset_review import _array_records, review_dataset
from ipde.dataset_teachers import disable_teacher, generate_teacher
from ipde.formats import sha256_file
from ipde.learned_depth import LearnedDepthConfig
from ipde.resource_lock import resource_lock
from ipde.trainer_cli import main
from verification.test_display_teacher_source import _capture, _prediction


def _relative(rgb):
    return np.linspace(.2, 1.8, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])


class DatasetTeacherTests(unittest.TestCase):
    def _dataset(self, root, model="depthpro"):
        sources = [root / f"capture-{index}.heic" for index in range(3)]
        results = {}
        for source in sources:
            source.write_bytes(b"original spatial capture")
            rgb = _capture(source, display_shape=(96, 128)).assets[-1].array
            relative = _relative(rgb)
            results[str(source)] = _prediction(rgb,
                units="meters" if model == "depthpro" else "relative_inverse_depth",
                depth=1. / (relative * .25 + .1) if model == "depthpro" else relative)
        with patch("ipde.dataset.discover_file", side_effect=lambda source: _capture(source, display_shape=(96, 128))), \
                redirect_stderr(io.StringIO()):
            manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                teacher=LearnedDepthConfig(model=model), workers=1, skip_bad_photos=False), display_teacher_results=results)
        return root / "dataset", manifest

    def _factory(self, calls, closed, *, hook=None):
        class Predictor:
            def __init__(self, config):
                self.config = config
            def __call__(self, rgb, **kwargs):
                calls.append((self.config.model, rgb.shape, kwargs))
                if hook:
                    hook()
                relative = _relative(rgb)
                depth = 1. / (relative * .25 + .1) if self.config.model == "depthpro" else (
                    1. / relative if self.config.model == "depth-anything-3" else relative)
                return _prediction(rgb, units="meters" if self.config.model == "depthpro" else (
                    "relative_depth" if self.config.model == "depth-anything-3" else "relative_inverse_depth"), depth=depth)
            def close(self):
                closed.append(self.config.model)
        return Predictor

    def test_selected_only_deduplicated_missing_generation_reuses_meter_anchor_and_offline_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, before = self._dataset(Path(directory))
            original_arrays = {record["path"]: (dataset / record["path"]).read_bytes() for record in _array_records(before["samples"])}
            for sample in before["samples"]:
                Path(sample["source_path"]).unlink()  # Stored display RGB is sufficient.
            calls, closed = [], []
            first, second = (sample["id"] for sample in before["samples"][:2])
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                result = generate_teacher(dataset, [first, first, second], LearnedDepthConfig(model="depth-anything-3"))
            after = load_dataset(dataset)
            self.assertEqual(len(calls), 2)
            self.assertEqual(closed, ["depth-anything-3"])
            self.assertTrue(all(model == "depth-anything-3" and shape == (96, 128, 3) and kwargs["reference_label"] == "display"
                                for model, shape, kwargs in calls))
            self.assertEqual(len(result["generated_samples"]), 2)
            self.assertEqual(after["samples"][:3], before["samples"])
            for sample, base in zip(after["samples"][3:], before["samples"][:2]):
                for key in ("rgb", "right_rgb", "display_rgb", "raw_assets", "calibration", "group_id", "split", "source_sha256"):
                    self.assertEqual(sample[key], base[key])
                self.assertEqual(sample["training_target_choice"], "anchored_display_teacher")
                self.assertEqual(sample["anchored_display_teacher"]["units"], "meters")
                self.assertTrue(sample["teacher_generation"]["metric_anchor_reused"])
                self.assertNotIn("display_metric_anchor", sample)
                self.assertEqual(sample["metric_anchor_provenance"]["target_array_sha256"], base["display_teacher"]["target"]["array_sha256"])
                self.assertEqual(sample["display_teacher"]["target"], sample["anchored_display_teacher"]["target"])
                self.assertFalse(sample["original_teacher_provenance"]["payload_retained"])
                self.assertIn("valid_mask", sample["anchored_display_teacher"])
            self.assertTrue(result["training_eligibility"]["trainable"])
            self.assertFalse(result["warnings"])
            distinct_targets = {sample["display_teacher"]["target"]["path"] for sample in after["samples"][3:]}
            self.assertEqual(len(list((dataset / "arrays" / "teachers").rglob("*.exr"))), len(distinct_targets))
            self.assertEqual(original_arrays, {name: (dataset / name).read_bytes() for name in original_arrays})

    def test_existing_variant_enable_is_noop_or_metadata_only_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            sample = manifest["samples"][0]
            digest = sha256_file(dataset / "dataset.json")
            with patch("ipde.learned_depth.LearnedDepthPredictor") as predictor:
                result = generate_teacher(dataset, [sample["id"]], LearnedDepthConfig())
                predictor.assert_not_called()
            self.assertEqual(result["manifest_sha256"], digest)
            manifest["samples"][0]["excluded"] = True
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            with patch("ipde.learned_depth.LearnedDepthPredictor") as predictor:
                result = generate_teacher(dataset, [sample["id"]], LearnedDepthConfig())
                predictor.assert_not_called()
            self.assertEqual(result["reenabled_samples"], [sample["id"]])
            after = load_dataset(dataset)
            self.assertFalse(after["samples"][0]["excluded"])
            self.assertEqual(after["samples"][0]["split"], sample["split"])
            self.assertFalse(list(dataset.parent.glob(".teacher-generation-*")))

    def test_off_deletes_unused_depth_keeps_core_and_restore_regenerates_same_row(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            # Distinct depth prevents the fixture's other photos sharing it.
            from ipde.array_storage import array_record
            sample = manifest["samples"][0]
            distinct = np.full((96, 128), 7., np.float32)
            record = array_record(dataset, dataset / "distinct.exr", distinct, storage="images")
            sample["teacher"]["target"] = sample["display_teacher"]["target"] = record
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            core = {item["path"]: (dataset / item["path"]).read_bytes() for key in ("rgb", "right_rgb", "display_rgb", "raw_assets")
                    for item in _array_records(sample[key])}
            result = disable_teacher(dataset, [sample["id"]], "depthpro")
            self.assertGreater(result["discarded_array_storage_bytes"], 0)
            self.assertFalse((dataset / "distinct.exr").exists())
            removed = load_dataset(dataset)["samples"][0]
            self.assertTrue(removed["teacher_payload_removed"])
            self.assertTrue(removed["excluded"])
            self.assertEqual(removed["teacher"]["target"], record)
            self.assertEqual(core, {name: (dataset / name).read_bytes() for name in core})
            with self.assertRaisesRegex(DatasetError, "Generate its teacher again"):
                apply_dataset_edits(dataset, {"keep": [entry["id"] for entry in manifest["samples"]]})
            self.assertEqual(review_dataset(dataset)["samples"][0]["labels"], [])
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                restored = generate_teacher(dataset, [sample["id"]], LearnedDepthConfig())
            after = load_dataset(dataset)
            self.assertEqual(len(calls), 1)
            self.assertEqual(restored["generated_samples"], [sample["id"]])
            self.assertEqual(len(after["samples"]), len(manifest["samples"]))
            self.assertNotIn("teacher_payload_removed", after["samples"][0])
            self.assertFalse(after["samples"][0]["excluded"])
            self.assertEqual(after["samples"][0]["split"], sample["split"])

    def test_shared_teacher_payload_is_retained_and_active_reader_refuses_off(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            sample = manifest["samples"][0]
            target = dataset / sample["teacher"]["target"]["path"]
            before = target.read_bytes()
            digest = sha256_file(dataset / "dataset.json")
            with resource_lock(dataset, shared=True), self.assertRaisesRegex(RuntimeError, "in use"):
                disable_teacher(dataset, [sample["id"]], "depthpro")
            self.assertEqual(sha256_file(dataset / "dataset.json"), digest)
            result = disable_teacher(dataset, [sample["id"]], "depthpro")
            self.assertEqual(result["discarded_array_storage_bytes"], 0)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(len(load_dataset(dataset)["samples"]), 3)

    def test_stale_unknown_and_concurrent_generation_do_not_publish_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            sample = manifest["samples"][0]
            with patch("ipde.learned_depth.LearnedDepthPredictor") as predictor:
                for ids, digest, text in ((["missing"], None, "Unknown"), ([sample["id"]], "0" * 64, "changed")):
                    with self.assertRaisesRegex(DatasetError, text):
                        generate_teacher(dataset, ids, LearnedDepthConfig(model="depth-anything-v2"), expected_manifest_sha256=digest)
                predictor.assert_not_called()
            changed = dict(manifest, name="concurrent edit")
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed,
                       hook=lambda: (dataset / "dataset.json").write_text(json.dumps(changed)))):
                with self.assertRaisesRegex(DatasetError, "changed during"):
                    generate_teacher(dataset, [sample["id"]], LearnedDepthConfig(model="depth-anything-v2"))
            self.assertEqual(json.loads((dataset / "dataset.json").read_text()), changed)
            self.assertFalse(list(dataset.parent.glob(".teacher-generation-*")))
            self.assertFalse((dataset / "arrays" / "teachers").exists())
            self.assertEqual(closed, ["depth-anything-v2"])

    def test_failed_inference_and_failed_manifest_publication_keep_original(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            before = (dataset / "dataset.json").read_bytes()
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)), \
                    patch("ipde.dataset_teachers.os.replace", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    generate_teacher(dataset, [manifest["samples"][0]["id"]], LearnedDepthConfig(model="depth-anything-v2"))
            self.assertEqual((dataset / "dataset.json").read_bytes(), before)
            self.assertFalse(list((dataset / "arrays" / "teachers").glob("*")))
            self.assertFalse(list(dataset.parent.glob(".teacher-generation-*")))
            self.assertEqual(load_dataset(dataset), manifest)

    def test_no_anchor_reports_units_and_explicit_anchor_loads_after_teacher_close(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory), model="depth-anything-v2")
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                result = generate_teacher(dataset, [manifest["samples"][0]["id"]], LearnedDepthConfig(model="depth-anything-3"))
            self.assertTrue(result["warnings"])
            self.assertEqual(closed, ["depth-anything-3"])
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                result = generate_teacher(dataset, [manifest["samples"][1]["id"]], LearnedDepthConfig(model="depth-anything-3"),
                                          metric_anchor=LearnedDepthConfig())
            after = load_dataset(dataset)
            added = next(sample for sample in after["samples"] if sample["id"] == result["generated_samples"][0])
            self.assertEqual(added["training_target_choice"], "anchored_display_teacher")
            self.assertEqual(closed[-2:], ["depth-anything-3", "depthpro"])

    def test_cli_generate_and_disable_dispatch_selected_model_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            ids = [manifest["samples"][0]["id"], manifest["samples"][1]["id"]]
            calls, closed = [], []
            output = io.StringIO()
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)), \
                    redirect_stdout(output), redirect_stderr(io.StringIO()):
                code = main(["--json", "generate-teacher", str(dataset), "--sample-id", ids[0], "--sample-id", ids[1],
                             "--model", "depth-anything-v2", "--expected-manifest-sha256", sha256_file(dataset / "dataset.json")])
            self.assertEqual(code, 0, output.getvalue())
            result = json.loads(output.getvalue())
            self.assertEqual(len(result["generated_samples"]), 2)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                code = main(["--json", "disable-teacher", str(dataset), "--sample-id", ids[0], "--model", "depth-anything-v2"])
            self.assertEqual(code, 0, output.getvalue())
            self.assertEqual(len(json.loads(output.getvalue())["removed_samples"]), 1)

    def test_lean_anchor_provenance_does_not_keep_depthpro_payload_after_all_are_off(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            ids = [sample["id"] for sample in manifest["samples"]]
            original_depth = dataset / manifest["samples"][0]["display_teacher"]["target"]["path"]
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                result = generate_teacher(dataset, [ids[0]], LearnedDepthConfig(model="depth-anything-3"))
            generated = result["generated_samples"][0]
            disable_teacher(dataset, ids, "depthpro")
            self.assertFalse(original_depth.exists())
            after = load_dataset(dataset)
            student_label = next(sample for sample in after["samples"] if sample["id"] == generated)
            self.assertFalse(student_label["excluded"])
            self.assertTrue((dataset / student_label["teacher"]["target"]["path"]).is_file())
            self.assertNotIn("display_metric_anchor", student_label)
            # Even an entirely inactive dataset retains a readable photo roster.
            disable_teacher(dataset, [generated], "depth-anything-3")
            final = load_dataset(dataset)
            self.assertTrue(all(sample["teacher_payload_removed"] and sample["excluded"] for sample in final["samples"]))
            self.assertFalse(list((dataset / "arrays" / "teachers").rglob("*.exr")))

    def test_archival_retention_keeps_raw_relative_and_anchor_records(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            manifest["storage_policy"]["retain_intermediates"] = True
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                result = generate_teacher(dataset, [manifest["samples"][0]["id"]], LearnedDepthConfig(model="depth-anything-v2"))
            added = next(sample for sample in load_dataset(dataset)["samples"] if sample["id"] == result["generated_samples"][0])
            self.assertEqual(added["display_teacher"]["units"], "relative_inverse_depth")
            self.assertEqual(added["anchored_display_teacher"]["units"], "meters")
            self.assertIn("display_metric_anchor", added)
            self.assertIn("native_target", added["display_teacher"])

    def test_off_manifest_failure_does_not_delete_a_file_or_change_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            before = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
            with patch("ipde.dataset_teachers.os.replace", side_effect=OSError("cannot save")):
                with self.assertRaisesRegex(OSError, "cannot save"):
                    disable_teacher(dataset, [manifest["samples"][0]["id"]], "depthpro")
            self.assertEqual(before, {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()})

    def test_inference_failure_releases_model_and_abandons_staged_planes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            before = (dataset / "dataset.json").read_bytes()
            calls, closed = [], []
            def failure():
                if len(calls) == 2:
                    raise RuntimeError("inference failed")
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed, hook=failure)):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    generate_teacher(dataset, [sample["id"] for sample in manifest["samples"]], LearnedDepthConfig(model="depth-anything-v2"))
            self.assertEqual(closed, ["depth-anything-v2"])
            self.assertEqual((dataset / "dataset.json").read_bytes(), before)
            self.assertFalse(list(dataset.parent.glob(".teacher-generation-*")))

    def test_cancellation_after_commit_preserves_newly_referenced_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            calls, closed = [], []
            def cancel(event):
                if event["event"] == "complete":
                    raise RuntimeError("cancelled after commit")
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                with self.assertRaisesRegex(RuntimeError, "cancelled after commit"):
                    generate_teacher(dataset, [manifest["samples"][0]["id"]], LearnedDepthConfig(model="depth-anything-v2"), progress_callback=cancel)
            self.assertEqual(len(load_dataset(dataset)["samples"]), 4)

    def test_cancel_after_commit_and_subsequent_metadata_edit_preserves_published_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            calls, closed = [], []
            def edit_then_cancel(event):
                if event["event"] == "complete":
                    path = dataset / "dataset.json"
                    updated = json.loads(path.read_bytes())
                    updated["edit_revision"] += 1
                    updated["name"] = "later metadata edit"
                    path.write_text(json.dumps(updated))
                    raise RuntimeError("cancelled after another edit")
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                with self.assertRaisesRegex(RuntimeError, "cancelled after another edit"):
                    generate_teacher(dataset, [manifest["samples"][0]["id"]], LearnedDepthConfig(model="depth-anything-v2"),
                                     progress_callback=edit_then_cancel)
            after = load_dataset(dataset)
            self.assertEqual(len(after["samples"]), 4)
            self.assertEqual(after["name"], "later metadata edit")

    def test_measured_reference_survives_off_and_restoration_from_another_teacher_row(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._dataset(Path(directory))
            from ipde.array_storage import array_record
            base = manifest["samples"][0]
            reference = {"units": "meters", "label_kind": "user_supplied_measured_reference",
                         "coordinate_reference": "spatial_left", "validity_policy": "positive_finite",
                         "target": array_record(dataset, dataset / "measured.npy", np.full((3, 4), 3., np.float64))}
            base["reference"] = reference
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            reference_bytes = (dataset / reference["target"]["path"]).read_bytes()
            calls, closed = [], []
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                generated = generate_teacher(dataset, [base["id"]], LearnedDepthConfig(model="depth-anything-3"))["generated_samples"][0]
            disable_teacher(dataset, [base["id"]], "depthpro")
            self.assertEqual((dataset / reference["target"]["path"]).read_bytes(), reference_bytes)
            with patch("ipde.learned_depth.LearnedDepthPredictor", self._factory(calls, closed)):
                restored = generate_teacher(dataset, [generated], LearnedDepthConfig())
            self.assertEqual(restored["generated_samples"], [base["id"]])
            row = next(sample for sample in load_dataset(dataset)["samples"] if sample["id"] == base["id"])
            self.assertEqual(row["reference"], reference)
            self.assertEqual((dataset / reference["target"]["path"]).read_bytes(), reference_bytes)


if __name__ == "__main__":
    unittest.main()
