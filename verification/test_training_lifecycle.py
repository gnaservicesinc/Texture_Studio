"""Real optimizer/checkpoint regressions for the v0.9 training lifecycle."""
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.training import TrainingError, TrainingIntegrityError, TrainingOptions, _native_training_dataset, train_dataset, training_step_plan
from test_training_progress import _dataset, _fake_raft_modules


def _mark_native(manifest, directory):
    manifest.update(generation_state="complete", generation_output_dir=str(directory))
    for sample in manifest["samples"]:
        sample["coordinate_reference"] = "spatial_left: exact decoded sample coordinates, no EXIF rotation"
        sample["calibration"].update(is_spatial_photo=True, metadata_source="macOS ImageIO container properties")
        sample["raw_assets"].extend({"kind": "spatial_view", "semantic_name": role, "storage": dict(sample[key])}
            for role, key in (("spatial_left", "rgb"), ("spatial_right", "right_rgb")))


class TrainingLifecycleTests(unittest.TestCase):
    def _run(self, root, dataset, destination, options, *, progress=None, original=None):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        original = original or root / "initial.pth"
        if not original.exists():
            torch.save(fixture(None).state_dict(), original)
        def resolve(raft_options):
            return root, raft_options.model or original, None
        with patch.dict(sys.modules, modules), patch("ipde.training.resolve_raft_resources", side_effect=resolve), redirect_stderr(io.StringIO()):
            return train_dataset(dataset, destination, options, progress_callback=progress)

    def _options(self, **values):
        return TrainingOptions(patch_size=64, iterations=1, device="cpu", learning_rate=1e-3,
            prefetch_samples=0, **values)

    def test_each_epoch_visits_every_training_entry_and_flushes_accumulation(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            events = []
            options = self._options(epochs=2, steps_per_update=2, checkpoint_schedule="end")
            report = self._run(root, dataset, root / "epochs.pth", options, progress=events.append)
            visits = [event for event in events if event["stage"] == "epoch_step" and event["status"] == "started"]
            for epoch in (1, 2):
                self.assertEqual(sorted(event["sample_id"] for event in visits if event["epoch"] == epoch),
                    ["sample-0", "sample-2", "sample-4"])
            updates = [event for event in events if event["stage"] == "epoch_step" and event["status"] == "finished"]
            self.assertEqual([event["images_in_update"] for event in updates], [2, 1, 2, 1])
            self.assertEqual((report["total_steps"], report["completed_images"], report["epochs_completed"]), (4, 6, 2))
            saved = torch.load(root / "epochs.pth", weights_only=True)
            for state in saved["ipde_resume"]["optimizer_state"]["state"].values():
                self.assertEqual(int(state["step"]), 4)
            self.assertEqual(training_step_plan(3, options)["total_steps"], 4)

    def test_exact_step_limit_and_bit_exact_resume_after_cooperative_stop(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=8)
            control = root / "control.json"
            options = self._options(limit_mode="steps", total_steps=5, steps_per_update=2,
                checkpoint_schedule="end", validation_schedule="checkpoint", validation_samples=1, control_file=control)
            def stop_after_two(event):
                if event["stage"] == "epoch_step" and event["status"] == "finished" and event["completed_steps"] == 2:
                    control.write_text(json.dumps({"command": "stop", "request_id": "stop-two"}))
            stopped = self._run(root, dataset, root / "stopped.pth", options, progress=stop_after_two)
            self.assertEqual((stopped["total_steps"], stopped["stop_reason"]), (2, "user_stopped"))
            control.unlink()
            resumed = self._run(root, dataset, root / "resumed.pth", replace(options, resume_from=root / "stopped.pth"))
            reference = self._run(root, dataset, root / "reference.pth", options)
            self.assertEqual((resumed["total_steps"], resumed["completed_images"]), (5, 10))
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(resumed["epoch_image_cursor"], 2)
            result = torch.load(root / "resumed.pth", weights_only=True)
            expected = torch.load(root / "reference.pth", weights_only=True)
            for key in result["state_dict"]:
                self.assertTrue(torch.equal(result["state_dict"][key], expected["state_dict"][key]), key)
            self.assertEqual(resumed["total_steps"], reference["total_steps"])

    def test_manual_and_every_n_steps_checkpoints_are_ready_to_use(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=8)
            control = root / "control.json"
            options = self._options(limit_mode="steps", total_steps=4, checkpoint_schedule="steps", checkpoint_every=3,
                validation_schedule="checkpoint", validation_samples=1, control_file=control)
            events = []
            def request_manual(event):
                events.append(event)
                if event["stage"] == "epoch_step" and event["status"] == "finished" and event["completed_steps"] == 1:
                    control.write_text(json.dumps({"command": "save", "request_id": "save-one"}))
            report = self._run(root, dataset, root / "model.pth", options, progress=request_manual)
            saved = [event for event in events if event["stage"] == "checkpoint_saved"]
            self.assertEqual([event["checkpoint_reason"] for event in saved], ["manual", "scheduled", "completed"])
            for event in saved:
                payload = torch.load(event["checkpoint_path"], weights_only=True)
                self.assertIn("state_dict", payload)
                self.assertIn("optimizer_state", payload["ipde_resume"])
                self.assertTrue(Path(event["checkpoint_path"] + ".json").is_file())
            epoch_validations = [event for event in events if event["stage"] == "epoch_validation"]
            self.assertEqual(epoch_validations, [])
            self.assertEqual(report["total_steps"], 4)

    def test_subsample_threshold_requires_full_held_out_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            options = self._options(epochs=3, validation_samples=1, early_stop_error=1.0, checkpoint_schedule="end")
            from ipde.training import _evaluate
            calls = []
            def sampled(torch, model, device, padder, samples, options, **keywords):
                result = _evaluate(torch, model, device, padder, samples, options, **keywords)
                calls.append(len(keywords["sample_indices"]))
                result["mean_absolute_flow_error_pixels"] = .5 if result["full_validation"] and len(calls) >= 5 else 2. if result["full_validation"] else .5
                return result
            with patch("ipde.training._evaluate", side_effect=sampled):
                report = self._run(root, dataset, root / "early.pth", options)
            self.assertEqual(calls, [1, 1, 3, 1, 3])
            self.assertEqual(report["stop_reason"], "early_stop_error_reached")
            self.assertEqual(report["epochs_completed"], 2)
            self.assertTrue(report["validation"]["full_validation"])
            self.assertEqual(report["validation"]["mean_absolute_flow_error_pixels"], .5)

    def test_native_startup_is_lazy_but_consumed_corruption_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            path = dataset / "dataset.json"
            manifest = json.loads(path.read_text())
            _mark_native(manifest, dataset)
            path.write_text(json.dumps(manifest))
            # An unrelated stored auxiliary is not an input to RAFT training.
            (dataset / manifest["samples"][0]["raw_assets"][0]["storage"]["path"]).write_bytes(b"unrelated unused auxiliary")
            events = []
            with patch("ipde.training._SamplePool.prepare", side_effect=AssertionError("native startup must stay lazy")):
                report = self._run(root, dataset, root / "native.pth", self._options(limit_mode="steps", total_steps=1,
                    validation_samples=1, checkpoint_schedule="end"), progress=events.append)
            self.assertEqual(report["total_steps"], 1)
            self.assertIn("Consumed-array", report["integrity_policy"])
            corruption = dataset / manifest["samples"][0]["rgb"]["path"]
            changed = np.load(corruption, allow_pickle=False)
            changed[0, 0, 0] ^= np.uint8(1)
            np.save(corruption, changed, allow_pickle=False)
            with self.assertRaisesRegex(TrainingIntegrityError, "checksum mismatch"):
                self._run(root, dataset, root / "corrupt.pth", self._options(epochs=1, checkpoint_schedule="end"))
            self.assertNotIn("sample-0", [sample["sample_id"] for sample in report["excluded_samples"]])

    def test_native_composed_curated_compacted_and_edited_sets_use_lazy_preflight(self):
        from ipde.dataset import load_dataset
        from ipde.dataset_collection import CollectionOptions, compose_datasets, compress_dataset
        from ipde.dataset_edit import apply_dataset_edits, edit_dataset
        from ipde.dataset_review import curate_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _dataset(root, count=6)
            manifest_path = source / "dataset.json"
            original = json.loads(manifest_path.read_text())
            _mark_native(original, source)
            # Historical native sets had no generation markers. Their original
            # raw policy plus per-photo provenance must survive descendants.
            original.pop("generation_output_dir")
            original.pop("generation_state")
            original["precision_policy"] = "Raw samples/auxiliaries preserved bit-for-bit in NPY; no normalization, gamma, or resampling"
            manifest_path.write_text(json.dumps(original))
            composed, curated, compacted, edited = (root / name for name in ("composed", "curated", "compacted", "edited"))
            with redirect_stderr(io.StringIO()):
                compose_datasets([source], composed, CollectionOptions(validation_fraction=.5), workers=1)
                ids = [sample["id"] for sample in json.loads((composed / "dataset.json").read_text())["samples"]]
                curate_dataset(composed, ids, curated, workers=1)
                compress_dataset(curated, compacted, workers=1)
                edit_dataset(compacted, {}, edited, workers=1)
                apply_dataset_edits(edited, {"keep": ids})
            shutil.rmtree(source)
            for index, dataset in enumerate((composed, curated, compacted, edited)):
                with self.subTest(dataset=dataset.name):
                    manifest = json.loads((dataset / "dataset.json").read_text())
                    self.assertNotIn("generation_output_dir", manifest)
                    self.assertTrue(_native_training_dataset(manifest))
                    with patch("ipde.training.load_dataset", wraps=load_dataset) as checking, patch(
                            "ipde.training._SamplePool.prepare", side_effect=AssertionError("native descendant must not eagerly prepare every image")):
                        report = self._run(root, dataset, root / f"descendant-{index}.pth", self._options(
                            limit_mode="steps", total_steps=1, validation_samples=1, checkpoint_schedule="end"))
                    self.assertTrue(checking.call_args.kwargs["metadata_only"])
                    self.assertEqual(report["total_steps"], 1)
                    self.assertIn("Consumed-array", report["integrity_policy"])
            from ipde.array_storage import read_array, write_array
            target_path = edited / json.loads((edited / "dataset.json").read_text())["samples"][0]["teacher"]["target"]["path"]
            changed = np.array(read_array(target_path), copy=True)
            changed[1, 1] += np.float32(.125)
            write_array(target_path, changed, compressed=True)
            with self.assertRaisesRegex(TrainingIntegrityError, "checksum mismatch"):
                self._run(root, edited, root / "changed-descendant.pth", self._options(epochs=1, checkpoint_schedule="end"))

    def test_imported_mixed_and_unknown_lineage_keep_full_preflight(self):
        from ipde.dataset import load_dataset
        from ipde.dataset_collection import CollectionOptions, compose_datasets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _dataset(root, count=6)
            path = source / "dataset.json"
            unknown = json.loads(path.read_text())
            self.assertFalse(_native_training_dataset(unknown))
            composed = root / "unknown-composed"
            with redirect_stderr(io.StringIO()):
                compose_datasets([source], composed, CollectionOptions(validation_fraction=.5), workers=1)
            self.assertFalse(_native_training_dataset(json.loads((composed / "dataset.json").read_text())))
            with patch("ipde.training.load_dataset", wraps=load_dataset) as checking:
                self._run(root, composed, root / "unknown.pth", self._options(limit_mode="steps", total_steps=1, checkpoint_schedule="end"))
            self.assertFalse(checking.call_args.kwargs["metadata_only"])
            _mark_native(unknown, source)
            self.assertTrue(_native_training_dataset(unknown))
            # A native base cannot conceal imported additions, including after
            # in-place edits preserve its old generation_output_dir.
            unknown["samples"][-1]["external_import"] = {"provenance": {"backend": "huggingface-datasets"}}
            self.assertFalse(_native_training_dataset(unknown))
            path.write_text(json.dumps(unknown))
            unused_raw = source / unknown["samples"][0]["raw_assets"][0]["storage"]["path"]
            unused_raw.write_bytes(b"corrupt unused auxiliary in a mixed/external dataset")
            with patch("ipde.training.load_dataset", wraps=load_dataset) as checking:
                with self.assertRaisesRegex(TrainingError, "Cannot read dataset array"):
                    self._run(root, source, root / "mixed.pth", self._options(limit_mode="steps", total_steps=1, checkpoint_schedule="end"))
            self.assertFalse(checking.call_args.kwargs["metadata_only"])

    def test_zero_high_and_nonfinite_loss_stop_with_finite_resumable_weights(self):
        import torch
        for loss_value, reason in ((0., "zero_loss"), (2000., "loss_limit_exceeded"), (float("nan"), "nonfinite_loss")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = _dataset(root)
                with patch("ipde.training.sequence_loss", return_value=torch.tensor(loss_value)):
                    report = self._run(root, dataset, root / "stopped.pth", self._options(epochs=1, checkpoint_schedule="end"))
                self.assertEqual((report["total_steps"], report["stop_reason"]), (0, reason))
                payload = torch.load(root / "stopped.pth", weights_only=True)
                self.assertEqual(payload["ipde_resume"]["cursor"], 0)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in payload["state_dict"].values()))

    def test_divergence_limit_uses_mean_error_independent_of_iteration_count(self):
        from ipde.training import sequence_loss
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            updates = []
            for iterations in (4, 16):
                events = []
                options = replace(self._options(epochs=1, checkpoint_schedule="end"), iterations=iterations)
                # Identical images, crop, weights and per-prediction error.
                # Only the number of predictions changes between runs.
                with patch("ipde.training.sequence_loss", side_effect=lambda *args: sequence_loss(*args) * 100):
                    report = self._run(root, dataset, root / f"iterations-{iterations}.pth", options, progress=events.append)
                update = next(event for event in events if event["stage"] == "epoch_step" and event["status"] == "finished")
                updates.append(update)
                self.assertLess(update["guard_error_pixels"], options.max_loss)
                self.assertEqual((report["total_steps"], report["stop_reason"]), (1, "completed"))
            self.assertLess(updates[0]["loss"], options.max_loss)
            self.assertGreater(updates[1]["loss"], options.max_loss)
            self.assertAlmostEqual(updates[0]["guard_error_pixels"], updates[1]["guard_error_pixels"], places=3)

    def test_runtime_failure_preserves_completed_updates_and_rewinds_unapplied_group(self):
        import torch
        from ipde.training import sequence_loss
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            calls = 0
            def failed_loss(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("synthetic accelerator out of memory")
                return sequence_loss(*args)
            with patch("ipde.training.sequence_loss", side_effect=failed_loss):
                report = self._run(root, dataset, root / "failed.pth", self._options(limit_mode="steps", total_steps=5,
                    checkpoint_schedule="end"))
            self.assertEqual((report["total_steps"], report["stop_reason"]), (1, "runtime_error"))
            payload = torch.load(root / "failed.pth", weights_only=True)
            self.assertEqual(payload["ipde_resume"]["cursor"], 1)
            self.assertIn("synthetic accelerator out of memory", report["warnings"])

    def test_resume_rejects_changed_dataset_or_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            options = self._options(epochs=1, checkpoint_schedule="end")
            self._run(root, dataset, root / "initial-run.pth", options)
            with self.assertRaisesRegex(TrainingError, "steps_per_update"):
                self._run(root, dataset, root / "different.pth", replace(options,
                    steps_per_update=2, resume_from=root / "initial-run.pth"))
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["edit_revision"] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TrainingError, "dataset has changed"):
                self._run(root, dataset, root / "changed.pth", replace(options, resume_from=root / "initial-run.pth"))

    def test_native_resume_preserves_order_when_lazy_geometry_excluded_a_target(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=8)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            _mark_native(manifest, dataset)
            # Seed0 order for four training entries starts at index2/sample4.
            manifest["samples"][4]["calibration"]["baseline_meters"] = -1.
            manifest_path.write_text(json.dumps(manifest))
            control = root / "control.json"
            options = self._options(limit_mode="steps", total_steps=3, control_file=control, checkpoint_schedule="end")
            def stop_after_update(event):
                if event["stage"] == "epoch_step" and event["status"] == "finished" and event["completed_steps"] == 1:
                    control.write_text(json.dumps({"command": "stop", "request_id": "after-skip"}))
            stopped = self._run(root, dataset, root / "skipped.pth", options, progress=stop_after_update)
            self.assertEqual([record["sample_id"] for record in stopped["excluded_samples"]], ["sample-4"])
            payload = torch.load(root / "skipped.pth", weights_only=True)
            self.assertIn("sample-4", payload["ipde_resume"]["epoch_order_ids"])
            self.assertNotIn("sample-4", payload["ipde_resume"]["active_train_ids"])
            control.unlink()
            events = []
            resumed = self._run(root, dataset, root / "skipped-resumed.pth", replace(options, resume_from=root / "skipped.pth"),
                progress=events.append)
            self.assertEqual(resumed["total_steps"], 3)
            self.assertNotIn("sample-4", resumed["train_sample_ids"])
            updates = [event for event in events if event["stage"] == "epoch_step" and event["status"] == "finished"]
            self.assertEqual([event["step"] for event in updates], [2, 3])
            self.assertTrue(all(event["steps_per_epoch"] == 3 for event in updates))

    def test_stop_during_baseline_validation_saves_without_finishing_full_dataset(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            control = root / "control.json"
            events = []
            def stop_during_validation(event):
                events.append(event)
                if event["stage"] == "baseline_validation" and event.get("sample_patch") == 1:
                    control.write_text(json.dumps({"command": "stop", "request_id": "baseline-stop"}))
            report = self._run(root, dataset, root / "cancelled.pth", self._options(control_file=control), progress=stop_during_validation)
            self.assertEqual((report["total_steps"], report["stop_reason"]), (0, "user_stopped"))
            self.assertFalse(any(event["stage"] == "epoch_step" for event in events))
            self.assertIn("ipde_resume", torch.load(root / "cancelled.pth", weights_only=True))


if __name__ == "__main__":
    unittest.main()
