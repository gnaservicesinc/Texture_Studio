"""Real optimizer coverage for the unregistered full-display student lifecycle."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, write_array
from ipde.display_training import display_target_eligibility, export_display_checkpoint, train_display_dataset
from ipde.training import TrainingError, TrainingIntegrityError, TrainingOptions
from test_training_progress import _dataset
from verification.test_training_lifecycle import _mark_native


def _display_dataset(root, count=6, shape=(73, 107), *, native=True, units="meters"):
    dataset = _dataset(root, count=count)
    manifest = json.loads((dataset / "dataset.json").read_text())
    for index, sample in enumerate(manifest["samples"]):
        display = np.random.default_rng(index + 12).integers(0, 256, (*shape, 3), dtype=np.uint8)
        rgb = array_record(dataset, dataset / f"display-{index}.npy", display)
        target = np.linspace(1.5, 2.5, shape[0] * shape[1], dtype=np.float32).reshape(shape)
        target.view(np.uint32)[0, 0] = 0x7FC12345
        target[0, 1] = 0
        label = {"coordinate_reference": "display", "units": units,
            "target": array_record(dataset, dataset / f"display-target-{index}.npy", target),
            "native_target": array_record(dataset, dataset / f"native-teacher-{index}.npy", target[::8, ::8].copy()),
            "valid_mask": array_record(dataset, dataset / f"display-mask-{index}.npy", np.isfinite(target) & (target > 0)),
            "metadata": {"reference_label": "display", "checkpoint_sha256": "a" * 64, "input_rgb_sha256": rgb["array_sha256"]}}
        sample.update(display_rgb=rgb, display_teacher=label, teacher=dict(label), teacher_view="display",
            training_target_choice="display_teacher")
    if native:
        _mark_native(manifest, dataset)
    (dataset / "dataset.json").write_text(json.dumps(manifest))
    return dataset


def _fixture_model(torch):
    import torch._dynamo
    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(6, 2, 1)
            self.decoder = torch.nn.Conv2d(2, 1, 1)
            self.native_shapes, self.bounds, self.encoder_backward_count = [], [], 0
            def backward_hook(_module, _inputs, _outputs):
                self.encoder_backward_count += 1
            self.encoder.register_full_backward_hook(backward_hook)

        def encode(self, left, right):
            self.native_shapes.append(tuple(left.shape[-2:]))
            features = self.encoder(torch.cat((left, right), dim=1) / 255.)
            return {"left": features, "global": features.mean(dim=(2, 3)), "input_shape": tuple(left.shape[-2:])}

        def render(self, context, output_shape, tile_bounds):
            self.bounds.append((output_shape, tile_bounds, self.training))
            h, w = output_shape
            y0, y1, x0, x1 = tile_bounds
            y = (torch.arange(y0, y1, dtype=torch.float32, device=context["left"].device) + .5) * (2. / h) - 1.
            x = (torch.arange(x0, x1, dtype=torch.float32, device=context["left"].device) + .5) * (2. / w) - 1.
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            sampled = torch.nn.functional.grid_sample(context["left"], torch.stack((xx, yy), -1)[None], align_corners=False)
            sampled = sampled + context["global"][:, :, None, None]
            return torch.nn.functional.softplus(self.decoder(sampled)) + 1e-6
    return TinyStudent


class DisplayTrainingTests(unittest.TestCase):
    def _options(self, **values):
        return TrainingOptions(patch_size=32, iterations=1, device="cpu", learning_rate=1e-3,
            checkpoint_schedule="end", validation_schedule="checkpoint", **values)

    def _run(self, dataset, output, options, progress=None, *, modify_model=None):
        import torch
        cls = _fixture_model(torch)
        models = []
        def create(**values):
            model = cls().to(values["device"])
            if modify_model:
                modify_model(model)
            models.append(model)
            return model, {"architecture": "fixture-display", "iterations": values["iterations"], "units": values["units"]}, values["device"]
        def load(payload, **values):
            saved = torch.load(payload, weights_only=True, map_location="cpu") if not isinstance(payload, dict) else payload
            model = cls().to(values["device"])
            model.load_state_dict(saved["state_dict"], strict=True)
            models.append(model)
            return model, saved["architecture"], values["device"]
        with patch("ipde.display_student.create_student", side_effect=create), \
                patch("ipde.display_student.load_student_checkpoint", side_effect=load), redirect_stderr(io.StringIO()):
            result = train_display_dataset(dataset, output, options, progress_callback=progress)
        return result, models[-1]

    def test_full_native_inputs_full_display_pixels_and_one_encoder_backward_per_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            before = {path.name: path.read_bytes() for path in dataset.iterdir()}
            events = []
            result, model = self._run(dataset, root / "student.pth", self._options(epochs=1, steps_per_update=2), events.append)
            self.assertEqual((result["total_steps"], result["completed_images"], result["epochs_completed"]), (2, 3, 1))
            self.assertEqual(model.encoder_backward_count, 3)
            self.assertTrue(all(shape == (64, 96) for shape in model.native_shapes))
            for sample_id in result["train_sample_ids"]:
                tiles = [event for event in events if event["stage"] == "display_tile" and not event["validation"] and event["sample_id"] == sample_id]
                self.assertEqual(len(tiles), 12)
                self.assertTrue(all(event["output_shape"] == [73, 107] for event in tiles))
            self.assertEqual(result["validation"]["evaluated_pixels"], 3 * (73 * 107 - 2))
            self.assertEqual(before, {path.name: path.read_bytes() for path in dataset.iterdir()})

    def test_exact_step_limit_and_bit_exact_resume_after_epoch_boundary(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            control = root / "control.json"
            options = self._options(limit_mode="steps", total_steps=5, steps_per_update=2, control_file=control)
            def stop(event):
                if event["stage"] == "epoch_step" and event["status"] == "finished" and event["completed_steps"] == 2:
                    control.write_text(json.dumps({"command": "stop", "request_id": "stop-two"}))
            result, _ = self._run(dataset, root / "stopped.pth", options, stop)
            self.assertEqual((result["total_steps"], result["stop_reason"]), (2, "user_stopped"))
            control.unlink()
            resumed, _ = self._run(dataset, root / "resumed.pth", replace(options, resume_from=root / "stopped.pth"))
            reference, _ = self._run(dataset, root / "reference.pth", options)
            self.assertEqual((resumed["total_steps"], resumed["completed_images"]), (5, 8))
            self.assertEqual(reference["total_steps"], resumed["total_steps"])
            a, b = (torch.load(root / name, weights_only=True) for name in ("resumed.pth", "reference.pth"))
            for key in a["state_dict"]:
                self.assertTrue(torch.equal(a["state_dict"][key], b["state_dict"][key]), key)

    def test_tile_gradient_accumulation_matches_single_full_display_loss(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            options = self._options(limit_mode="steps", total_steps=1)
            a, _ = self._run(dataset, root / "tiles.pth", options)
            b, _ = self._run(dataset, root / "whole.pth", replace(options, patch_size=128))
            tiled, whole = (torch.load(root / name, weights_only=True) for name in ("tiles.pth", "whole.pth"))
            for key in tiled["state_dict"]:
                torch.testing.assert_close(tiled["state_dict"][key], whole["state_dict"][key], atol=1e-6, rtol=1e-6)
            self.assertAlmostEqual(a["validation"]["mean_absolute_fractional_depth_error"], b["validation"]["mean_absolute_fractional_depth_error"], places=6)

    def test_manual_checkpoint_and_every_n_steps_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            control = root / "control.json"
            options = replace(self._options(limit_mode="steps", total_steps=4, control_file=control), checkpoint_schedule="steps", checkpoint_every=3)
            def save(event):
                if event["stage"] == "epoch_step" and event["status"] == "finished" and event["completed_steps"] == 1:
                    control.write_text(json.dumps({"command": "save", "request_id": "save-one"}))
            result, _ = self._run(dataset, root / "final.pth", options, save)
            reports = [json.loads(Path(path + ".json").read_text()) for path in result["checkpoint_paths"]]
            self.assertEqual([(item["total_steps"], item["stop_reason"]) for item in reports], [(1, "manual"), (3, "scheduled"), (4, "completed")])
            self.assertTrue(all(item["resumable"] for item in reports))

    def test_early_stop_sample_is_confirmed_on_every_validation_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            events = []
            options = replace(self._options(epochs=4, validation_samples=1, early_stop_error=2.), checkpoint_schedule="epoch", validation_schedule="epoch")
            result, _ = self._run(dataset, root / "early.pth", options, events.append)
            self.assertEqual((result["epochs_completed"], result["stop_reason"]), (1, "early_stop_error_reached"))
            self.assertTrue(result["validation"]["full_validation"])
            self.assertEqual(result["validation"]["sample_count"], 3)
            self.assertTrue(any(event["stage"] == "early_stop_confirmation" for event in events))

    def test_requested_baseline_subset_starts_quickly_and_final_validation_stays_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            events = []
            options = self._options(limit_mode="steps", total_steps=1, validation_samples=1)
            result, _ = self._run(dataset, root / "subset.pth", options, events.append)
            baseline = [event for event in events if event["stage"] == "baseline_validation" and event["status"] == "finished"]
            final = [event for event in events if event["stage"] == "final_validation" and event["status"] == "finished"]
            self.assertEqual(baseline[0]["sample_count"], 1)
            self.assertFalse(baseline[0]["full_validation"])
            self.assertEqual(final[-1]["sample_count"], 3)
            self.assertTrue(final[-1]["full_validation"])
            self.assertTrue(result["validation"]["full_validation"])
            self.assertFalse(result["baseline_validation"]["full_validation"])

    def test_fractional_error_units_are_reported_and_teacher_scale_is_not_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root, units="relative_inverse_depth")
            result, _ = self._run(dataset, root / "relative.pth", self._options(epochs=1))
            self.assertEqual(result["units"], "relative_inverse_depth")
            self.assertEqual(result["validation"]["units"], "relative_inverse_depth")
            self.assertGreater(result["validation"]["mean_absolute_depth_error"], 0)
            self.assertGreater(result["validation"]["mean_absolute_fractional_depth_error"], 0)
            manifest = json.loads((dataset / "dataset.json").read_text())
            manifest["samples"][0]["display_teacher"]["units"] = "meters"
            self.assertFalse(display_target_eligibility(manifest)["trainable"])
            self.assertIn("one target unit", display_target_eligibility(manifest)["reason"])

    def test_consumed_hash_failure_is_fatal_and_never_skipped_or_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            manifest = json.loads((dataset / "dataset.json").read_text())
            path = dataset / manifest["samples"][1]["display_teacher"]["target"]["path"]
            write_array(path, np.full((73, 107), 6., np.float32), compressed=path.suffix == ".npz")
            with self.assertRaisesRegex(TrainingIntegrityError, "checksum"):
                self._run(dataset, root / "corrupt.pth", self._options(epochs=1))
            self.assertFalse((root / "corrupt.pth").exists())

    def test_stop_during_partial_image_rolls_back_to_resumable_completed_update(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            control = root / "control.json"
            options = self._options(limit_mode="steps", total_steps=3, control_file=control)
            def stop(event):
                if event["stage"] == "display_tile" and not event["validation"] and event["completed_steps"] == 1 and event["tile"] == 2:
                    control.write_text(json.dumps({"command": "stop", "request_id": "partial"}))
            result, _ = self._run(dataset, root / "partial.pth", options, stop)
            self.assertEqual((result["total_steps"], result["completed_images"], result["stop_reason"]), (1, 1, "user_stopped"))
            control.unlink()
            self._run(dataset, root / "resumed.pth", replace(options, resume_from=root / "partial.pth"))
            self._run(dataset, root / "reference.pth", options)
            a, b = (torch.load(root / name, weights_only=True) for name in ("resumed.pth", "reference.pth"))
            self.assertTrue(all(torch.equal(a["state_dict"][key], b["state_dict"][key]) for key in a["state_dict"]))

    def test_nonfinite_and_large_loss_save_last_finite_state_with_reason(self):
        import torch
        for kind in ("nonfinite", "large", "runtime"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = _display_dataset(root)
                def modify(model):
                    original = model.render
                    def render(*args):
                        if model.training:
                            if kind == "runtime":
                                raise RuntimeError("synthetic resource exhaustion")
                            return original(*args) * (float("nan") if kind == "nonfinite" else 1e6)
                        return original(*args)
                    model.render = render
                result, _ = self._run(dataset, root / "safe.pth", self._options(epochs=1, max_loss=1000.), modify_model=modify)
                expected = {"nonfinite": "nonfinite_prediction", "large": "loss_limit_exceeded", "runtime": "runtime_error"}[kind]
                self.assertEqual(result["stop_reason"], expected)
                self.assertEqual(result["total_steps"], 0)
                payload = torch.load(root / "safe.pth", weights_only=True)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in payload["state_dict"].values()))
                self.assertEqual(payload["ipde_resume"]["completed_steps"], 0)

    def test_resume_requires_identical_manifest_and_math_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            options = self._options(limit_mode="steps", total_steps=1)
            self._run(dataset, root / "first.pth", options)
            with self.assertRaisesRegex(TrainingError, "preserve steps_per_update"):
                self._run(dataset, root / "changed.pth", replace(options, total_steps=2, steps_per_update=2, resume_from=root / "first.pth"))
            path = dataset / "dataset.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(TrainingError, "dataset has changed"):
                self._run(dataset, root / "changed-dataset.pth", replace(options, total_steps=2, resume_from=root / "first.pth"))

    def test_runtime_error_after_completed_update_preserves_optimizer_and_cursor(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            def modify(model):
                original = model.render
                images = 0
                def render(context, shape, bounds):
                    nonlocal images
                    if model.training and bounds[0] == bounds[2] == 0:
                        images += 1
                        if images == 2:
                            raise RuntimeError("resource failure after useful work")
                    return original(context, shape, bounds)
                model.render = render
            result, _ = self._run(dataset, root / "recovered.pth", self._options(limit_mode="steps", total_steps=3), modify_model=modify)
            self.assertEqual((result["total_steps"], result["completed_images"], result["stop_reason"]), (1, 1, "runtime_error"))
            payload = torch.load(root / "recovered.pth", weights_only=True)
            self.assertEqual(payload["ipde_resume"]["cursor"], 1)
            self.assertTrue(all(int(item["step"]) == 1 for item in payload["ipde_resume"]["optimizer_state"]["state"].values()))

    def test_export_preserves_weights_and_removes_resume_state_and_source_paths(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _display_dataset(root)
            self._run(dataset, root / "trained.pth", self._options(limit_mode="steps", total_steps=1))
            cls = _fixture_model(torch)
            def load(path, **values):
                payload = torch.load(path, weights_only=True)
                model = cls()
                model.load_state_dict(payload["state_dict"], strict=True)
                return model, payload["architecture"], "cpu"
            with patch("ipde.display_student.load_student_checkpoint", side_effect=load):
                manifest = export_display_checkpoint(root / "trained.pth", root / "export", raft_root=root)
                with self.assertRaisesRegex(TrainingError, "already exists"):
                    export_display_checkpoint(root / "trained.pth", root / "export", raft_root=root)
            portable = torch.load(root / "export/display-model.pth", weights_only=True)
            self.assertNotIn("ipde_resume", portable)
            self.assertNotIn("options", portable["ipde_training"])
            self.assertFalse(manifest["source_photos_included"])
            self.assertEqual(portable["schema"], "ipde-display-depth-v1")
            self.assertEqual(portable["architecture"], portable["ipde_configuration"])

    def test_stereo_target_and_measured_left_reference_are_ineligible(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            manifest = json.loads((dataset / "dataset.json").read_text())
            self.assertFalse(display_target_eligibility(manifest)["trainable"])
            sample = manifest["samples"][0]
            sample["display_rgb"] = dict(sample["rgb"])
            sample["reference"] = dict(sample["teacher"])
            self.assertFalse(display_target_eligibility(manifest, "supervised")["trainable"])

    def test_distillation_does_not_silently_use_measured_display_references(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _display_dataset(Path(directory))
            manifest = json.loads((dataset / "dataset.json").read_text())
            for sample in manifest["samples"]:
                sample["reference"] = dict(sample["display_teacher"])
                sample["reference"]["label_kind"] = "user_supplied_measured_reference"
                sample["training_target_choice"] = "reference"
            self.assertTrue(display_target_eligibility(manifest, "supervised")["trainable"])
            self.assertTrue(display_target_eligibility(manifest, "mixed")["trainable"])
            self.assertTrue(display_target_eligibility(manifest, "auto")["trainable"])
            rejected = display_target_eligibility(manifest, "distillation")
            self.assertFalse(rejected["trainable"])
            self.assertEqual(rejected["excluded_count"], len(manifest["samples"]))

    def test_cli_defaults_to_display_student_with_full_grid_controls(self):
        from ipde import trainer_cli
        captured = {}
        def train(dataset, output, options, **kwargs):
            captured.update(dataset=dataset, output=output, options=options)
            return {"schema": "ipde-display-training-report-v1", "total_steps": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("ipde.display_training.train_display_dataset", side_effect=train), \
                    patch("ipde.training.train_dataset") as legacy, redirect_stdout(stdout), redirect_stderr(stderr):
                result = trainer_cli.main(["--json", "train", str(root / "dataset"), "--checkpoint", str(root / "student.pth"),
                    "--limit-mode", "steps", "--total-steps", "2", "--steps-per-update", "3", "--patch-size", "512"])
            self.assertEqual(result, 0, stderr.getvalue())
            legacy.assert_not_called()
            self.assertEqual((captured["options"].total_steps, captured["options"].steps_per_update, captured["options"].patch_size), (2, 3, 512))
            self.assertEqual(json.loads(stdout.getvalue())["schema"], "ipde-display-training-report-v1")


if __name__ == "__main__":
    unittest.main()
