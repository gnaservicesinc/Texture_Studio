"""Depth-edge supervision is independent of tiles, masks and display RGB."""
from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ipde import display_student as student
from ipde.display_training import (GRADIENT_SCALES, GRADIENT_WEIGHT, _depth_probe_statistics,
                                  LOG_ORDINAL_OBJECTIVE, ORDINAL_MAX_LOG_MARGIN, ORDINAL_WEIGHT,
                                  PREVIOUS_DETAIL_OBJECTIVE, _diagnostic_windows, _gradient_error_sum,
                                  _gradient_pair_count, _ordinal_separation_loss)
from verification import test_display_student as student_fixtures
from verification import test_display_training as training_fixtures


class DepthGradientTests(unittest.TestCase):
    def test_halo_tiles_match_full_loss_and_gradients_with_masked_nan_targets(self):
        generator = np.random.default_rng(827)
        target = generator.uniform(.5, 5, (49, 67)).astype(np.float32)
        target.view(np.uint32)[7, 8] = 0x7FC12345
        target[10:17, 11:19] = 0
        original = target.tobytes()
        mask = np.isfinite(target) & (target > 0)
        valid = torch.from_numpy(mask)
        teacher = torch.from_numpy(target)
        values = torch.from_numpy(generator.uniform(.5, 4, target.shape).astype(np.float32))
        whole, tiled = values.clone().requires_grad_(), values.clone().requires_grad_()
        complete = _gradient_error_sum(whole, teacher, valid, target.shape)
        sum_tiles = tiled.new_zeros(())
        for y0, y1, x0, x1 in student.tile_bounds(target.shape, 32):
            query_y1, query_x1 = min(target.shape[0], y1 + max(GRADIENT_SCALES)), min(target.shape[1], x1 + max(GRADIENT_SCALES))
            sum_tiles = sum_tiles + _gradient_error_sum(tiled[y0:query_y1, x0:query_x1],
                teacher[y0:query_y1, x0:query_x1], valid[y0:query_y1, x0:query_x1], (y1-y0, x1-x0))
        count = sum(int((valid[:, :-step] & valid[:, step:]).sum())
                    + int((valid[:-step] & valid[step:]).sum()) for step in GRADIENT_SCALES)
        self.assertEqual(_gradient_pair_count(mask), count)
        torch.testing.assert_close(sum_tiles, complete, rtol=1e-6, atol=1e-3)
        (complete/count).backward(); (sum_tiles/count).backward()
        torch.testing.assert_close(tiled.grad, whole.grad, rtol=1e-6, atol=1e-7)
        self.assertTrue(bool(torch.isfinite(tiled.grad).all()))
        self.assertEqual(target.tobytes(), original)

    def test_depth_boundaries_are_supervised_and_extra_texture_is_penalized(self):
        target = torch.ones(40, 64)
        target[:, 30:] = 4
        valid = torch.ones_like(target, dtype=torch.bool)
        smooth = torch.linspace(1., 4., 64)[None].expand_as(target)
        textured = target + .2 * (torch.arange(64) % 2)[None]
        exact = _gradient_error_sum(target, target, valid, target.shape)
        self.assertEqual(float(exact), 0)
        self.assertGreater(float(_gradient_error_sum(smooth, target, valid, target.shape)), 0)
        self.assertGreater(float(_gradient_error_sum(textured, target, valid, target.shape)), 0)
        # Log-gradient agreement has the same geometry at another label scale.
        torch.testing.assert_close(_gradient_error_sum(smooth * 7, target * 7, valid, target.shape),
                                   _gradient_error_sum(smooth, target, valid, target.shape), rtol=1e-6, atol=1e-3)


class OrdinalDepthTrainingTests(unittest.TestCase):
    setUpClass = classmethod(student_fixtures.DisplayStudentTests.setUpClass.__func__)
    tearDownClass = classmethod(student_fixtures.DisplayStudentTests.tearDownClass.__func__)

    def test_flat_and_reversed_depths_receive_corrective_ordering_gradients(self):
        reference = torch.tensor([1., 8.])
        flat = torch.ones(2, requires_grad=True)
        loss = _ordinal_separation_loss(flat, reference)
        loss.backward()
        self.assertGreater(float(loss.detach()), .6)
        self.assertGreater(float(flat.grad[0]), 0)
        self.assertLess(float(flat.grad[1]), 0)
        reversed_loss = _ordinal_separation_loss(torch.tensor([1., .5]), reference)
        self.assertGreater(float(reversed_loss), float(loss.detach()))
        self.assertLess(float(_ordinal_separation_loss(torch.tensor([1., 2.]), reference)), 1e-6)

    def test_separation_is_bounded_and_scale_invariant_without_changing_labels(self):
        reference = torch.tensor([1., 1e6])
        original = reference.clone()
        flat = torch.ones(2)
        self.assertAlmostEqual(float(_ordinal_separation_loss(flat, reference)), ORDINAL_MAX_LOG_MARGIN, places=6)
        torch.testing.assert_close(_ordinal_separation_loss(flat, reference), _ordinal_separation_loss(flat*7, reference*7))
        torch.testing.assert_close(reference, original, rtol=0, atol=0)
        with self.assertRaisesRegex(Exception, "at most 256"):
            _ordinal_separation_loss(torch.ones(257), torch.ones(257))

    def test_sparse_far_geometry_recovers_correct_depths_from_a_flat_initial_model(self):
        # Two representable synthetic geometry categories, deliberately imbalanced
        # 95:5. This isolates objective behavior; it is not real-photo accuracy.
        category = torch.cat((torch.zeros(95, dtype=torch.long), torch.ones(5, dtype=torch.long)))
        reference = torch.cat((torch.ones(95), torch.full((5,), 8.)))
        parameters = torch.nn.Parameter(torch.full((2,), .54))
        optimizer = torch.optim.Adam([parameters], lr=.1)
        for _ in range(350):
            optimizer.zero_grad()
            prediction = torch.nn.functional.softplus(parameters[category]) + 1e-6
            loss = (prediction.log() - reference.log()).abs().mean()
            loss = loss + ORDINAL_WEIGHT * _ordinal_separation_loss(prediction, reference)
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.softplus(parameters[category]).detach() + 1e-6
        self.assertLess(float((final[:95] - 1.).abs().mean()), .05)
        self.assertLess(float((final[95:] - 8.).abs().mean()), .3)
        diagnostics = _depth_probe_statistics(reference.numpy(), final.numpy(), final.numpy())
        self.assertEqual(diagnostics["reference_ordering_agreement"], 1.)
        self.assertFalse(diagnostics["collapsed_against_reference"])


class NativeDetailTests(unittest.TestCase):
    setUpClass = classmethod(student_fixtures.DisplayStudentTests.setUpClass.__func__)
    tearDownClass = classmethod(student_fixtures.DisplayStudentTests.tearDownClass.__func__)
    setUp = student_fixtures.DisplayStudentTests.setUp
    model = student_fixtures.DisplayStudentTests.model
    payload = student_fixtures.DisplayStudentTests.payload
    context = student_fixtures.DisplayStudentTests.context

    def test_v2_weights_keep_exact_predictions_and_their_resume_architecture(self):
        _, configuration, _ = self.model()
        configuration["architecture"] = student.ALIGNED_ARCHITECTURE
        for key in ("detail_channels", "detail_stride", "detail_reference"):
            configuration.pop(key)
        previous = student._model(configuration, self.root)
        checkpoint = self.root / "v2.pth"
        torch.save(self.payload(previous, configuration), checkpoint)
        restored, saved, _ = student.load_student_checkpoint(checkpoint, raft_root=self.root, device="cpu", allow_legacy=False)
        self.assertEqual(saved["architecture"], student.ALIGNED_ARCHITECTURE)
        self.assertFalse(hasattr(restored, "detail_encoder"))
        with torch.no_grad():
            bounds = (0, 39, 0, 51)
            torch.testing.assert_close(restored.render(self.context(restored), (39, 51), bounds),
                                       previous.render(self.context(previous), (39, 51), bounds), rtol=0, atol=0)

    def test_fine_learned_field_can_predict_structure_absent_from_coarse_field(self):
        model, configuration, _ = self.model()
        channels = configuration["channels"]
        model.eval()
        # Isolate the learned fine descriptor: no RGB and no coarse structure.
        context = {"fused": torch.zeros(1, channels, 2, 8), "detail": torch.zeros(1, 8, 8, 32),
                   "global": torch.zeros(1, channels), "input_shape": (16, 64)}
        context["detail"][:, 0, :, 15] = 1
        with torch.no_grad():
            for layer in (model.depth[0], model.depth[2], model.depth[4]):
                layer.weight.zero_(); layer.bias.zero_()
            model.depth[0].weight[0, channels, 0, 0] = 1
            model.depth[2].weight[0, 0, 0, 0] = 1
            model.depth[4].weight[0, 0, 0, 0] = 1
            result = model.render(context, (16, 64), (0, 16, 0, 64))[0, 0]
        self.assertGreater(float(result[:, 30:32].mean()), float(result[:, :20].mean()) + .05)
        # A one-feature-wide native detail survives despite a constant scene field.
        self.assertLessEqual(int((result[8] > result[8, 0] + .01).sum()), 4)

    def test_real_v3_learns_a_thin_depth_object_without_copying_image_texture(self):
        model, _, _ = student.create_student(raft_root=self.root, raft_model=self.original,
            device="cpu", quality=0, iterations=1, seed=124)
        rgb = np.full((32, 64, 3), 80, np.uint8)
        rgb[:, 29:35] = (220, 70, 150)
        rgb[:, ::2] = np.minimum(rgb[:, ::2].astype(np.int16) + 20, 255).astype(np.uint8)
        original = rgb.tobytes()
        left = student.rgb_tensor(rgb, {}, torch, "cpu")
        target = torch.full((64, 128), 4.)
        target[:, 58:70] = 1.
        valid = torch.ones_like(target, dtype=torch.bool)
        pairs = _gradient_pair_count(valid.numpy())
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.003)
        def predict():
            return model.render(model.encode(left, left), target.shape, (0, 64, 0, 128))[0, 0]
        with torch.no_grad():
            initial = predict()
            initial_depth_error = float(((initial - target).abs() / target).mean())
            initial_gradient_error = float(_gradient_error_sum(initial, target, valid, target.shape) / pairs)
        for _ in range(60):
            optimizer.zero_grad()
            prediction = predict()
            loss = ((prediction - target).abs() / target).mean()
            loss = loss + GRADIENT_WEIGHT * _gradient_error_sum(prediction, target, valid, target.shape) / pairs
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            result = predict()
            depth_error = float(((result - target).abs() / target).mean())
            gradient_error = float(_gradient_error_sum(result, target, valid, target.shape) / pairs)
        self.assertLess(depth_error, initial_depth_error * .15)
        self.assertLess(gradient_error, initial_gradient_error * .5)
        self.assertLess(float(result[:, 58:70].mean()), 1.3)
        self.assertGreater(float(result[:, :45].mean()), 3.5)
        # The sharp learned boundary and flat background are supervised depth,
        # despite bright alternating RGB texture in the background input.
        self.assertLessEqual(int(((result[32] > 1.5) & (result[32] < 3.5)).sum()), 6)
        self.assertEqual(rgb.tobytes(), original)


class FullValidationTests(unittest.TestCase):
    _run = training_fixtures.DisplayTrainingTests._run
    _options = training_fixtures.DisplayTrainingTests._options

    def test_training_gradient_objective_is_independent_of_decoder_tile_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = training_fixtures._display_dataset(root)
            original = {path.name: path.read_bytes() for path in dataset.iterdir()}
            options = self._options(limit_mode="steps", total_steps=1)
            # Keep the real training/halo/backward path, with the inexpensive
            # differentiable fixture decoder used by the lifecycle suite.
            with patch.object(student, "ARCHITECTURE", "fixture-display"):
                tiled, _ = self._run(dataset, root / "tiled.pth", options)
                whole, _ = self._run(dataset, root / "whole.pth", replace(options, patch_size=128))
            self.assertEqual(tiled["optimization_objective"]["gradient_scales_display_pixels"], [1, 4, 16])
            self.assertEqual(tiled["optimization_objective"]["name"], LOG_ORDINAL_OBJECTIVE)
            self.assertTrue(tiled["optimization_objective"]["ordinal_term"]["enabled"])
            a, b = (torch.load(root / name, weights_only=True) for name in ("tiled.pth", "whole.pth"))
            for key in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], atol=1e-6, rtol=1e-6)
            self.assertEqual(original, {path.name: path.read_bytes() for path in dataset.iterdir()})

    def test_earlier_v3_resume_preserves_recorded_fractional_gradient_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = training_fixtures._display_dataset(root)
            options = self._options(limit_mode="steps", total_steps=1)
            with patch.object(student, "ARCHITECTURE", "fixture-display"):
                self._run(dataset, root / "new.pth", options)
                payload = torch.load(root / "new.pth", weights_only=True)
                payload["ipde_training"]["optimization_objective"]["name"] = PREVIOUS_DETAIL_OBJECTIVE
                torch.save(payload, root / "older-v3.pth")
                result, _ = self._run(dataset, root / "resumed.pth", replace(options, total_steps=2, resume_from=root / "older-v3.pth"))
            objective = result["optimization_objective"]
            self.assertEqual(objective["name"], PREVIOUS_DETAIL_OBJECTIVE)
            self.assertEqual(objective["depth_term"], "mean(abs(prediction-target)/target)")
            self.assertFalse(objective["ordinal_term"]["enabled"])

    def test_baseline_final_and_best_use_identical_full_held_out_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = training_fixtures._display_dataset(root)
            result, _ = self._run(dataset, root / "model.pth", self._options(epochs=1, validation_samples=0))
            baseline, final, best = (result[key] for key in ("baseline_validation", "validation", "best_validation"))
            self.assertEqual(baseline["sample_indices"], final["sample_indices"])
            self.assertEqual(best["sample_indices"], final["sample_indices"])
            self.assertTrue(all(item["full_validation"] and item["sample_count"] == 3 for item in (baseline, final, best)))
            best_path = Path(result["best_checkpoint_path"])
            self.assertTrue(best_path.is_file())
            saved = torch.load(best_path, weights_only=True)
            self.assertEqual(saved["ipde_training"]["validation"], best)
            self.assertEqual(saved["ipde_training"]["best_error"], result["best_error"])
            self.assertEqual(saved["ipde_training"]["validation"]["mean_absolute_fractional_depth_error"], result["best_error"])

    def test_resume_without_old_best_preserves_history_and_saves_recoverable_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = training_fixtures._display_dataset(root)
            options = self._options(limit_mode="steps", total_steps=1)
            initial, _ = self._run(dataset, root / "first.pth", options)
            Path(initial["best_checkpoint_path"]).unlink()
            resumed, _ = self._run(dataset, root / "resumed.pth", replace(options, total_steps=2, resume_from=root / "first.pth"))
            self.assertTrue(Path(resumed["best_checkpoint_path"]).is_file())
            self.assertEqual(resumed["historical_best_validation"]["error"], initial["best_error"])
            self.assertFalse(resumed["historical_best_validation"]["checkpoint_available"])

    def test_resume_keeps_a_legitimate_zero_best_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = training_fixtures._display_dataset(root)
            options = self._options(limit_mode="steps", total_steps=1)
            initial, _ = self._run(dataset, root / "first.pth", options)
            payload = torch.load(root / "first.pth", weights_only=True)
            payload["ipde_training"]["best_error"] = 0.
            torch.save(payload, root / "zero.pth")
            result, _ = self._run(dataset, root / "resumed.pth", replace(options, resume_from=root / "zero.pth"))
            self.assertEqual(result["best_error"], 0.)
            self.assertEqual(result["best_checkpoint_path"], initial["best_checkpoint_path"])


class CheckpointQualityTests(unittest.TestCase):
    def payload(self, full=True, final=.7, initial=.3):
        return {"architecture": {"architecture": student.ARCHITECTURE}, "ipde_training": {
            "total_steps": 12, "validation_completed_steps": 12,
            "baseline_validation": {"full_validation": full, "sample_indices": [0, 1], "sample_count": 2, "units": "meters", "mean_absolute_fractional_depth_error": initial},
            "validation": {"full_validation": full, "sample_indices": [0, 1], "sample_count": 2, "units": "meters", "mean_absolute_fractional_depth_error": final}}}

    def test_worse_checkpoint_and_unrelated_subsets_cannot_claim_improvement(self):
        payload = self.payload()
        self.assertEqual(student.checkpoint_quality(payload)["status"], "not_improved_full_validation")
        payload["ipde_training"]["validation"]["sample_indices"] = [2, 3]
        self.assertEqual(student.checkpoint_quality(payload)["status"], "validation_incomplete")
        self.assertEqual(student.checkpoint_quality(self.payload(full=False))["status"], "validation_incomplete")
        self.assertEqual(student.checkpoint_quality(self.payload(final=.1))["status"], "improved_full_validation")

    def test_stale_validation_and_empty_or_malformed_coverage_cannot_certify_weights(self):
        for modify in (
            lambda payload: payload["ipde_training"].update(validation_completed_steps=11),
            lambda payload: payload["ipde_training"].update(total_steps=True),
            lambda payload: payload["ipde_training"]["validation"].update(sample_indices=[], sample_count=0),
            lambda payload: payload["ipde_training"]["validation"].update(sample_count=99),
            lambda payload: payload["ipde_training"]["validation"].update(units="unknown"),
        ):
            payload = self.payload(final=.1)
            modify(payload)
            self.assertEqual(student.checkpoint_quality(payload)["status"], "validation_incomplete")

    def test_old_decoder_reports_new_training_required_without_altering_weights(self):
        payload = self.payload(final=.1)
        payload["architecture"]["architecture"] = student.ALIGNED_ARCHITECTURE
        original = deepcopy(payload)
        quality = student.checkpoint_quality(payload)
        self.assertTrue(any("new training run" in warning for warning in quality["warnings"]))
        self.assertEqual(payload, original)

    def test_geometry_diagnostics_flag_bad_depth_despite_improved_fractional_score(self):
        payload = self.payload(final=.1)
        payload["ipde_training"]["validation"]["depth_diagnostics"] = {
            "sample_count": 2, "ordering_pair_count": 20, "reference_ordering_agreement": .4,
            "supported_probe_pixels": 100, "mean_fractional_depth_change_under_photometric_probe": .2,
            "constant_depth_reference_fractional_error": .05,
            "collapsed_sample_count": 1, "appearance_sensitive_sample_count": 1}
        quality = student.checkpoint_quality(payload)
        self.assertEqual(quality["scientific_status"], "experimental_unconstrained_display_depth")
        self.assertEqual(quality["geometry_assessment"], "reference_diagnostic_flags")
        self.assertEqual(set(quality["diagnostic_flags"]), {"inferior_to_constant_depth_reference", "collapsed_depth_range",
                                                          "appearance_sensitive", "reference_ordering_errors"})
        payload["ipde_training"]["validation_completed_steps"] = 11
        self.assertEqual(student.checkpoint_quality(payload)["geometry_assessment"], "geometry_unvalidated")


class DepthGeometryDiagnosticTests(unittest.TestCase):
    def test_correct_depth_is_stable_under_brightness_changes_but_intensity_depth_fails(self):
        # Bright texture is on a far plane; dark texture is on a near plane.
        reference = np.r_[np.ones(100), np.full(100, 4.)].astype(np.float32)
        original = reference.tobytes()
        stable = _depth_probe_statistics(reference, reference, reference)
        self.assertEqual(stable["reference_ordering_agreement"], 1.)
        self.assertEqual(stable["mean_fractional_depth_change_under_photometric_probe"], 0.)
        self.assertFalse(stable["appearance_sensitive"])
        # Near-high grayscale intensity misread as camera depth reverses geometry.
        intensity_prediction = np.r_[np.full(100, 4.), np.ones(100)].astype(np.float32)
        appearance = _depth_probe_statistics(reference, intensity_prediction, intensity_prediction * .75 + .05)
        self.assertEqual(appearance["reference_ordering_agreement"], 0.)
        self.assertTrue(appearance["appearance_sensitive"])
        self.assertEqual(reference.tobytes(), original)

    def test_constant_depth_is_collapsed_for_a_variable_reference_and_flat_reference_is_valid(self):
        reference = np.linspace(1., 4., 200, dtype=np.float32)
        collapsed = _depth_probe_statistics(reference, np.ones(200), np.ones(200))
        self.assertTrue(collapsed["collapsed_against_reference"])
        self.assertEqual(collapsed["reference_ordering_agreement"], 0.)
        flat = _depth_probe_statistics(np.ones(200), np.ones(200), np.ones(200))
        self.assertFalse(flat["collapsed_against_reference"])
        self.assertEqual(flat["ordering_pair_count"], 0)

    def test_relative_inverse_depth_ordering_and_supported_masks_are_preserved(self):
        inverse = 1. / np.linspace(1., 4., 100, dtype=np.float32)
        inverse.view(np.uint32)[0] = 0x7FC12345
        original = inverse.tobytes()
        stats = _depth_probe_statistics(inverse, inverse, inverse)
        self.assertEqual(stats["reference_ordering_agreement"], 1.)
        self.assertEqual(stats["supported_probe_pixels"], 99)
        self.assertEqual(inverse.tobytes(), original)

    def test_query_windows_are_bounded_disjoint_and_span_original_display_extent(self):
        for shape in ((1, 1), (49, 67), (4284, 5712)):
            windows = _diagnostic_windows(shape)
            self.assertLessEqual(len(windows), 16)
            self.assertEqual(min(bounds[0] for bounds in windows), 0)
            self.assertEqual(max(bounds[1] for bounds in windows), shape[0])
            self.assertEqual(min(bounds[2] for bounds in windows), 0)
            self.assertEqual(max(bounds[3] for bounds in windows), shape[1])
            for i, (y0, y1, x0, x1) in enumerate(windows):
                self.assertLessEqual((y1-y0) * (x1-x0), 32 * 32)
                for other in windows[i+1:]:
                    self.assertFalse(max(y0, other[0]) < min(y1, other[1]) and max(x0, other[2]) < min(x1, other[3]))


if __name__ == "__main__":
    unittest.main()
