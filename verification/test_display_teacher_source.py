"""Full display pseudo-labels never become native stereo labels implicitly."""
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import read_array
from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.dataset_review import preview_sample, review_dataset
from ipde.extractor import Asset
from ipde.formats import sha256_array
from ipde.learned_depth import LearnedDepthConfig, LearnedDepthResult
from ipde.training import _target_exclusion, training_target_eligibility
from verification.test_spatial_scan import calibrated_capture


def _capture(source: Path, *, display_shape=(9, 12)):
    capture = calibrated_capture(source)
    capture.source_sha256 = hashlib.sha256(source.name.encode()).hexdigest()
    offset = int(source.stem.rsplit("-", 1)[-1]) if source.stem.rsplit("-", 1)[-1].isdigit() else 0
    capture.assets[0].array = capture.assets[0].array + offset * 3
    capture.assets[1].array = capture.assets[1].array + offset * 3
    capture.spatial_photo.update(focal_length_pixels_for_depth=20., is_spatial_photo=True,
                                metadata_source="macOS ImageIO container properties")
    display = (np.arange(np.prod((*display_shape, 3)), dtype=np.uint32) % 253).astype(np.uint8).reshape(*display_shape, 3)
    capture.assets.append(Asset("display_view", 2, 0, display, "RGB", 8, "display"))
    return capture


def _prediction(rgb, *, units="meters", depth=None):
    depth = np.full(rgb.shape[:2], 2., np.float32) if depth is None else depth
    native = np.array([[.25, .5, 1.], [2., 4., np.nan]], np.float32)
    return LearnedDepthResult(native, depth, {
        "units": units, "checkpoint_sha256": "a" * 64, "input_rgb_sha256": sha256_array(rgb),
        "model_input_shape": [16, 24], "requested_input_size": 0,
    }, confidence=np.full(native.shape, .75, np.float32))


class DisplayTeacherSourceTests(unittest.TestCase):
    def _build(self, root, *, options=None, capture=None, factory=None, **kwargs):
        sources = [root / f"capture-{index}.heic" for index in range(3)]
        for source in sources:
            source.write_bytes(b"capture")
        destination = root / "dataset"
        options = options or DatasetOptions(teacher=LearnedDepthConfig(), workers=1, skip_bad_photos=False)
        with patch("ipde.dataset.discover_file", side_effect=capture or _capture), \
                patch("ipde.learned_depth.LearnedDepthPredictor", side_effect=factory or (lambda config: lambda rgb, **values: _prediction(rgb))), \
                redirect_stderr(io.StringIO()):
            manifest = build_dataset(sources, destination, options, **kwargs)
        return destination, manifest

    def test_default_calls_teacher_only_on_display_and_preserves_all_prediction_grids(self):
        calls = []
        expected = {}
        def factory(config):
            def predict(rgb, **values):
                calls.append((sha256_array(rgb), rgb.shape, values))
                depth = np.linspace(.5, 4., rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
                depth.view(np.uint32)[0, 0] = 0x7FC12345
                result = _prediction(rgb, depth=depth)
                expected[sha256_array(rgb)] = result
                return result
            return predict
        with tempfile.TemporaryDirectory() as directory, \
                patch("ipde.registration.estimate_display_registration") as estimate, \
                patch("ipde.registration.register_display_depth") as transport:
            root = Path(directory)
            dataset, manifest = self._build(root, factory=factory, options=DatasetOptions(
                teacher=LearnedDepthConfig(), retain_intermediates=True, workers=1, skip_bad_photos=False))
            self.assertEqual(len(calls), 3)
            estimate.assert_not_called()
            transport.assert_not_called()
            self.assertEqual(manifest["teacher_view"], "display")
            self.assertEqual(load_dataset(dataset), manifest)
            for sample, call in zip(manifest["samples"], calls):
                rgb = sample["display_rgb"]
                self.assertEqual((call[0], call[1]), (rgb["array_sha256"], (9, 12, 3)))
                self.assertEqual(call[2]["reference_label"], "display")
                self.assertIsNone(call[2]["focal_pixels"])
                self.assertNotEqual(rgb["array_sha256"], sample["rgb"]["array_sha256"])
                prediction = expected[rgb["array_sha256"]]
                label = sample["display_teacher"]
                self.assertEqual(read_array(dataset / label["target"]["path"]).tobytes(), prediction.source_depth.tobytes())
                self.assertEqual(read_array(dataset / label["native_target"]["path"]).tobytes(), prediction.native_depth.tobytes())
                self.assertEqual(label["metadata"]["input_rgb_shape"], [9, 12, 3])
                self.assertEqual(label["metadata"]["model_input_shape"], [16, 24])
                self.assertEqual(label["metadata"]["native_prediction_shape"], [2, 3])
                self.assertEqual(label["metadata"]["stored_target_shape"], [9, 12])
                self.assertEqual(sample["teacher"]["alias_of"], "display_teacher")
                self.assertEqual(sample["training_target_choice"], "display_teacher")
                self.assertNotIn("registered_display_teacher", sample)
                self.assertNotIn("raft_target", sample)
                self.assertNotIn("display_registration", sample)
                self.assertEqual(read_array(dataset / sample["teacher"]["confidence"]["path"]).tobytes(), prediction.confidence.tobytes())
                self.assertIn("display-grid student", _target_exclusion(sample, "distillation")["reason"])
            report = review_dataset(dataset)
            self.assertTrue(all(sample["training_ready"] for sample in report["samples"]))
            self.assertTrue(all(sample["rgb_reference"] == "display" for sample in report["samples"]))
            self.assertTrue(report["training_eligibility"]["trainable"])
            self.assertFalse(report["raft_training_eligibility"]["trainable"])
            preview = preview_sample(dataset, manifest["samples"][0]["id"], "training", root / "preview")
            self.assertEqual(preview["rgb_reference"], "display")
            alias_preview = preview_sample(dataset, manifest["samples"][0]["id"], "teacher", root / "alias-preview")
            self.assertEqual(alias_preview["rgb_reference"], "display")
            self.assertEqual(alias_preview["label_title"], "Teacher on full display image")

    def test_display_mode_ignores_legacy_display_diagnostic_flags_and_registration_results(self):
        for registration in ({"accepted": True, "reference_role": "left"},
                             {"accepted": True, "reference_role": "right"}, {"accepted": False}):
            with self.subTest(registration=registration), tempfile.TemporaryDirectory() as directory, \
                    patch("ipde.registration.estimate_display_registration", return_value=registration) as estimate:
                options = DatasetOptions(teacher=LearnedDepthConfig(), include_display_teacher=False,
                    prefer_registered_display_teacher=True, workers=1)
                _, manifest = self._build(Path(directory), options=options)
                estimate.assert_not_called()
                self.assertTrue(all(sample["training_target_choice"] == "display_teacher" for sample in manifest["samples"]))

    def test_display_anchor_uses_display_rgb_and_relative_source_is_preserved(self):
        calls = []
        def factory(config):
            def predict(rgb, **values):
                calls.append((config.model, sha256_array(rgb), rgb.shape, values))
                relative = np.linspace(.2, 1.8, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
                return _prediction(rgb, units="meters" if config.model == "depthpro" else "relative_inverse_depth",
                    depth=1. / (relative * .25 + .1) if config.model == "depthpro" else relative)
            return predict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest = self._build(root,
                options=DatasetOptions(teacher=LearnedDepthConfig(model="depth-anything-v2"),
                    metric_anchor=LearnedDepthConfig(model="depthpro"), workers=1),
                capture=lambda source: _capture(source, display_shape=(96, 128)), factory=factory)
            self.assertEqual(len(calls), 6)
            for sample in manifest["samples"]:
                self.assertEqual(sample["training_target_choice"], "anchored_display_teacher")
                self.assertNotIn("metric_anchor", sample)
                self.assertNotIn("anchored_teacher", sample)
                self.assertTrue(sample["display_pseudo_calibration"]["accepted"])
                self.assertEqual(sample["display_teacher"]["units"], "relative_inverse_depth")
                self.assertEqual(sample["display_teacher"]["target"]["shape"], [96, 128])
                self.assertEqual(sample["display_metric_anchor"]["metadata"]["input_rgb_sha256"], sample["display_rgb"]["array_sha256"])
                self.assertIn("display-grid student", _target_exclusion(sample, "distillation")["reason"])
            self.assertTrue(all(call[2] == (96, 128, 3) and call[3]["reference_label"] == "display" for call in calls))
            self.assertEqual(load_dataset(dataset), manifest)

    def test_missing_display_never_falls_back_to_stereo_inference(self):
        with tempfile.TemporaryDirectory() as directory, patch("ipde.registration.estimate_display_registration") as estimate:
            calls = []
            with self.assertRaisesRegex(DatasetError, "full display image; stereo-view fallback is disabled"):
                self._build(Path(directory), capture=calibrated_capture, factory=lambda config: calls.append(config))
            self.assertFalse(calls)
            estimate.assert_not_called()

    def test_provided_stereo_results_are_not_used_as_display_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def factory(config):
                def predict(rgb, **values):
                    calls.append(rgb.shape)
                    return _prediction(rgb)
                return predict
            sources = [root / f"capture-{index}.heic" for index in range(3)]
            left_results = {str(source): _prediction(calibrated_capture(source).assets[0].array) for source in sources}
            _, manifest = self._build(root, factory=factory, teacher_results=left_results)
            self.assertEqual(calls, [(9, 12, 3)] * 3)
            self.assertTrue(all(sample["teacher_view"] == "display" for sample in manifest["samples"]))

    def test_provided_display_prediction_must_match_display_rgb_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture-0.heic"
            result = _prediction(_capture(source).assets[-1].array)
            result.metadata["input_rgb_sha256"] = sha256_array(calibrated_capture(source).assets[0].array)
            with self.assertRaisesRegex(DatasetError, "does not match the extracted display"):
                self._build(root, display_teacher_results={str(source): result})

    def test_display_alias_cannot_become_native_teacher_even_when_grids_match(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._build(Path(directory), capture=lambda source: _capture(source, display_shape=(3, 4)))
            for sample in manifest["samples"]:
                sample.pop("training_target_choice")
                self.assertIsNotNone(_target_exclusion(sample, "distillation"))
                sample["training_target_choice"] = "teacher"
                self.assertIsNotNone(_target_exclusion(sample, "distillation"))

    def test_measured_left_reference_remains_separate_and_supervised_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npy"
            np.save(reference, np.full((3, 4), 3., np.float64))
            paths = {str(root / f"capture-{index}.heic"): reference for index in range(3)}
            dataset, manifest = self._build(root, options=DatasetOptions(teacher=LearnedDepthConfig(),
                reference_paths=paths, workers=1))
            self.assertEqual(manifest["samples"][0]["teacher"]["target"]["shape"], [9, 12])
            self.assertEqual(manifest["samples"][0]["reference"]["target"]["shape"], [3, 4])
            self.assertTrue(training_target_eligibility(manifest, "supervised", require_display_teacher=True)["trainable"])
            self.assertEqual(load_dataset(dataset), manifest)

    def test_explicit_legacy_stereo_left_remains_available_without_display(self):
        with tempfile.TemporaryDirectory() as directory:
            def native_capture(source):
                capture = _capture(source)
                capture.assets.pop()
                return capture
            _, manifest = self._build(Path(directory), capture=native_capture,
                options=DatasetOptions(teacher=LearnedDepthConfig(), teacher_view="stereo-left", include_display_teacher=False, workers=1))
            self.assertEqual(manifest["teacher_view"], "stereo-left")
            self.assertTrue(training_target_eligibility(manifest)["trainable"])
            guarded = training_target_eligibility(manifest, require_display_teacher=True)
            self.assertFalse(guarded["trainable"])
            self.assertTrue(all("Regenerate display-only" in record["reason"] for record in guarded["excluded"]))

    def test_invalid_teacher_view_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as directory, patch("ipde.dataset.discover_file") as discover:
            root = Path(directory)
            source = root / "capture.heic"
            source.write_bytes(b"capture")
            with self.assertRaisesRegex(DatasetError, "teacher_view"):
                build_dataset([source], root / "dataset", replace(DatasetOptions(teacher=LearnedDepthConfig()), teacher_view="right"))
            discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
