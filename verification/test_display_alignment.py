"""Display images must not silently become native stereo training labels."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ipde.dataset import DatasetError, DatasetOptions, load_dataset
from ipde.dataset_review import curate_dataset, preview_sample, review_dataset
from ipde.registration import _fit_matches, _sample_depth, register_display_depth
from test_training_progress import _dataset


class DisplayAlignmentTests(unittest.TestCase):
    def test_native_left_teacher_is_the_default_even_with_display_diagnostics(self):
        self.assertFalse(DatasetOptions(teacher=None).prefer_registered_display_teacher)

    def test_multiple_feature_orientations_do_not_inflate_independent_evidence(self):
        positions = np.random.default_rng(3).uniform([0, 0], [799, 599], (24, 2))
        display = np.repeat(positions * 2, 4, axis=0)
        stereo = np.repeat(positions, 4, axis=0)
        result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
        self.assertEqual(result["feature_match_count"], 96)
        self.assertEqual(result["unique_feature_match_count"], 24)
        self.assertFalse(result["accepted"])
        stereo[:, 0] += np.tile(np.arange(4) * .01, 24)
        result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
        self.assertEqual(result["unique_feature_match_count"], 24)
        self.assertFalse(result["accepted"])

    def test_distributed_same_camera_affine_fit_retains_valid_registration(self):
        stereo = np.random.default_rng(3).uniform([0, 0], [799, 599], (1024, 2))
        display = (stereo - [3., -2.]) * 2
        result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
        self.assertTrue(result["accepted"])
        np.testing.assert_allclose(result["display_to_stereo_homography"],
            [[.5, 0., 3.], [0., .5, -2.], [0., 0., 1.]], atol=1e-5)
        self.assertGreaterEqual(len(result["stereo_inlier_hull"]), 3)

    def test_validated_cells_do_not_allow_extrapolation_outside_feature_hull(self):
        record = {"accepted": True, "source_sha256": "fixture", "reference_role": "left",
            "display_shape": [8, 8], "stereo_shape": [8, 8],
            "display_to_stereo_homography": np.eye(3).tolist(),
            "stereo_to_display_homography": np.eye(3).tolist(),
            "supported_cells": np.ones((4, 4), bool).tolist(),
            "stereo_inlier_hull": [[1., 1.], [6., 1.], [6., 6.], [1., 6.]]}
        depth = np.full((8, 8), 2., np.float32)
        original = depth.tobytes()
        expected = np.zeros((8, 8), bool)
        expected[1:7, 1:7] = True
        for hull in (record["stereo_inlier_hull"], record["stereo_inlier_hull"][::-1]):
            record["stereo_inlier_hull"] = hull
            output, valid = register_display_depth(depth, SimpleNamespace(source_sha256="fixture"), record)
            np.testing.assert_array_equal(valid, expected)
            self.assertTrue(np.isnan(output[~valid]).all())
            np.testing.assert_array_equal(output[valid], depth[valid])
        self.assertEqual(depth.tobytes(), original)

    def test_every_positive_interpolation_weight_requires_finite_data(self):
        depth = np.array([[2., np.nan]], np.float32)
        values, valid = _sample_depth(depth, np.array([0., 1e-9]), np.array([0., 0.]))
        self.assertEqual(values[0], 2.)
        self.assertTrue(valid[0])
        self.assertTrue(np.isnan(values[1]))
        self.assertFalse(valid[1])

    def test_rejected_registration_explains_failures_in_pixels_and_coverage(self):
        rng = np.random.default_rng(7)
        display = rng.uniform([0, 0], [1599, 1199], (1024, 2))
        stereo = display / 2
        stereo[:, 0] += rng.uniform(0, 25, len(stereo))
        result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
        self.assertFalse(result["accepted"])
        self.assertIn("held-out median", result["reason"])
        self.assertIn("px", result["reason"])
        self.assertIn("spatial cells", result["reason"])

    def test_display_rejection_is_visible_without_changing_selected_native_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            path = dataset / "dataset.json"
            manifest = json.loads(path.read_text())
            registration = {"accepted": False, "reference_role": "right", "reason": "held-out p90 13.20 px exceeds 3 px",
                "heldout_median_error_pixels": 2.15, "heldout_p90_error_pixels": 13.20,
                "display_shape": [128, 192], "stereo_shape": [64, 96]}
            manifest["samples"][0]["display_registration"] = registration
            path.write_text(json.dumps(manifest))
            before = {file.name: file.read_bytes() for file in dataset.iterdir() if file.is_file()}
            report = review_dataset(dataset)
            sample = report["samples"][0]
            self.assertEqual(sample["training_target_choice"], "teacher")
            self.assertTrue(sample["training_ready"])
            self.assertEqual(sample["rgb_reference"], "spatial_left")
            self.assertEqual(sample["display_registration"], registration)
            self.assertTrue(any("Display alignment was rejected" in warning and "13.20 px" in warning for warning in sample["warnings"]))
            preview = preview_sample(dataset, sample["id"], "training", root / "previews")
            self.assertEqual(preview["rgb_reference"], "spatial_left")
            self.assertEqual(preview["display_registration"], registration)
            self.assertEqual(before, {file.name: file.read_bytes() for file in dataset.iterdir() if file.is_file()})

    def test_parallel_curation_preserves_bytes_and_rolls_back_failed_copy(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            before = {file.name: file.read_bytes() for file in dataset.iterdir() if file.is_file()}
            original_copy = shutil.copyfile
            barrier = threading.Barrier(2)
            observed: set[int] = set()
            guard = threading.Lock()
            def concurrent_copy(source, target):
                with guard:
                    identity = threading.get_ident()
                    synchronize = identity not in observed
                    observed.add(identity)
                # Only synchronize each worker's first file; later copies need
                # no barrier and the array writer retains exact bytes.
                if synchronize:
                    barrier.wait(timeout=5)
                return original_copy(source, target)
            with patch("ipde.dataset_review.shutil.copyfile", side_effect=concurrent_copy):
                output = root / "reviewed"
                curate_dataset(dataset, ["sample-0"], output, workers=2)
            self.assertEqual(len(observed), 2)
            saved = load_dataset(output, workers=2)
            self.assertEqual([sample["id"] for sample in saved["samples"]], ["sample-0"])
            self.assertEqual(saved["samples"][0]["split"], "train")
            for file in output.iterdir():
                if file.name != "dataset.json":
                    self.assertEqual(file.read_bytes(), before[file.name])
            self.assertEqual(before, {file.name: file.read_bytes() for file in dataset.iterdir() if file.is_file()})
            failed = root / "failed"
            with patch("ipde.dataset_review.shutil.copyfile", side_effect=OSError("fixture copy failure")):
                with self.assertRaisesRegex(OSError, "fixture copy failure"):
                    curate_dataset(dataset, ["sample-0"], failed, workers=2)
            self.assertFalse(failed.exists())
            self.assertFalse(list(root.glob(".failed-*")))
            with self.assertRaises(DatasetError):
                curate_dataset(dataset, ["sample-0"], root / "invalid", workers=-1)


if __name__ == "__main__":
    unittest.main()
