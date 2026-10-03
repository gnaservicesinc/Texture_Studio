from __future__ import annotations

import unittest

import numpy as np

from ipde.formats import arrays_bit_equal
from ipde.pseudo_calibration import PseudoCalibrationError, anchor_relative_depth


class PseudoCalibrationTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:128, :192]
        self.relative = (1 + x / 50 + y / 90).astype(np.float32)
        self.metric = (1 / (.3 * self.relative + .2)).astype(np.float32)

    def test_inverse_teacher_recovers_positive_affine_scale_without_changing_raw_bits(self):
        self.relative[0, 0] = np.nan
        before_relative, before_metric = self.relative.copy(), self.metric.copy()
        output, valid, metadata = anchor_relative_depth(self.relative, "relative_inverse_depth", self.metric)
        np.testing.assert_allclose(output[valid], self.metric[valid], rtol=1e-6)
        self.assertFalse(valid[0, 0])
        self.assertTrue(np.isnan(output[0, 0]))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(arrays_bit_equal(self.relative, before_relative))
        self.assertTrue(arrays_bit_equal(self.metric, before_metric))
        self.assertAlmostEqual(metadata["scale_inverse_meters_per_relative_inverse_unit"], .3, places=6)
        self.assertAlmostEqual(metadata["offset_inverse_meters"], .2, places=6)
        self.assertEqual(len(metadata["validation"]["spatial_tiles"]), 64)
        self.assertFalse(metadata["anchor_is_measured"])

    def test_relative_depth_teacher_is_reciprocated_before_fitting(self):
        output, valid, metadata = anchor_relative_depth(1 / self.relative, "relative_depth", self.metric)
        np.testing.assert_allclose(output[valid], self.metric[valid], rtol=1e-6)
        self.assertEqual(metadata["relative_inverse_formula"], "1/relative")

    def test_robust_fit_preserves_relative_geometry_despite_anchor_outliers(self):
        anchor = self.metric.copy()
        rng = np.random.default_rng(3)
        outliers = rng.random(anchor.shape) < .08
        anchor[outliers] *= 5
        output, valid, metadata = anchor_relative_depth(self.relative, "relative_inverse_depth", anchor)
        self.assertLess(np.median(np.abs(output[valid] / self.metric[valid] - 1)), .01)
        self.assertLess(metadata["validation"]["median_relative_depth_error"], .02)

    def test_rejects_bad_grid_units_constant_negative_scale_and_weak_agreement(self):
        rng = np.random.default_rng(7)
        cases = [
            (self.relative, "meters", self.metric),
            (self.relative, "relative_depth", self.metric[:1]),
            (self.relative.astype(np.uint8), "relative_inverse_depth", self.metric),
            (np.ones_like(self.relative), "relative_inverse_depth", self.metric),
            (self.relative, "relative_inverse_depth", np.ones_like(self.metric)),
            (self.relative, "relative_inverse_depth", self.metric[:, ::-1]),
            (self.relative, "relative_inverse_depth", rng.uniform(.2, 5, self.metric.shape)),
            (self.relative, "relative_inverse_depth", np.full_like(self.metric, np.nan)),
        ]
        for relative, units, anchor in cases:
            with self.subTest(units=units, shape=anchor.shape), self.assertRaises(PseudoCalibrationError):
                anchor_relative_depth(relative, units, anchor)

    def test_validation_is_spatially_balanced_and_rejects_local_geometry_disagreement(self):
        distorted = self.metric.copy()
        # Four quadrants disagree systematically: a global median fit alone can
        # conceal regional distortion, so every tile retains a held-out report.
        distorted[:, :96] *= 4
        with self.assertRaises(PseudoCalibrationError):
            anchor_relative_depth(self.relative, "relative_inverse_depth", distorted)


if __name__ == "__main__":
    unittest.main()
