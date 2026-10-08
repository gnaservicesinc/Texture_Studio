"""Guard diagnostic scale alignment and raw scalar-map handling."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('material_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_da3_material.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MaterialEvaluationTests(unittest.TestCase):
    def test_camera_z_alignment_has_outward_height_sign_and_declares_oracle(self):
        depth = np.linspace(1, 3, 1024, dtype=np.float32).reshape(32, 32)
        target = 1.2 - .3 * depth
        aligned, metadata = evaluation.optimistic_affine(depth, target)
        np.testing.assert_allclose(aligned, target, atol=1e-7)
        self.assertTrue(metadata['uses_heldout_ground_truth_for_alignment'])
        self.assertGreater(metadata['inverse_camera_z_slope'], 0)

    def test_flat_scene_prediction_cannot_manufacture_target_detail(self):
        target = np.linspace(.1, .9, 1024, dtype=np.float32).reshape(32, 32)
        aligned, metadata = evaluation.optimistic_affine(np.ones_like(target), target)
        self.assertEqual(metadata['inverse_camera_z_slope'], 0)
        self.assertGreater(evaluation.metrics(aligned, target)['gradient_mae_stride_1'], 0)

    def test_replicated_scalar_rgba_keeps_adjacent_uint16_codes(self):
        values = np.array([[32767, 32768], [32769, 32770]], dtype=np.uint16)
        rgba = np.dstack([values, values, values, np.full_like(values, 65535)])
        decoded = evaluation.scalar_codes(rgba)
        np.testing.assert_array_equal(decoded, values)
        self.assertEqual(decoded.dtype, np.uint16)

    def test_color_or_alpha_cannot_be_silently_turned_into_height(self):
        color = np.zeros((2, 2, 3), dtype=np.uint16)
        color[:, :, 1] = 1
        with self.assertRaises(ValueError):
            evaluation.scalar_codes(color)
        with self.assertRaises(ValueError):
            evaluation.scalar_codes(np.zeros((2, 2, 4), dtype=np.uint16))


if __name__ == '__main__':
    unittest.main()
