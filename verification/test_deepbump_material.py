"""CPU regression coverage for pinned baseline data and coordinate contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_deepbump_material as baseline


class DeepBumpMaterialTests(unittest.TestCase):
    def test_uint16_display_input_keeps_codes_above_eight_bit(self):
        codes = np.full((2, 3, 3), 32768, np.uint16)
        codes[0, 0, 0] += 1
        rgb, valid = baseline.display_rgb(codes, "source_srgb_assumed")
        self.assertEqual(rgb.dtype, np.float32)
        self.assertGreater(rgb[0, 0, 0], rgb[0, 0, 1])
        self.assertTrue(valid.all())
        self.assertEqual(int(codes[0, 0, 0]), 32769)

    def test_only_declared_linear_diffuse_is_encoded_for_model(self):
        codes = np.full((2, 2, 3), 16384, np.uint16)
        srgb, _ = baseline.display_rgb(codes, "srgb")
        linear, _ = baseline.display_rgb(codes, "linear_color")
        self.assertAlmostEqual(float(srgb[0, 0, 0]), 16384 / 65535, places=6)
        self.assertGreater(float(linear[0, 0, 0]), 0.53)
        with self.assertRaisesRegex(ValueError, "explicit"):
            baseline.display_rgb(codes, "unknown")

    def test_reference_normal_tags_never_trigger_gamma(self):
        codes = np.full((2, 2, 3), 32768, np.uint16)
        codes[..., 2] = 65535
        normal = baseline.normal_codes(codes)
        self.assertAlmostEqual(float(normal[0, 0, 0]), 32768 / 65535, places=6)
        self.assertEqual(float(normal[0, 0, 2]), 1)

    def test_near_opaque_alpha_tolerated_without_changing_rgb(self):
        codes = np.full((20, 20, 4), 65535, np.uint16)
        codes[..., 3] -= 4
        rgb, valid = baseline.display_rgb(codes, "srgb")
        self.assertTrue(valid.all())
        self.assertTrue((rgb == 1).all())

    def test_large_transparency_rejected(self):
        codes = np.full((30, 30, 4), 65535, np.uint16)
        codes[:10, :, 3] = 0
        with self.assertRaisesRegex(ValueError, "10%"):
            baseline.display_rgb(codes, "srgb")

    def test_height_integration_preserves_opengl_ramp_axes_and_amplitude(self):
        y, x = np.mgrid[:19, :23]
        for px, py in ((0.3, 0), (0, -0.2), (-0.3, 0.2)):
            vector = np.array([-px, py, 1.0])
            vector /= np.linalg.norm(vector)
            normal = np.broadcast_to(vector / 2 + 0.5, (19, 23, 3))
            integrated = baseline.integrate_opengl_normals(normal)
            expected = px * x + py * y
            expected -= expected.mean()
            np.testing.assert_allclose(integrated, expected, atol=1e-6)
            self.assertEqual(integrated.dtype, np.float32)
            if px or py:
                self.assertGreater(float(np.ptp(integrated)), 1)

    def test_angular_metrics_exclude_invalid_pixels(self):
        predicted = np.tile([0.5, 0.5, 1.0], (2, 2, 1))
        reference = predicted.copy()
        reference[0, 0] = [1, 0.5, 0.5]
        valid = np.ones((2, 2), bool)
        valid[0, 0] = False
        self.assertEqual(baseline.angular_metrics(predicted, reference, valid)["mean_degrees"], 0)
        self.assertAlmostEqual(baseline.angular_metrics(predicted, reference, np.ones_like(valid))["mean_degrees"], 22.5)

    def test_undefined_reference_normals_only_rejected_when_valid(self):
        predicted = np.tile([0.5, 0.5, 1.0], (2, 2, 1))
        reference = predicted.copy()
        valid = np.ones((2, 2), bool)
        valid[0, 0] = False
        for value in ((0.5, 0.5, 0.5), (np.nan, np.nan, np.nan)):
            reference[0, 0] = value
            self.assertEqual(baseline.angular_metrics(predicted, reference, valid)["mean_degrees"], 0)
            with self.assertRaisesRegex(ValueError, "zero-length normal"):
                baseline.angular_metrics(predicted, reference, np.ones_like(valid))

    def test_scalar_height_rejects_alpha_and_unequal_rgb(self):
        codes = np.full((2, 2, 4), 32768, np.uint16)
        codes[..., 3] = 65535
        values = baseline.scalar_height(codes)
        self.assertAlmostEqual(float(values[0, 0]), 32768 / 65535, places=6)
        codes[0, 0, 3] = 65534
        with self.assertRaisesRegex(ValueError, "Transparent"):
            baseline.scalar_height(codes)
        codes[..., 3] = 65535
        codes[0, 0, 1] += 1
        with self.assertRaisesRegex(ValueError, "identical RGB"):
            baseline.scalar_height(codes)

    def test_arrays_validate_native_dimensions_bits_and_data_transfer(self):
        arrays = {"input": np.full((3, 2, 3), 32768, np.uint16), "normal": np.full((3, 2, 3), 32768, np.uint16), "height": np.full((3, 2), 32768, np.uint16)}
        arrays["normal"][..., 2] = 65535
        metadata = {"sample_pixel_dimensions": [2, 3], "map_metadata": {role: {"sample_bits": 16, "encoding": "srgb" if role == "input" else "linear_data"} for role in arrays}}
        rgb, valid, normal, height = baseline.validate_arrays(arrays, metadata)
        self.assertEqual(rgb.shape, (3, 3, 2))
        self.assertEqual(normal.shape, (3, 2, 3))
        self.assertEqual(height.shape, (3, 2))
        self.assertTrue(valid.all())
        metadata["sample_pixel_dimensions"] = [3, 2]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            baseline.validate_arrays(arrays, metadata)
        metadata["sample_pixel_dimensions"] = [2, 3]
        metadata["map_metadata"]["normal"]["encoding"] = "srgb"
        with self.assertRaisesRegex(ValueError, "linear_data"):
            baseline.validate_arrays(arrays, metadata)
        metadata["map_metadata"]["normal"]["encoding"] = "linear_data"
        metadata["map_metadata"]["input"]["sample_bits"] = 8
        with self.assertRaisesRegex(ValueError, "precision"):
            baseline.validate_arrays(arrays, metadata)

    def test_model_blob_mismatch_rejected_before_execution(self):
        payload = b"model data"
        identity = baseline.git_blob_identity(payload)
        with tempfile.TemporaryDirectory() as directory, patch.object(baseline, "FILES", {"model.onnx": (len(payload), identity)}):
            path = Path(directory) / "model.onnx"
            path.write_bytes(payload)
            self.assertEqual(baseline.model_files(Path(directory), False)[0]["git_blob_sha1"], identity)
            path.write_bytes(b"model datA")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                baseline.model_files(Path(directory), False)

    def test_float_exr_round_trip_no_quantization(self):
        values = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 100003
        with tempfile.TemporaryDirectory() as directory:
            baseline.write_float_exr(Path(directory) / "normal.exr", values)
            baseline.write_float_exr(Path(directory) / "height.exr", values[..., 0])

    def test_exr_rejects_wrong_channels_and_nonfinite_data(self):
        with tempfile.TemporaryDirectory() as directory:
            for values in (np.zeros((2, 2, 4)), np.zeros((2, 2, 1)), np.full((2, 2), np.nan)):
                with self.assertRaisesRegex(ValueError, "finite nonempty scalar or RGB"):
                    baseline.write_float_exr(Path(directory) / "invalid.exr", values)

    def test_selection_requires_validation_scope_and_consistent_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "samples/example"
            sample.mkdir(parents=True)
            row = {"material_id": "example", "sample_id": "example_003", "split": "validation", "status": "prepared", "path": "samples/example"}
            metadata = row | {"crop_values_verified": True, "source_precision_verified": True, "normal_convention": "OpenGL +Y"}
            index = {"split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "samples": [row]}
            (root / "dataset.json").write_text(json.dumps(index))
            (sample / "sample.json").write_text(json.dumps(metadata))
            self.assertEqual(len(baseline.select_samples(root, ["example"])[1]), 1)
            metadata["split"] = "train"
            (sample / "sample.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "differ"):
                baseline.select_samples(root, ["example"])

    def test_budget_checked_before_inference(self):
        class Helpers:
            @staticmethod
            def tiles_split(*args):
                return [np.zeros((1, 256, 256), np.float32)], (0, 0, 0, 0)

        class Session:
            def run(self, *args):
                raise AssertionError("Expired budget must not execute a tile")

        with self.assertRaises(TimeoutError):
            baseline.infer_normals(np.zeros((3, 256, 256), np.float32), Session(), Helpers(), time.monotonic() - 1, "LARGE")

    def test_inference_rejects_wrong_normal_tile_channels(self):
        class Helpers:
            @staticmethod
            def tiles_split(*args):
                return [np.zeros((1, 256, 256), np.float32)], (0, 0, 0, 0)

        class Session:
            def run(self, *args):
                return [np.zeros((1, 4, 256, 256), np.float32)]

        with self.assertRaisesRegex(ValueError, "tile shape"):
            baseline.infer_normals(np.zeros((3, 256, 256), np.float32), Session(), Helpers(), time.monotonic() + 60, "LARGE")


if __name__ == "__main__":
    unittest.main()
