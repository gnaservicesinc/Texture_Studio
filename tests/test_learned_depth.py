"""Numerical and model-contract tests without downloading test weights."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ipde.formats import sha256_array
from ipde.learned_depth import (
    LearnedDepthConfig, LearnedDepthError, LearnedDepthPredictor,
    _checkpoint_state, _fp32_math, _load_safetensors_model, _prepare_rgb, infer_learned_depth,
    resolve_learned_depth_resources,
)


class LearnedDepthInputTests(unittest.TestCase):
    def test_cuda_tf32_is_disabled_and_caller_flags_restored_even_on_error(self):
        from contextlib import nullcontext
        fake = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
            autocast=lambda **kwargs: nullcontext(),
        )
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with _fp32_math(fake, "cuda"):
                self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
                self.assertFalse(fake.backends.cudnn.allow_tf32)
                raise RuntimeError("fixture")
        self.assertTrue(fake.backends.cuda.matmul.allow_tf32)
        self.assertTrue(fake.backends.cudnn.allow_tf32)

    def test_import_has_no_torch_or_network_side_effect(self):
        source = Path(__file__).resolve().parents[1] / "src"
        process = subprocess.run(
            [sys.executable, "-c", "import sys; import ipde.learned_depth; assert 'torch' not in sys.modules"],
            env={**os.environ, "PYTHONPATH": str(source)}, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_higher_bit_rgb_uses_nominal_scale_without_quantization_or_mutation(self):
        rgb = np.array([[[1023, 1001, 1000], [0, 1, 2]]], dtype=np.uint16)
        before = rgb.copy()
        output, limit = _prepare_rgb(rgb, LearnedDepthConfig(input_max_value=1023))
        np.testing.assert_array_equal(rgb, before)
        self.assertEqual(limit, 1023)
        self.assertEqual(output.dtype, np.float32)
        self.assertGreater(output[0, 0, 1], output[0, 0, 2])
        self.assertEqual(output[0, 1, 0], 0)
        self.assertEqual(output[0, 0, 0], 1)

    def test_rgb_is_not_stretched_to_observed_range(self):
        rgb = np.full((2, 3, 3), 128, dtype=np.uint8)
        output, _ = _prepare_rgb(rgb, LearnedDepthConfig())
        np.testing.assert_array_equal(output, np.full(rgb.shape, np.float32(128 / 255)))

    def test_rejects_depth_plane_and_invalid_nominal_input(self):
        for value in (np.zeros((3, 4), np.float32), np.zeros((3, 4, 4), np.uint8),
                      np.full((3, 4, 3), np.nan, np.float32), np.full((3, 4, 3), 2, np.float32)):
            with self.assertRaises(LearnedDepthError):
                _prepare_rgb(value, LearnedDepthConfig())

    def test_explicit_missing_resources_fail_without_using_another_local_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.pt"
            with self.assertRaisesRegex(LearnedDepthError, "never downloads"):
                resolve_learned_depth_resources(LearnedDepthConfig(model_path=path))

    def test_da3_requires_both_local_config_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps({"model_name": "da3-giant"}))
            with self.assertRaisesRegex(LearnedDepthError, "model.safetensors"):
                resolve_learned_depth_resources(LearnedDepthConfig(model="depth-anything-3", model_path=path))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is not installed")
class LearnedDepthNumericalTests(unittest.TestCase):
    def make_predictor(self, model_name="depthpro"):
        import torch

        class Model(torch.nn.Module):
            img_size = 2

            def forward(self, image):
                self.last_input = image.detach().clone()
                self.autocast_enabled = torch.is_autocast_enabled("cpu")
                canonical = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]], dtype=torch.float32)
                return canonical, torch.tensor([60.0], dtype=torch.float32)

        predictor = LearnedDepthPredictor.__new__(LearnedDepthPredictor)
        predictor.config = LearnedDepthConfig(model=model_name, device="cpu")
        predictor.torch = torch
        predictor.device = "cpu"
        predictor.model = Model()
        predictor.base_metadata = {"checkpoint_sha256": "fixture", "computation_dtype": "float32"}
        return predictor

    def test_metric_source_resizes_inverse_depth_before_reciprocal(self):
        import torch
        predictor = self.make_predictor()
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        before = rgb.copy()
        result = predictor(rgb, focal_pixels=4, reference_label="spatial_left")
        inverse = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
        expected = 1 / torch.nn.functional.interpolate(inverse, (4, 4), mode="bilinear", align_corners=False)
        np.testing.assert_array_equal(result.source_depth, expected[0, 0].numpy())
        np.testing.assert_array_equal(result.native_depth, (1 / inverse)[0, 0].numpy())
        wrong = torch.nn.functional.interpolate(1 / inverse, (4, 4), mode="bilinear", align_corners=False)[0, 0].numpy()
        self.assertFalse(np.allclose(result.source_depth, wrong))
        np.testing.assert_array_equal(rgb, before)
        self.assertEqual(result.native_depth.dtype, np.float32)
        self.assertEqual(result.source_depth.dtype, np.float32)
        self.assertGreater(len(np.unique(result.source_depth)), 8)
        self.assertEqual(result.metadata["input_rgb_sha256"], sha256_array(rgb))
        self.assertEqual(result.metadata["units"], "meters")
        self.assertFalse(predictor.model.autocast_enabled)

    def test_estimated_focal_uses_original_reference_width(self):
        predictor = self.make_predictor()
        result = predictor(np.zeros((4, 8, 3), np.uint8))
        self.assertAlmostEqual(result.metadata["focal_pixels"], 4 / np.tan(np.deg2rad(30)), places=5)
        self.assertEqual(result.metadata["focal_source"], "DepthPro field-of-view estimate")

    def test_singleton_source_spatial_dimension_is_preserved(self):
        predictor = self.make_predictor()
        result = predictor(np.zeros((1, 4, 3), np.uint8), focal_pixels=4)
        self.assertEqual(result.source_depth.shape, (1, 4))

    def test_calibrated_focal_changes_metric_scale_and_no_display_normalization(self):
        predictor = self.make_predictor()
        rgb = np.zeros((4, 4, 3), np.uint8)
        first = predictor(rgb, focal_pixels=4)
        second = predictor(rgb, focal_pixels=12)
        np.testing.assert_array_equal(second.native_depth, first.native_depth * 3)
        self.assertGreater(second.native_depth.max(), 1)
        self.assertEqual(second.metadata["output_normalization"], "none")

    def test_fp16_checkpoint_provenance_is_observable(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"weight": torch.ones((4, 3), dtype=torch.float16)}, path)
            state, counts = _checkpoint_state(torch, path)
        self.assertEqual(counts, {"float16": 12})
        self.assertEqual(state["weight"].dtype, torch.float16)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "optional safetensors is not installed")
    def test_da3_shared_parameters_load_strictly_without_random_missing_weights(self):
        import torch
        from safetensors.torch import save_model, save_file

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.LayerNorm(3)
                self.second = self.first

        original = Shared()
        with torch.no_grad():
            original.first.weight[:] = torch.tensor([0.5, 1.5, 2.5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            save_model(original, path)
            restored = Shared()
            counts = _load_safetensors_model(restored, path)
            self.assertTrue(torch.equal(restored.second.weight, original.second.weight))
            self.assertEqual(counts, {"float32": 6})
            save_file({"first.weight": original.first.weight}, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                _load_safetensors_model(Shared(), path)

    def test_v2_uses_rgb_order_custom_device_and_relative_semantics(self):
        import torch
        predictor = self.make_predictor("depth-anything-v2")
        transformations = []

        class Resize:
            def __init__(self, **kwargs):
                transformations.append(kwargs)
            def __call__(self, sample):
                return sample

        class Normalize:
            def __init__(self, **kwargs):
                pass
            def __call__(self, sample):
                return sample

        class Prepare:
            def __call__(self, sample):
                sample["image"] = np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
                return sample

        class Model(torch.nn.Module):
            def forward(self, tensor):
                self.input = tensor.detach().clone()
                return torch.tensor([[[2.5, 8.75], [9.25, 12.125]]])

        predictor.module = SimpleNamespace(Resize=Resize, NormalizeImage=Normalize, PrepareForNet=Prepare)
        predictor.model = Model()
        rgb = np.empty((2, 2, 3), np.uint8)
        rgb[:] = [255, 128, 0]
        result = predictor(rgb)
        np.testing.assert_array_equal(predictor.model.input[0, :, 0, 0].numpy(), np.array([1, 128 / 255, 0], np.float32))
        self.assertEqual(predictor.model.input.device.type, "cpu")
        self.assertEqual(result.metadata["units"], "relative_inverse_depth")
        self.assertEqual(result.source_depth[1, 1], np.float32(12.125))
        self.assertEqual(transformations[0]["ensure_multiple_of"], 14)

    def test_da3_bypasses_autocasting_wrapper_and_keeps_native_confidence(self):
        import torch
        predictor = self.make_predictor("depth-anything-3")

        class Inner(torch.nn.Module):
            def forward(self, image, *args):
                self.autocast = torch.is_autocast_enabled("cpu")
                return {"depth": torch.full((1, 1, 14, 14), 2.125), "depth_conf": torch.full((1, 1, 14, 14), 1.75)}

        class Outer:
            model = Inner()
            def input_processor(self, images, **kwargs):
                self.input = images[0]
                return torch.zeros((1, 3, 14, 14)), None, None
            def __call__(self, *args):
                raise AssertionError("high-level wrapper enables mixed precision")

        predictor.model = Outer()
        result = predictor(np.zeros((28, 42, 3), np.uint8))
        self.assertEqual(result.metadata["units"], "relative_depth")
        self.assertEqual(result.native_depth.shape, (14, 14))
        self.assertEqual(result.source_depth.shape, (28, 42))
        self.assertEqual(result.confidence.shape, (14, 14))
        self.assertEqual(result.confidence[0, 0], 1.75)
        self.assertFalse(predictor.model.model.autocast)
        with self.assertRaisesRegex(LearnedDepthError, "uint8 RGB only"):
            predictor(np.zeros((28, 42, 3), np.uint16))


if __name__ == "__main__":
    unittest.main()
