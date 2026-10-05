"""Oversized DA3 requests fail before model loading or scientific-data writes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.learned_depth import (DA3_MAX_PATCH_TOKENS, LearnedDepthConfig,
                               LearnedDepthError, LearnedDepthPredictor,
                               infer_learned_depth, validate_learned_depth_input)


class DA3AdmissionTests(unittest.TestCase):
    def test_iphone_native_request_preserves_full_model_grid(self):
        self.assertEqual(validate_learned_depth_input((4284, 5712, 3),
            LearnedDepthConfig(model="depth-anything-3", input_size=0)), (4284, 5712))

    def test_unsupported_request_rejected_before_rgb_copy_and_model_load(self):
        rgb = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (8192, 8192, 3))
        with patch("ipde.learned_depth._validate_rgb") as validate, \
                patch("ipde.learned_depth._cached_predictor") as load:
            with self.assertRaisesRegex(LearnedDepthError, r"342,225.*never silently"):
                infer_learned_depth(rgb, LearnedDepthConfig(model="depth-anything-3", input_size=0))
        validate.assert_not_called()
        load.assert_not_called()

    def test_supported_grid_matches_upstream_rounding_and_keeps_native_small(self):
        self.assertEqual(validate_learned_depth_input((4284, 5712, 3),
            LearnedDepthConfig(model="depth-anything-3", input_size=1036)), (784, 1036))
        self.assertEqual(validate_learned_depth_input((5712, 4284, 3),
            LearnedDepthConfig(model="depth-anything-3", input_size=1036)), (1036, 784))
        self.assertEqual(validate_learned_depth_input((42, 56, 3),
            LearnedDepthConfig(model="depth-anything-3", input_size=0)), (42, 56))
        self.assertEqual(validate_learned_depth_input((4284, 5712, 3),
            LearnedDepthConfig(model="depth-anything-3", input_size=1536)), (1148, 1540))
        with self.assertRaisesRegex(LearnedDepthError, "342,225"):
            validate_learned_depth_input((8192, 8192, 3),
                LearnedDepthConfig(model="depth-anything-3", input_size=0))

    def test_changed_processor_cannot_dispatch_oversized_attention(self):
        import torch
        predictor = LearnedDepthPredictor.__new__(LearnedDepthPredictor)
        predictor.config = LearnedDepthConfig(model="depth-anything-3", input_size=1036)
        predictor.torch, predictor.device = torch, "cpu"
        oversized = torch.zeros(1, 3, 1, 1).expand(1, 3, 8190, 8190)
        network = unittest.mock.Mock()
        predictor.model = SimpleNamespace(input_processor=lambda *a, **k: (oversized, None, None), model=network)
        with self.assertRaisesRegex(LearnedDepthError, "processor exceeded"):
            predictor._depth_anything_3(np.zeros((42, 56, 3), np.uint8), 255)
        network.assert_not_called()

    def test_selected_photo_refusal_keeps_dataset_bytes_and_avoids_model_load(self):
        from test_dataset_teachers import DatasetTeacherTests
        from ipde.dataset_teachers import generate_teacher
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = DatasetTeacherTests()._dataset(Path(directory))
            before = {path.relative_to(dataset): path.read_bytes()
                      for path in dataset.rglob("*") if path.is_file()}
            with patch("ipde.dataset.require_dataset_teacher"), patch("ipde.learned_depth.LearnedDepthPredictor") as load:
                with self.assertRaisesRegex(LearnedDepthError, "patch limit"):
                    generate_teacher(dataset, [manifest["samples"][0]["id"]],
                        LearnedDepthConfig(model="depth-anything-3", input_size=8192))
            load.assert_not_called()
            self.assertEqual(before, {path.relative_to(dataset): path.read_bytes()
                                     for path in dataset.rglob("*") if path.is_file()})
            self.assertFalse(any(dataset.parent.glob(".teacher-generation-*")))


if __name__ == "__main__":
    unittest.main()
