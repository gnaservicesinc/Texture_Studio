"""Scientific AI outputs retain float values and the selected camera grid."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.extractor import Asset, Discovery, ExtractOptions, ExtractionError, extract_file, inspect_file
from ipde.formats import read_exr_exact, read_png_exact
from ipde.learned_depth import LearnedDepthResult


class LearnedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "photo.heic"
        self.source.write_bytes(b"fixture")
        self.left = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.display = np.ones((4, 6, 3), np.uint8)
        self.discovery = Discovery(self.source, 7, "hash", "image/heic", 0, [], [
            Asset("spatial_view", 1, 0, self.left, "RGB", 8, "spatial_left"),
            Asset("display_view", 0, 0, self.display, "RGB", 8, "display"),
        ], spatial_photo={"focal_length_pixels_for_depth": 10.0})
        self.patcher = patch("ipde.extractor.discover_file", return_value=self.discovery)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def export(self, *products, **kwargs):
        return extract_file(self.source, ExtractOptions(selected_products=products,
            output_dir=self.root / "out", write_npy=False, **kwargs))

    def test_float32_values_are_not_normalized_and_native_is_separate(self):
        depth = np.array([[.501234, 1.41421, 2.13131], [3.777777, 5.223, np.nan]], np.float32)
        native = np.array([[.65, 1.15], [2.02, 4.33]], np.float32)
        result = LearnedDepthResult(native, depth, {"units": "meters"})
        with patch("ipde.extractor.infer_learned_depth", return_value=result) as infer:
            self.export("learned-depth", "learned-native")
        infer.assert_called_once()
        self.assertEqual(infer.call_args.kwargs["focal_pixels"], 10.0)
        np.testing.assert_array_equal(read_exr_exact(self.root / "out/photo_spatial_left_depthpro_depth.exr", (2, 3)), depth)
        np.testing.assert_array_equal(read_exr_exact(self.root / "out/photo_spatial_left_depthpro_native_depth.exr", (2, 2)), native)
        np.testing.assert_array_equal(self.left, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
        self.assertEqual(len(list((self.root / "out").iterdir())), 2)

    def test_display_uses_its_own_grid_and_does_not_reuse_stereo_focal(self):
        result = LearnedDepthResult(np.ones((2, 2), np.float32), np.ones((4, 6), np.float32), {"units": "meters"})
        with patch("ipde.extractor.infer_learned_depth", return_value=result) as infer:
            self.export("learned-display-depth")
        np.testing.assert_array_equal(infer.call_args.args[0], self.display)
        self.assertIsNone(infer.call_args.kwargs["focal_pixels"])
        self.assertEqual(infer.call_args.kwargs["reference_label"], "display")
        products = {p["id"]: p for p in inspect_file(self.source, include_learned=True)["available_products"]}
        self.assertEqual(products["learned-display-depth"]["width"], 6)
        self.assertEqual(products["learned-depth"]["width"], 3)

    def test_relative_depth_and_inverse_depth_preview_directions_differ(self):
        depth = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
        for units, expected_near in (("relative_depth", (0, 0)), ("relative_inverse_depth", (1, 2))):
            result = LearnedDepthResult(depth, depth, {"units": units})
            with patch("ipde.extractor.infer_learned_depth", return_value=result):
                self.export("learned-preview", overwrite=True)
            preview = read_png_exact(self.root / "out/photo_spatial_left_depthpro_depth_preview.png")
            self.assertEqual(int(preview[expected_near][0]), 65535)
            self.assertTrue((preview[:, :, 1] == 65535).all())
            np.testing.assert_array_equal(depth, [[1, 2, 3], [4, 5, 6]])

    def test_collision_is_checked_before_model_runs(self):
        out = self.root / "out"
        out.mkdir()
        (out / "photo_spatial_left_depthpro_depth.exr").write_bytes(b"existing")
        with patch("ipde.extractor.infer_learned_depth") as infer:
            with self.assertRaisesRegex(ExtractionError, "already exists"):
                self.export("learned-depth")
        infer.assert_not_called()

    def test_raw_selection_does_not_load_ai_model(self):
        with patch("ipde.extractor.infer_learned_depth") as infer:
            self.export("raw:0")
        infer.assert_not_called()

    def test_ipde_inventory_keeps_teacher_training_choices_out(self):
        products = inspect_file(self.source)["available_products"]
        self.assertFalse(any(product["id"].startswith("learned-") for product in products))

    def test_relative_teacher_can_resolve_its_selected_local_resources(self):
        result = LearnedDepthResult(np.ones((2, 3), np.float32), np.ones((2, 3), np.float32), {"units": "relative_depth"})
        with patch("ipde.extractor.infer_learned_depth", return_value=result) as infer:
            self.export("learned-native", learned_model="depth-anything-3")
        configuration = infer.call_args.args[1]
        self.assertEqual(configuration.model, "depth-anything-3")
        self.assertIsNone(configuration.model_path)
        self.assertIsNone(configuration.source_dir)
