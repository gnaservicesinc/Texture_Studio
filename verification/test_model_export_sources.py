"""Built-in depth sources are one unmodified full-display float32 map each."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.extractor import (Asset, Discovery, ExtractOptions, ExtractionError,
                            extract_file, inspect_file)
from ipde.formats import read_exr_exact


class ModelExportSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "photo.heic"
        self.source.write_bytes(b"fixture")
        self.display = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        self.left = np.full((2, 3, 3), 47, np.uint8)
        self.auxiliary = np.array([[.125, -.5], [4.5, 27]], np.float16)
        self.discovery = Discovery(self.source, 7, "source-sha256", "image/heic", 0, [], [
            Asset("display_view", 0, 0, self.display, "RGB", 8, "display"),
            Asset("spatial_view", 1, 0, self.left, "RGB", 8, "spatial_left"),
            Asset("auxiliary", 0, 0, self.auxiliary, "F;16", 16, "portrait_matte"),
        ], spatial_photo={"focal_length_pixels_for_depth": 1575})
        # Include values normalization would change, signed zero and a NaN payload.
        self.prediction = np.nextafter(np.linspace(-7, 41, 24, dtype=np.float32),
                                       np.float32(np.inf)).reshape(4, 6)
        self.prediction.view(np.uint32)[0, :2] = [0x80000000, 0x7fc01234]

    def test_inventory_has_exactly_three_full_display_sources_and_preserves_auxiliary(self):
        full_display = np.broadcast_to(self.display[:1, :1], (4284, 5712, 3))
        self.discovery.assets[0].array = full_display
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                patch("ipde.extractor.infer_learned_depth") as infer:
            report = inspect_file(self.source, include_learned=True)
        infer.assert_not_called()
        products = report["available_products"]
        learned = [entry for entry in products if entry["id"].startswith("learned-")]
        self.assertEqual([(entry["id"], entry["name"]) for entry in learned], [
            ("learned-depthpro", "DepthPro"), ("learned-da3", "DA3"), ("learned-da2", "DA2")])
        self.assertEqual({(entry["width"], entry["height"], entry["precision"]) for entry in learned},
                         {(5712, 4284, "32-bit float EXR")})
        self.assertIn("raw:2", {entry["id"] for entry in products})

    def test_checked_models_generate_only_one_bit_exact_display_exr_each(self):
        originals = [array.copy() for array in (self.display, self.left, self.auxiliary, self.prediction)]
        calls = []

        def predict(image, config, **kwargs):
            self.assertIs(image, self.display)
            self.assertIsNone(kwargs["focal_pixels"])
            self.assertEqual(kwargs["reference_label"], "display")
            calls.append(config)
            # There is deliberately no native_depth: the export must never use it.
            return SimpleNamespace(source_depth=self.prediction, metadata={"model": config.model,
                "units": "meters" if config.model == "depthpro" else "relative_depth"})

        options = ExtractOptions(output_dir=self.directory / "exports", write_npy=False,
            selected_products=("learned-depthpro", "learned-da3", "learned-da2"),
            learned_model_path=Path("/models/depthpro.pt"), learned_source_dir=Path("/sources/depthpro"),
            learned_model_settings={"depth-anything-3": {"model_path": "/models/da3", "source_dir": "/sources/da3"},
                                    "depth-anything-v2": {"model_path": "/models/da2.pth"}})
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                patch("ipde.extractor.infer_learned_depth", side_effect=predict), \
                patch("ipde.extractor.write_png", side_effect=AssertionError("preview generated")), \
                patch("ipde.extractor.reconstruct_physical_disparity", side_effect=AssertionError("auxiliary calculation")), \
                patch("ipde.extractor.run_raft_stereo", side_effect=AssertionError("stereo generated")):
            report = extract_file(self.source, options)
        self.assertEqual({call.model for call in calls}, {"depthpro", "depth-anything-3", "depth-anything-v2"})
        by_model = {call.model: call for call in calls}
        self.assertEqual(by_model["depthpro"].model_path, Path("/models/depthpro.pt"))
        self.assertEqual(by_model["depth-anything-3"].model_path, Path("/models/da3"))
        self.assertEqual(by_model["depth-anything-v2"].model_path, Path("/models/da2.pth"))
        self.assertIsNone(by_model["depth-anything-v2"].source_dir)
        self.assertEqual(len(list(options.output_dir.iterdir())), 3)
        self.assertEqual(len(report["assets"][0]["outputs"]), 3)
        self.assertFalse(report["assets"][1]["outputs"] or report["assets"][2]["outputs"])
        for output in report["assets"][0]["outputs"]:
            self.assertFalse(output["derivation"]["normalization"])
            self.assertEqual(output["derivation"]["reference_image"], "display")
            actual = read_exr_exact(Path(output["path"]), self.prediction.shape)
            np.testing.assert_array_equal(actual.view(np.uint32), self.prediction.view(np.uint32))
        for array, original in zip((self.display, self.left, self.auxiliary, self.prediction), originals):
            np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))

    def test_legacy_depth_alias_selects_the_current_model_once(self):
        result = SimpleNamespace(source_depth=self.prediction, metadata={"units": "relative_inverse_depth"})
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                patch("ipde.extractor.infer_learned_depth", return_value=result) as infer:
            report = extract_file(self.source, ExtractOptions(output_dir=self.directory / "legacy",
                learned_model="depth-anything-v2", selected_products=("learned-display-depth", "learned-depth", "learned-da2")))
        infer.assert_called_once()
        self.assertEqual(report["selected_products"], ["learned-da2"])
        outputs = report["assets"][0]["outputs"]
        self.assertEqual(len(outputs), 2)  # one EXR plus the requested exact NumPy companion
        self.assertEqual({Path(output["path"]).suffix for output in outputs}, {".exr", ".npy"})
        npy = next(Path(output["path"]) for output in outputs if output["path"].endswith(".npy"))
        np.testing.assert_array_equal(np.load(npy, allow_pickle=False).view(np.uint32), self.prediction.view(np.uint32))

    def test_removed_subimage_selectors_do_not_trigger_inference(self):
        for product in ("learned-native", "learned-preview", "learned-displacement",
                        "learned-display-native", "learned-display-preview", "learned-display-displacement"):
            with self.subTest(product=product), patch("ipde.extractor.discover_file", return_value=self.discovery), \
                    patch("ipde.extractor.infer_learned_depth") as infer:
                with self.assertRaisesRegex(ExtractionError, "unavailable selection"):
                    extract_file(self.source, ExtractOptions(output_dir=self.directory / "removed", selected_products=(product,)))
                infer.assert_not_called()

    def test_raw_only_export_keeps_model_sources_available_without_generating_them(self):
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                patch("ipde.extractor.infer_learned_depth") as infer:
            report = extract_file(self.source, ExtractOptions(output_dir=self.directory / "raw-only",
                write_learned_depth=True, write_npy=False, selected_products=("raw:2",)))
        infer.assert_not_called()
        self.assertEqual(len(list((self.directory / "raw-only").iterdir())), 1)
        self.assertEqual({entry["id"] for entry in report["available_products"] if entry["id"].startswith("learned-")},
                         {"learned-depthpro", "learned-da3", "learned-da2"})

    def test_left_image_does_not_substitute_for_missing_display(self):
        self.discovery.assets = self.discovery.assets[1:]
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                patch("ipde.extractor.infer_learned_depth") as infer:
            with self.assertRaisesRegex(ExtractionError, "unavailable selection"):
                extract_file(self.source, ExtractOptions(selected_products=("learned-da3",)))
            infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
