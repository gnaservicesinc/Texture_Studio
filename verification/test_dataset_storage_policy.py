"""Training datasets store exact useful images without archival-only payloads."""
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import read_array
from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.display_training import _DisplayPool, display_target_eligibility
from ipde.extractor import Asset
from ipde.learned_depth import LearnedDepthConfig
from ipde.training import TrainingOptions
from verification.test_display_teacher_source import _capture, _prediction


def _with_unused_assets(source):
    capture = _capture(source)
    # Unneeded assets precede and separate required views. The source ordinal
    # must never be confused with the index of a retained training asset.
    capture.assets.insert(0, Asset("auxiliary", 0, 0, np.full((17, 23), .5, np.float32),
                                    "float32", 32, "portrait_matte"))
    capture.assets.insert(2, Asset("color_view", 3, 0, np.full((19, 29, 3), 170, np.uint16),
                                    "RGB;16", 10, "unused_color"))
    return capture


class DatasetStoragePolicyTests(unittest.TestCase):
    def _build(self, root, options=None, factory=None):
        sources = [root / f"capture-{index}.heic" for index in range(3)]
        for source in sources:
            source.write_bytes(b"original container unchanged")
        destination = root / "dataset"
        with patch("ipde.dataset.discover_file", side_effect=_with_unused_assets), \
                patch("ipde.learned_depth.LearnedDepthPredictor", side_effect=factory or (
                    lambda config: lambda rgb, **kwargs: _prediction(rgb))), redirect_stderr(io.StringIO()):
            manifest = build_dataset(sources, destination, options or DatasetOptions(
                teacher=LearnedDepthConfig(), workers=1, skip_bad_photos=False))
        return destination, manifest

    def test_default_has_one_lossless_depth_plane_and_only_required_raw_views(self):
        target = np.linspace(.5, 4., 9 * 12, dtype=np.float32).reshape(9, 12)
        target.view(np.uint32)[0, 0] = 0x7FC12345
        target[0, 1] = -0.
        target[0, 2] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest = self._build(root, factory=lambda config: lambda rgb, **kwargs: _prediction(rgb, depth=target))
            self.assertEqual(load_dataset(dataset), manifest)
            self.assertEqual(manifest["array_storage"], "lossless_exr_png_npz")
            self.assertFalse(manifest["storage_policy"]["retain_intermediates"])
            self.assertFalse(manifest["storage_policy"]["preserve_auxiliary_assets"])
            self.assertTrue(display_target_eligibility(manifest)["trainable"])
            for sample in manifest["samples"]:
                original = _with_unused_assets(Path(sample["source_path"]))
                expected = {asset.semantic_name: asset.array for asset in original.assets}
                self.assertEqual({asset["semantic_name"] for asset in sample["raw_assets"]},
                                 {"spatial_left", "spatial_right", "display"})
                for key, role in (("rgb", "spatial_left"), ("right_rgb", "spatial_right"), ("display_rgb", "display")):
                    record = sample[key]
                    self.assertEqual(Path(record["path"]).suffix, ".png")
                    self.assertEqual(read_array(dataset / record["path"]).tobytes(), expected[role].tobytes())
                for key in ("teacher", "display_teacher"):
                    label = sample[key]
                    self.assertNotIn("native_target", label)
                    self.assertNotIn("valid_mask", label)
                    self.assertNotIn("confidence", label)
                    self.assertIs(label["native_prediction_retained"], False)
                    self.assertEqual(label["validity_policy"], "positive_finite")
                    self.assertEqual(label["metadata"]["native_prediction_shape"], [2, 3])
                    self.assertEqual(Path(label["target"]["path"]).suffix, ".exr")
                    self.assertEqual(read_array(dataset / label["target"]["path"]).tobytes(), target.tobytes())
                self.assertEqual(sample["teacher"]["target"], sample["display_teacher"]["target"])
                self.assertEqual(Path(sample["source_path"]).read_bytes(), b"original container unchanged")
            # Identical teacher planes are shared; no native/confidence/mask
            # sidecars or auxiliary payloads are written in this default.
            self.assertEqual(len(list(dataset.rglob("*.exr"))), 1)
            self.assertFalse(list(dataset.rglob("*.npz")))
            pool = _DisplayPool(dataset, manifest["samples"], TrainingOptions(prefetch_samples=0))
            try:
                loaded = pool.get(0)
                np.testing.assert_array_equal(loaded["valid"], np.isfinite(target) & (target > 0))
                self.assertEqual(loaded["target"].tobytes(), target.tobytes())
            finally:
                pool.close()

    def test_archival_options_retain_every_array_and_intermediate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._build(Path(directory), DatasetOptions(
                teacher=LearnedDepthConfig(), workers=1, array_format="numpy",
                preserve_auxiliary_assets=True, retain_intermediates=True))
            self.assertEqual(load_dataset(dataset), manifest)
            for sample in manifest["samples"]:
                original = _with_unused_assets(Path(sample["source_path"]))
                self.assertEqual(len(sample["raw_assets"]), len(original.assets))
                for asset, record in zip(original.assets, sample["raw_assets"]):
                    stored = read_array(dataset / record["storage"]["path"])
                    self.assertEqual((stored.dtype, stored.shape, stored.tobytes()),
                                     (asset.array.dtype, asset.array.shape, asset.array.tobytes()))
                for key in ("teacher", "display_teacher"):
                    for field in ("target", "native_target", "valid_mask", "confidence"):
                        self.assertEqual(Path(sample[key][field]["path"]).suffix, ".npz")
            self.assertTrue(list(dataset.rglob("*.npz")))
            self.assertFalse(list(dataset.rglob("*.exr")))

    def test_uncompressed_override_preserves_legacy_npy_and_no_redundant_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._build(Path(directory), DatasetOptions(
                teacher=LearnedDepthConfig(), workers=1, compress_arrays=False))
            self.assertEqual(load_dataset(dataset), manifest)
            self.assertEqual(manifest["array_storage"], "npy")
            self.assertTrue(list(dataset.rglob("*.npy")))
            self.assertFalse(list(dataset.rglob("*.png")))
            self.assertNotIn("valid_mask", manifest["samples"][0]["teacher"])

    def test_anchored_label_keeps_restrictive_mask_and_default_omits_native_anchor(self):
        def factory(config):
            def predict(rgb, **kwargs):
                relative = np.linspace(.2, 1.8, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
                return _prediction(rgb, units="meters" if config.model == "depthpro" else "relative_inverse_depth",
                                   depth=1. / (relative * .25 + .1) if config.model == "depthpro" else relative)
            return predict
        with tempfile.TemporaryDirectory() as directory:
            # The scale estimator needs enough pixels to fit and validate.
            def capture(source):
                result = _with_unused_assets(source)
                display = _capture(source, display_shape=(96, 128)).assets[-1]
                result.assets[-1] = display
                return result
            root = Path(directory)
            sources = [root / f"capture-{index}.heic" for index in range(3)]
            for source in sources:
                source.write_bytes(b"capture")
            with patch("ipde.dataset.discover_file", side_effect=capture), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", side_effect=factory), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=LearnedDepthConfig(model="depth-anything-v2"), metric_anchor=LearnedDepthConfig(model="depthpro"), workers=1))
            self.assertEqual(load_dataset(root / "dataset"), manifest)
            for sample in manifest["samples"]:
                self.assertIn("valid_mask", sample["anchored_display_teacher"])
                self.assertNotIn("native_target", sample["display_metric_anchor"])
                self.assertEqual(sample["display_metric_anchor"]["metadata"]["native_prediction_shape"], [2, 3])

    def test_invalid_storage_format_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as directory, patch("ipde.dataset.discover_file") as discovery:
            root = Path(directory)
            source = root / "capture.heic"
            source.write_bytes(b"capture")
            with self.assertRaisesRegex(DatasetError, "array_format"):
                build_dataset([source], root / "dataset", replace(DatasetOptions(teacher=LearnedDepthConfig()), array_format="jpeg"))
            discovery.assert_not_called()

    def test_omission_declarations_and_present_records_cannot_hide_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, manifest = self._build(Path(directory))
            path = dataset / "dataset.json"
            for field in ("native_prediction_retained", "validity_policy"):
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(manifest))
                    changed["samples"][0]["teacher"].pop(field)
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(DatasetError, "incomplete scientific array record"):
                        load_dataset(dataset)
            for field in ("native_target", "valid_mask"):
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(manifest))
                    changed["samples"][0]["teacher"][field] = {"path": "incomplete.npy"}
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(DatasetError, "incomplete scientific array record"):
                        load_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
