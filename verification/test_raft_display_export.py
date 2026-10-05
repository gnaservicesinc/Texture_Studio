"""Selected display products must dispatch, preserve data and identify their grid."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import OpenEXR

from ipde.extractor import Asset, Discovery, ExtractOptions, extract_file
from ipde.formats import read_exr_exact, read_png_exact
from ipde.spatial import RaftStereoResult, derive_raft_height_and_depth


class RaftDisplayExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "photo.heic"
        self.source.write_bytes(b"fixture")
        self.left = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.right = self.left[:, ::-1].copy()
        self.display = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        self.spatial = {"left_image_index": 1, "right_image_index": 2,
                        "rectified_stereo_ready": True,
                        "focal_length_pixels_for_depth": 10.0, "baseline_meters": .1,
                        "principal_point_delta_x_pixels": 3.0,
                        "left_camera": {"width": 3, "height": 2}}
        self.discovery = Discovery(self.source, 7, "hash", "image/heic", 0, [], [
            Asset("display_view", 0, 0, self.display, "RGB", 8, "display"),
            Asset("spatial_view", 1, 0, self.left, "RGB", 8, "spatial_left"),
            Asset("spatial_view", 2, 0, self.right, "RGB", 8, "spatial_right"),
        ], spatial_photo=self.spatial)
        self.flow = np.array([[2, 1, 0], [-1, -2, -3]], dtype=np.float32)
        self.height, self.depth = derive_raft_height_and_depth(self.flow, self.spatial)
        self.support = np.array([[True, False, True], [False, True, True]])
        self.raft = RaftStereoResult(self.flow, self.height, self.depth,
                                    {"engine": "RAFT-Stereo", "reference_image": "spatial_left",
                                     "depth_and_disparity_filtered_by_support": False}, self.support)
        self.registered = np.array([[1, 1, np.nan, np.nan, .25, .25],
                                    [1, 1, np.nan, np.nan, .25, .25],
                                    [np.nan, np.nan, .2, .2, .125, .125],
                                    [np.nan, np.nan, .2, .2, .125, .125]], dtype=np.float32)

    def test_display_depth_or_preview_alone_writes_only_selected_product(self):
        originals = [array.copy() for array in (self.left, self.right, self.display,
                                               self.flow, self.height, self.depth, self.support)]
        registration = {"accepted": True, "reference_role": "left"}
        for product in ("raft-display-depth", "raft-display-preview"):
            with self.subTest(product=product):
                output = self.directory / product
                with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                     patch("ipde.extractor.run_raft_stereo", return_value=self.raft) as raft, \
                     patch("ipde.extractor.run_stereo_matching") as classical, \
                     patch("ipde.registration.estimate_display_registration", return_value=registration), \
                     patch("ipde.registration.project_left_depth_to_display", return_value=self.registered.copy()) as project:
                    report = extract_file(self.source, ExtractOptions(output_dir=output,
                        selected_products=(product,), write_npy=False))
                raft.assert_called_once()
                classical.assert_not_called()
                project.assert_called_once()
                # Geometric disparity already includes the principal-point delta;
                # reprojection must not apply it again or alter the source arrays.
                expected_left = np.where(self.support, self.depth, np.float32(np.nan))
                np.testing.assert_array_equal(project.call_args.args[0], expected_left)
                suffix = "depth_meters.exr" if product.endswith("-depth") else "depth_preview.png"
                path = output / ("photo_spatial_raft_stereo_display_" + suffix)
                self.assertEqual(list(output.iterdir()), [path])
                self.assertEqual(report["selected_products"], [product])
                self.assertIsNone(report["manifest_path"])
                if product.endswith("-depth"):
                    np.testing.assert_array_equal(read_exr_exact(path, self.registered.shape), self.registered)
                    with OpenEXR.File(str(path)) as image:
                        header = image.header()
                        self.assertEqual(header["ipdeUnits"], "meters")
                        self.assertIn("RAFT", header["ipdeSemantic"])
                        self.assertIn("display grid", header["ipdeSemantic"])
                        self.assertIn("unsupported = NaN", header["ipdeTransform"])
                        metadata = json.loads(header["ipdeDerivation"])
                else:
                    preview = read_png_exact(path)
                    self.assertEqual(preview.shape, (*self.registered.shape, 2))
                    np.testing.assert_array_equal(preview[..., 1], np.where(np.isfinite(self.registered), 65535, 0))
                    self.assertEqual(int(preview[0, 0, 0]), 0)
                    self.assertEqual(int(preview[-1, -1, 0]), 65535)
                    metadata = report["assets"][1]["outputs"][0]["derivation"]
                    self.assertIn("supported registered depth", metadata["alpha_semantics"])
                self.assertEqual(metadata["reference_image"], "display")
                self.assertTrue(metadata["resampled_derivative"])
                self.assertTrue(metadata["depth_filtered_by_support"])
                self.assertTrue(metadata["depth_and_disparity_filtered_by_support"])
                self.assertEqual(metadata["shape"][:2], [4, 6])
        for array, original in zip((self.left, self.right, self.display,
                                    self.flow, self.height, self.depth, self.support), originals):
            np.testing.assert_array_equal(array, original)


if __name__ == "__main__":
    unittest.main()
