"""Precision, registration and transactional regressions for new media studios."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ipde.extractor import Asset, Discovery
from ipde.formats import arrays_bit_equal, read_npy_exact, read_png_exact, write_npy
from ipde.learned_depth import LearnedDepthResult
from ipde.photo_workflow import (MediaWorkflowError, compose_mattes, export_array,
                                 export_photo, rewrite_photo, upscale_depth)
from ipde.raw_workflow import RawCapture, export_raw, validate_crop


def asset(name, values, kind="auxiliary", parent=0, bits=8, metadata=None):
    return Asset(kind, parent, 0, values, str(values.dtype), bits, name, metadata=metadata or {})


def photo(*assets):
    return Discovery(Path("test.heic"), 1, "hash", "image/heic", 0,
                     [{"source_bit_depth": 8}], list(assets))


class MediaWorkflowTests(unittest.TestCase):
    def test_matte_union_keeps_color_bits_and_partial_boundary(self):
        color = np.array([[[14, 25, 36], [245, 131, 87]]], dtype=np.uint8)
        portrait = np.array([[128, 0]], dtype=np.uint8)
        skin = np.array([[64, 255]], dtype=np.uint8)
        discovery = photo(asset("portrait_effects_matte", portrait), asset("semantic_skin_matte", skin),
                          asset("display", color, "color_view"))
        rgba, alpha, metadata = compose_mattes(discovery, [0, 1, 1])
        self.assertTrue(arrays_bit_equal(rgba[..., :3], color))
        np.testing.assert_array_equal(alpha, [[128, 255]])
        np.testing.assert_array_equal(portrait, [[128, 0]])
        self.assertEqual(metadata["matte_layers"], ["portrait_effects_matte", "semantic_skin_matte"])

    def test_nonperson_sky_and_foreign_camera_are_rejected(self):
        color = np.zeros((2, 2, 3), np.uint8)
        discovery = photo(asset("semantic_sky_matte", np.ones((2, 2), np.uint8)),
                          asset("portrait_effects_matte", np.ones((2, 2), np.uint8), parent=1),
                          asset("display", color, "color_view"))
        for index in (0, 1):
            with self.assertRaises(MediaWorkflowError): compose_mattes(discovery, [index])

    def test_guided_depth_keeps_invalid_regions_and_original_precision(self):
        depth = np.array([[1, np.nan], [2, 3]], np.float16)
        original = depth.tobytes()
        discovery = photo(asset("apple_depth", depth, "native_depth", bits=8,
                                metadata={"representation": "depth", "accuracy": "relative"}),
                          asset("display", np.zeros((8, 8, 3), np.uint8), "color_view"))
        output, metadata = upscale_depth(discovery, 0)
        self.assertEqual(output.shape, (8, 8))
        self.assertTrue(np.isnan(output[:4, 4:]).all())
        self.assertEqual(depth.tobytes(), original)
        self.assertFalse(metadata["measured_precision_added"])

    def test_foreign_depth_and_wrong_aspect_are_rejected(self):
        discovery = photo(asset("apple_depth", np.ones((2, 2), np.float16), "native_depth", parent=1),
                          asset("display", np.zeros((4, 4, 3), np.uint8), "color_view"))
        with self.assertRaises(MediaWorkflowError): upscale_depth(discovery, 0)
        discovery.assets[0].parent_image_index = 0
        discovery.assets[0].array = np.ones((2, 8), np.float16)
        with self.assertRaises(MediaWorkflowError): upscale_depth(discovery, 0)

    def test_nan_payloads_and_unsigned16_codes_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bits = np.array([[0x7e55, 0x8000, 0x3555]], dtype=np.uint16)
            value = bits.view(np.float16)
            out = export_array(root, "native", value, "npy", role="depth")
            self.assertTrue(arrays_bit_equal(value, read_npy_exact(Path(out["path"]))))
            codes = np.array([[[1, 4095, 65535, 27123]]], np.uint16)
            out = export_array(root, "cutout", codes, "png", role="rgba")
            self.assertTrue(arrays_bit_equal(codes, read_png_exact(Path(out["path"]))))

    def test_group_export_failure_removes_completed_products(self):
        discovery = photo(asset("matte", np.zeros((2, 2), np.uint8)),
                          asset("apple_depth", np.ones((2, 2), np.float16), "native_depth"),
                          asset("display", np.zeros((2, 2, 3), np.uint8), "color_view"))
        with tempfile.TemporaryDirectory() as directory, patch("ipde.photo_workflow.discover_photo", return_value=discovery):
            with self.assertRaises(Exception): export_photo("x.heic", Path(directory), "original", "png")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_native_writer_success_does_not_bypass_retained_pixel_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = np.ones((2, 2), np.float16)
            metadata = {"representation": "depth", "accuracy": "relative",
                        "description": {"Width": 2, "Height": 2, "PixelFormat": int.from_bytes(b"hdep", "big"), "BytesPerRow": 4}}
            source = photo(asset("apple_depth", depth, "native_depth", metadata=metadata),
                           asset("display", np.zeros((2, 2, 3), np.uint8), "color_view"))
            after = photo(asset("apple_depth", depth.copy(), "native_depth", metadata=metadata),
                          asset("display", np.ones((2, 2, 3), np.uint8), "color_view"))
            binary = root / "native.bin"; binary.write_bytes(depth.tobytes())
            replacement = root / "replacement.npy"; write_npy(replacement, depth)
            def native(action, source_path, output, *args):
                if action == "inventory":
                    return {"auxiliary": [{"semantic": "depth", "parent": 0, "path": str(binary), "description": metadata["description"]}]}
                output.write_bytes(b"candidate")
                return {"replacement_metadata_verified": True}
            with patch("ipde.photo_workflow.discover_photo", side_effect=[source, after]), patch("ipde.photo_workflow.run_native", side_effect=native):
                with self.assertRaisesRegex(MediaWorkflowError, "sample bits"):
                    rewrite_photo("x.heic", root / "output", replacements={0: replacement})
            self.assertEqual(list((root / "output").iterdir()), [])

    def test_invalid_crop_never_slices_negative_or_truncated_area(self):
        for crop in ((-1, 0, 2, 2), (0, 0, 0, 2), (7, 0, 2, 2), (0, 6, 2, 3)):
            with self.assertRaises(MediaWorkflowError): validate_crop(crop, (8, 8))

    def test_raw_selected_aux_does_not_export_other_planes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.dng"; source.write_bytes(b"original")
            capture = RawCapture(source, sensor=np.zeros((3, 4), np.uint16),
                                 rendered=np.zeros((3, 4, 3), np.float32), auxiliary=[
                {"semantic": "hdr_gain_map", "array": np.array([[71]], np.uint8), "data": b"G", "description": {}},
                {"semantic": "iso_gain_map", "array": np.array([[81]], np.uint8), "data": b"Q", "description": {}}])
            with patch("ipde.raw_workflow.capture_raw", return_value=capture):
                report = export_raw(source, root / "output", "auxiliary", "npy", asset_index=3)
            arrays = [p for p in report["outputs"] if p["role"] != "calibration-metadata"]
            self.assertEqual(len(arrays), 1)
            self.assertEqual(arrays[0]["role"], "iso_gain_map")
            self.assertEqual(read_npy_exact(Path(arrays[0]["path"])).item(), 81)

    def test_raw_learned_crop_retains_unclipped_linear_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.dng"; source.write_bytes(b"raw")
            rgb = np.array([[[-0.01, 1.2, 0.5], [0.8, 0.4, 0.1]]], np.float32)
            capture = RawCapture(source, rendered=rgb, native_metadata={"rendering": {"nominal_max": 1.0}})
            result = LearnedDepthResult(np.array([[3]], np.float32), np.array([[3, 4]], np.float32), {})
            def predict(model_rgb, *_args, **_kwargs):
                self.assertTrue(np.all((model_rgb >= 0) & (model_rgb <= 1)))
                return result
            with patch("ipde.raw_workflow.capture_raw", return_value=capture), patch("ipde.raw_workflow.infer_learned_depth", side_effect=predict):
                report = export_raw(source, root / "out", "learned-depth", crop=(0, 0, 1, 1))
            color = next(o for o in report["outputs"] if o["role"] == "color-region")
            self.assertTrue(arrays_bit_equal(rgb[:, :1], read_npy_exact(Path(color["path"]))))

    def test_unsupported_raw_still_captures_the_entire_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "unsupported.raw"
            original = b"raw-camera-format-not-supported" + bytes(range(256))
            source.write_bytes(original)
            with patch("ipde.raw_workflow.native_bridge", return_value=None):
                report = export_raw(source, root / "out", "original")
            output = next(item for item in report["outputs"] if item["role"] == "original-raw")
            self.assertEqual(Path(output["path"]).read_bytes(), original)
            self.assertTrue(any("original-file capture" in warning for warning in report["warnings"]))

    def test_cutout_png_and_tiff_retain_source_profile_and_rgb_bits(self):
        import base64
        from PIL import Image
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.array([[[17, 81, 203, 127]]], np.uint8)
            profile = b"original-ICC-profile-bytes"
            metadata = {"color_profile": {"type": "prof", "data": {"data": base64.b64encode(profile).decode()}}}
            png = export_array(root, "profile-png", value, "png", role="rgba", metadata=metadata)
            tiff = export_array(root, "profile-tiff", value, "tiff", role="rgba", metadata=metadata)
            with Image.open(png["path"]) as image:
                self.assertEqual(image.info["icc_profile"], profile)
            with tifffile.TiffFile(tiff["path"]) as image:
                self.assertEqual(image.pages[0].tags[34675].value, profile)
            self.assertTrue(arrays_bit_equal(read_png_exact(Path(png["path"])), value))

    def test_person_mask_uses_rendered_grid_model_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "orientation3.dng"; source.write_bytes(b"raw")
            rendered = np.zeros((2, 3, 3), np.float32); rendered[0, 0] = [1, 0, 0]
            capture = RawCapture(source, rendered=rendered, native_metadata={"rendering": {"nominal_max": 1.0}})
            def native(action, reference, output):
                self.assertNotEqual(reference, source)
                model = read_png_exact(reference)
                np.testing.assert_array_equal(model[0, 0], [255, 0, 0])
                output.write_bytes(bytes([255, 0, 0, 0, 0, 0]))
                return {"width": 3, "height": 2, "reference_width": 3, "reference_height": 2}
            with patch("ipde.raw_workflow.capture_raw", return_value=capture), patch("ipde.raw_workflow.run_native", side_effect=native):
                report = export_raw(source, root / "out", "person-mask")
            mask = next(o for o in report["outputs"] if o["role"] == "person-mask")
            self.assertEqual(read_npy_exact(Path(mask["path"]))[0, 0], 255)

if __name__ == "__main__": unittest.main()
