"""Production display exports use native stereo and the student's declared units."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import OpenEXR

from ipde.extractor import ExtractOptions, ExtractionError, extract_file
from ipde.formats import read_exr_exact, read_png_exact
import test_raft_display_export as fixture


class StudentDisplayExportTests(unittest.TestCase):
    setUp = fixture.RaftDisplayExportTests.setUp
    def test_native_stereo_direct_display_depth_and_height_units(self):
        checkpoint = self.directory / "display-model.pth"
        checkpoint.write_bytes(b"student fixture")
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        originals = [a.copy() for a in (self.left, self.right, self.display, prediction)]
        for units in ("meters", "relative_depth", "relative_inverse_depth"):
            for product in ("student-display-depth", "student-display-displacement", "student-display-preview"):
                with self.subTest(units=units, product=product):
                    output = self.directory / (units + product)
                    with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                         patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                             "units": units, "reference_image": "display", "teacher_transport": "none", "output_resampling": "none"})) as student, \
                         patch("ipde.extractor.run_raft_stereo") as stock, \
                         patch("ipde.registration.estimate_display_registration") as register:
                        report = extract_file(self.source, ExtractOptions(output_dir=output,
                            selected_products=(product,), raft_model=checkpoint, write_npy=False))
                    stock.assert_not_called(); register.assert_not_called()
                    self.assertIs(student.call_args.args[0], self.left)
                    self.assertIs(student.call_args.args[1], self.right)
                    self.assertEqual(student.call_args.args[2], (4, 6))
                    files = list(output.iterdir())
                    self.assertEqual(len(files), 1)
                    if product.endswith("-depth"):
                        actual = read_exr_exact(files[0], prediction.shape)
                        np.testing.assert_array_equal(actual.view(np.uint32), prediction.view(np.uint32))
                    else:
                        expected = ((prediction-1) if units == "relative_inverse_depth" else (24-prediction)) / np.float32(23)
                        if product.endswith("-preview"):
                            np.testing.assert_array_equal(read_png_exact(files[0]), np.rint(expected.astype(np.float64)*65535).astype(np.uint16))
                        else:
                            np.testing.assert_array_equal(read_exr_exact(files[0], prediction.shape), expected)
                    metadata = report["assets"][0]["outputs"][0]["derivation"]
                    self.assertEqual(metadata["reference_image"], "display")
                    self.assertFalse(metadata["display_rgb_used_for_inference"])
                    self.assertEqual(metadata["teacher_transport"], "none")
                    if product.endswith("-depth"):
                        with OpenEXR.File(str(files[0])) as image:
                            self.assertEqual(set(image.channels()), {"Y"})
                            self.assertEqual(image.header()["ipdeUnits"], units)
                            self.assertFalse(json.loads(image.header()["ipdeDerivation"])["normalization"])
        for value, original in zip((self.left, self.right, self.display, prediction), originals):
            np.testing.assert_array_equal(value, original)

    def test_color_preprocessing_refused_before_student_inference(self):
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.display_student.predict_display_depth") as infer:
            with self.assertRaisesRegex(ExtractionError, "Disable Color Matching"):
                extract_file(self.source, ExtractOptions(output_dir=self.directory / "color",
                    selected_products=("student-display-depth",), raft_model=Path("display-model.pth"),
                    histogram_color_matching=True))
        infer.assert_not_called()

    def test_comparison_preserves_different_camera_grids_without_a_warp(self):
        import torch
        from ipde.model_comparison import compare_models
        from ipde.formats import read_png_exact
        self.spatial["raft_stereo_ready"] = True
        baseline, candidate = self.directory / "stock.pth", self.directory / "display.pth"
        torch.save({"fixture": True}, baseline)
        torch.save({"schema": "ipde-display-depth-v1"}, candidate)
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        with patch("ipde.model_comparison.discover_file", return_value=self.discovery), \
             patch("ipde.model_comparison.run_raft_stereo", return_value=self.raft) as stock, \
             patch("ipde.display_student.predict_display_depth", return_value=(prediction, {"units": "relative_depth"})) as student, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = compare_models(self.source, self.directory / "comparison", baseline_model=baseline, candidate_model=candidate)
        stock.assert_called_once(); student.assert_called_once(); register.assert_not_called()
        self.assertEqual(report["comparison"]["mode"], "separate-camera-grids")
        self.assertNotIn("difference_preview_path", report)
        self.assertEqual([(sample["height"], sample["width"]) for sample in report["samples"]], [(2, 3), (4, 6)])
        for sample, source in zip(report["samples"], (self.left, self.display)):
            np.testing.assert_array_equal(read_png_exact(Path(sample["rgb_preview_path"])), source)


if __name__ == "__main__":
    unittest.main()
