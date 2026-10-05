"""Production display exports use native stereo and the student's declared units."""
import json
import hashlib
import os
import zipfile
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import OpenEXR

from ipde.extractor import ExtractOptions, ExtractionError, extract_file, inspect_file
from ipde.formats import read_exr_exact, read_png_exact
from ipde.spatial import RaftStereoOptions
import test_raft_display_export as fixture


class StudentDisplayExportTests(unittest.TestCase):
    setUp = fixture.RaftDisplayExportTests.setUp

    def test_selected_display_checkpoint_routes_raft_products_by_schema(self):
        import torch

        # The checkpoint name deliberately carries no hint about its output grid.
        checkpoint = self.directory / "chosen-checkpoint.pth"
        torch.save({"schema": "ipde-display-depth-v1"}, checkpoint)
        prediction = np.nextafter(np.linspace(.125, 5.875, 24, dtype=np.float32),
                                  np.float32(np.inf)).reshape(4, 6)
        originals = [array.copy() for array in (self.left, self.right, self.display, prediction)]
        aliases = {
            "raft-depth": "student-display-depth",
            "raft-display-depth": "student-display-depth",
            "raft-displacement": "student-display-displacement",
            "raft-preview": "student-display-preview",
            "raft-display-preview": "student-display-preview",
        }
        for requested, product in aliases.items():
            with self.subTest(requested=requested):
                output = self.directory / requested
                with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                     patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                         "units": "relative_inverse_depth", "reference_image": "display",
                         "teacher_transport": "none", "output_resampling": "none"})) as student, \
                     patch("ipde.extractor.run_raft_stereo") as stock, \
                     patch("ipde.extractor.run_stereo_matching") as classical, \
                     patch("ipde.registration.estimate_display_registration") as register, \
                     patch("ipde.registration.project_left_depth_to_display") as project:
                    report = extract_file(self.source, ExtractOptions(output_dir=output,
                        selected_products=(requested,), raft_model=checkpoint, write_npy=False))
                student.assert_called_once()
                self.assertIs(student.call_args.args[0], self.left)
                self.assertIs(student.call_args.args[1], self.right)
                self.assertEqual(student.call_args.args[2], self.display.shape[:2])
                self.assertEqual(student.call_args.args[3], checkpoint)
                stock.assert_not_called(); classical.assert_not_called()
                register.assert_not_called(); project.assert_not_called()
                self.assertEqual(report["selected_products"], [product])
                self.assertEqual(report["requested_products"], [requested])
                self.assertEqual(report["product_remapping"], {requested: product})
                self.assertIsNone(report["manifest_path"])
                files = list(output.iterdir())
                self.assertEqual(len(files), 1)
                metadata = report["assets"][0]["outputs"][0]["derivation"]
                self.assertEqual(metadata["reference_image"], "display")
                self.assertFalse(metadata["display_rgb_used_for_inference"])
                self.assertEqual(metadata["teacher_transport"], "none")
                if product.endswith("-depth"):
                    actual = read_exr_exact(files[0], prediction.shape)
                    np.testing.assert_array_equal(actual.view(np.uint32), prediction.view(np.uint32))
                    with OpenEXR.File(str(files[0])) as image:
                        self.assertEqual(image.header()["ipdeUnits"], "relative_inverse_depth")
                        self.assertFalse(json.loads(image.header()["ipdeDerivation"])["normalization"])
                else:
                    expected = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                    if product.endswith("-preview"):
                        np.testing.assert_array_equal(read_png_exact(files[0]),
                            np.rint(expected.astype(np.float64) * 65535).astype(np.uint16))
                    else:
                        np.testing.assert_array_equal(read_exr_exact(files[0], prediction.shape), expected)
        for array, original in zip((self.left, self.right, self.display, prediction), originals):
            np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))

    def test_selected_display_checkpoint_deduplicates_aliases_and_student_products(self):
        import torch

        checkpoint = self.directory / "another-selected-model.pth"
        torch.save({"schema": "ipde-display-depth-v1"}, checkpoint)
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        output = self.directory / "deduplicated"
        selected = ("raft-depth", "raft-display-depth", "student-display-depth",
                    "raft-displacement", "student-display-displacement",
                    "raft-preview", "raft-display-preview", "student-display-preview")
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                 "units": "relative_depth", "reference_image": "display"})) as student, \
             patch("ipde.extractor.run_raft_stereo") as stock, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = extract_file(self.source, ExtractOptions(output_dir=output,
                selected_products=selected, raft_model=checkpoint, write_npy=False))
        student.assert_called_once()
        self.assertEqual(student.call_args.args[3], checkpoint)
        stock.assert_not_called(); register.assert_not_called()
        self.assertEqual(report["selected_products"], ["student-display-depth",
            "student-display-displacement", "student-display-preview"])
        files = list(output.iterdir())
        self.assertEqual(len(files), 3)
        self.assertEqual(len(report["assets"][0]["outputs"]), 3)
        depth_file = next(path for path in files if path.name.endswith("_student_display_depth.exr"))
        np.testing.assert_array_equal(read_exr_exact(depth_file, prediction.shape).view(np.uint32),
                                      prediction.view(np.uint32))

    def test_inspection_recognizes_renamed_display_checkpoint_and_offers_its_products(self):
        import torch

        checkpoint = self.directory / "renamed-model.pth"
        torch.save({"schema": "ipde-display-depth-v1",
                    "architecture": {"units": "relative_inverse_depth"}}, checkpoint)
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("torch.load", wraps=torch.load) as load, \
             patch("ipde.extractor.run_raft_stereo") as stock, \
             patch("ipde.display_student.predict_display_depth") as student:
            report = inspect_file(self.source, raft_options=RaftStereoOptions(model=checkpoint))
        stock.assert_not_called(); student.assert_not_called()
        self.assertEqual(report["selected_model"]["kind"], "display_student")
        self.assertEqual(report["selected_model"]["schema"], "ipde-display-depth-v1")
        self.assertEqual(Path(report["selected_model"]["path"]).resolve(), checkpoint.resolve())
        self.assertEqual(report["selected_model"]["units"], "relative_inverse_depth")
        self.assertIsNone(report["selected_model"]["member"])
        self.assertTrue(load.called)
        for call in load.call_args_list:
            self.assertTrue(call.kwargs["weights_only"])
            self.assertEqual(call.kwargs["map_location"], "cpu")
        products = {product["id"]: product for product in report["available_products"]}
        self.assertEqual(set(products), {"raw:0", "raw:1", "raw:2",
            "student-display-depth", "student-display-displacement", "student-display-preview",
            "stereo-depth", "stereo-displacement", "stereo-preview", "stereo-height",
            "stereo-support", "stereo-supported-depth"})
        for product in ("student-display-depth", "student-display-displacement", "student-display-preview"):
            self.assertEqual((products[product]["height"], products[product]["width"]), self.display.shape[:2])

    def test_cli_inspection_forwards_selected_checkpoint(self):
        from ipde.cli import main

        checkpoint = self.directory / "selected-for-inspection.pth"
        report = {"fixture": "selected model inspection"}
        stdout = StringIO()
        with patch("ipde.cli.inspect_file", return_value=report) as inspect, redirect_stdout(stdout):
            result = main([str(self.source), "--inspect", "--json", "--raft-model", str(checkpoint)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertEqual(inspect.call_args.args[0], self.source)
        self.assertEqual(inspect.call_args.kwargs["raft_options"].model, checkpoint)

    def test_environment_selected_display_checkpoint_is_used(self):
        import torch

        checkpoint = self.directory / "environment-choice.pth"
        torch.save({"schema": "ipde-display-depth-v1"}, checkpoint)
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        with patch.dict(os.environ, {"IPDE_RAFT_MODEL": str(checkpoint)}), \
             patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                 "units": "relative_depth", "reference_image": "display"})) as student, \
             patch("ipde.extractor.run_raft_stereo") as stock, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = extract_file(self.source, ExtractOptions(output_dir=self.directory / "environment",
                selected_products=("raft-depth",), write_npy=False))
        student.assert_called_once()
        self.assertEqual(student.call_args.args[3], checkpoint)
        stock.assert_not_called(); register.assert_not_called()
        self.assertEqual(report["selected_products"], ["student-display-depth"])

    def test_stock_checkpoint_named_display_model_keeps_native_stereo_output(self):
        import torch

        checkpoint = self.directory / "display-model.pth"
        torch.save({"fixture": True}, checkpoint)
        output = self.directory / "stock-choice"
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.extractor.run_raft_stereo", return_value=self.raft) as stock, \
             patch("ipde.display_student.predict_display_depth") as student, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = extract_file(self.source, ExtractOptions(output_dir=output,
                selected_products=("raft-depth",), raft_model=checkpoint, write_npy=False))
        stock.assert_called_once(); student.assert_not_called(); register.assert_not_called()
        self.assertEqual(stock.call_args.args[3].model, checkpoint)
        self.assertEqual(report["selected_products"], ["raft-depth"])
        self.assertNotIn("requested_products", report)
        files = list(output.iterdir())
        self.assertEqual(len(files), 1)
        np.testing.assert_array_equal(read_exr_exact(files[0], self.depth.shape).view(np.uint32),
                                      self.depth.view(np.uint32))

    def test_legacy_raft_flags_use_selected_display_checkpoint(self):
        import torch

        checkpoint = self.directory / "legacy-choice.pth"
        torch.save({"schema": "ipde-display-depth-v1"}, checkpoint)
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                 "units": "relative_depth", "reference_image": "display"})) as student, \
             patch("ipde.extractor.run_raft_stereo") as stock, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = extract_file(self.source, ExtractOptions(output_dir=self.directory / "legacy",
                raft_model=checkpoint, write_raft_stereo=True, write_raft_diagnostics=True,
                write_displacement_maps=True, write_npy=False))
        student.assert_called_once()
        self.assertEqual(student.call_args.args[3], checkpoint)
        stock.assert_not_called(); register.assert_not_called()
        self.assertIsNone(report["selected_products"])
        products = [output for asset in report["assets"] for output in asset["outputs"]
                    if output["role"].startswith("derived_student_")]
        self.assertEqual({output["role"] for output in products},
            {"derived_student_display_depth", "derived_student_display_displacement"})
        self.assertEqual(len(products), 2)
        depth_file = Path(next(output["path"] for output in products
                              if output["role"] == "derived_student_display_depth"))
        np.testing.assert_array_equal(read_exr_exact(depth_file, prediction.shape).view(np.uint32),
                                      prediction.view(np.uint32))

    def test_selected_zip_member_routes_display_alias_and_retains_member_identity(self):
        import torch

        archive = self.directory / "chosen-models.zip"
        member = "nested/models/selected.pth"
        payload = BytesIO()
        torch.save({"schema": "ipde-display-depth-v1"}, payload)
        member_bytes = payload.getvalue()
        stock_payload = BytesIO()
        torch.save({"fixture": True}, stock_payload)
        with zipfile.ZipFile(archive, "w") as models:
            models.writestr(member, member_bytes)
            models.writestr("raftstereo-middlebury.pth", stock_payload.getvalue())
        original = archive.read_bytes()
        prediction = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
        with patch("ipde.extractor.discover_file", return_value=self.discovery), \
             patch("ipde.display_student.predict_display_depth", return_value=(prediction, {
                 "units": "relative_depth", "reference_image": "display"})) as student, \
             patch("ipde.extractor.run_raft_stereo") as stock, \
             patch("ipde.registration.estimate_display_registration") as register:
            report = extract_file(self.source, ExtractOptions(output_dir=self.directory / "zip",
                selected_products=("raft-depth",), raft_model=archive,
                raft_model_member=member, write_npy=False))
        student.assert_called_once()
        self.assertEqual(student.call_args.args[3], archive)
        self.assertEqual(student.call_args.kwargs["checkpoint_member"], member)
        stock.assert_not_called(); register.assert_not_called()
        self.assertEqual(report["selected_products"], ["student-display-depth"])
        self.assertEqual(report["selected_model"]["member"], member)
        output = report["assets"][0]["outputs"][0]
        np.testing.assert_array_equal(read_exr_exact(Path(output["path"]), prediction.shape).view(np.uint32),
                                      prediction.view(np.uint32))
        self.assertEqual(output["derivation"]["checkpoint_sha256"], hashlib.sha256(member_bytes).hexdigest())
        self.assertEqual(archive.read_bytes(), original)

    def test_display_alias_loads_and_runs_real_student_decoder(self):
        import torch
        from ipde import display_student as student
        from test_display_student import FixturePadder, FixtureRAFT

        root = self.directory / "fixture-raft"
        (root / "core").mkdir(parents=True)
        (root / "core/raft_stereo.py").write_text("# source-tree identity fixture\n")
        base = root / "base.pth"
        torch.save(FixtureRAFT(None).state_dict(), base)
        checkpoint = self.directory / "chosen-real-decoder.pth"
        output = self.directory / "real-decoder"
        originals = [array.copy() for array in (self.left, self.right, self.display)]
        previous_threads = torch.get_num_threads()
        previous_calls = list(FixtureRAFT.calls)
        torch.set_num_threads(2)
        try:
            with torch.random.fork_rng(devices=[]), \
                 patch.object(student, "_upstream", return_value=(FixtureRAFT, FixturePadder)):
                model, architecture, _ = student.create_student(raft_root=root, raft_model=base,
                    device="cpu", quality=0, iterations=2, seed=123, units="relative_inverse_depth")
                torch.save({"schema": student.SCHEMA, "architecture": architecture,
                    "ipde_configuration": architecture, "state_dict": model.state_dict()}, checkpoint)
                expected, _ = student.predict_display_depth(self.left, self.right, self.display.shape[:2],
                    checkpoint, raft_root=root, device="cpu")
                FixtureRAFT.calls.clear()
                with patch("ipde.extractor.discover_file", return_value=self.discovery), \
                     patch("ipde.extractor.run_raft_stereo") as stock, \
                     patch("ipde.registration.estimate_display_registration") as register:
                    report = extract_file(self.source, ExtractOptions(output_dir=output,
                        selected_products=("raft-depth",), raft_model=checkpoint,
                        raft_root=root, raft_device="cpu", write_npy=False))
                stock.assert_not_called(); register.assert_not_called()
                self.assertEqual(len(FixtureRAFT.calls), 1)
        finally:
            torch.set_num_threads(previous_threads)
            FixtureRAFT.calls[:] = previous_calls
        self.assertEqual(report["selected_products"], ["student-display-depth"])
        files = list(output.iterdir())
        self.assertEqual(len(files), 1)
        np.testing.assert_array_equal(read_exr_exact(files[0], expected.shape).view(np.uint32),
                                      expected.view(np.uint32))
        self.assertEqual(report["assets"][0]["outputs"][0]["derivation"]["units"], "relative_inverse_depth")
        for array, original in zip((self.left, self.right, self.display), originals):
            np.testing.assert_array_equal(array, original)

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
