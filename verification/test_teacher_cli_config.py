"""GUI/CLI teacher and metric-anchor settings reach the same model contract."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipde.trainer_cli import main


class TeacherCLIConfigTests(unittest.TestCase):
    def call_dataset(self, root, teachers, extra=()):
        config = root / "teachers.json"
        config.write_text(json.dumps(teachers))
        output = io.StringIO()
        with patch("ipde.dataset.build_dataset", return_value={}) as build, \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            status = main(["--json", "dataset", str(root / "source.heic"), "--output-dir", str(root / "dataset"),
                           "--teachers-json", str(config), *extra])
        self.assertEqual(status, 0, output.getvalue())
        return build.call_args.args[2]

    def test_default_anchor_reuses_complete_selected_depthpro_json_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pro = {"id": "pro", "model": "depthpro", "model_path": "/custom/pro.pt",
                   "source_dir": "/custom/pro-source", "device": "cpu", "input_size": 1050}
            da3 = {"id": "da3", "model": "depth-anything-3", "model_path": "/custom/da3",
                   "source_dir": "/custom/da3-source", "device": "mps", "input_size": 0}
            for teachers in ([pro, da3], [da3, pro]):
                with self.subTest(order=[teacher["id"] for teacher in teachers]):
                    options = self.call_dataset(root, teachers, ["--metric-anchor", "depthpro"])
                    selected = next(config for config in (options.teacher, *options.additional_teachers) if config.model == "depthpro")
                    self.assertIs(options.metric_anchor, selected)
                    self.assertEqual(options.metric_anchor.device, "cpu")
                    self.assertEqual(options.metric_anchor.input_size, 1050)
                    self.assertEqual(options.metric_anchor.model_path, Path("/custom/pro.pt"))
                    self.assertEqual(options.metric_anchor.source_dir, Path("/custom/pro-source"))
            options = self.call_dataset(root, [pro, da3])
            self.assertIsNone(options.metric_anchor)

    def test_explicit_anchor_override_remains_distinct_and_preserves_chosen_inference_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pro = {"id": "pro", "model": "depthpro", "model_path": "/selected/pro.pt",
                   "source_dir": "/selected/source", "device": "mps", "input_size": 0}
            options = self.call_dataset(root, [pro], ["--metric-anchor", "depthpro",
                "--anchor-model-path", "/anchor/pro.pt", "--anchor-source-dir", "/anchor/source",
                "--device", "cpu", "--input-size", "1024"])
            self.assertNotEqual(options.metric_anchor, options.teacher)
            self.assertEqual(options.metric_anchor.model_path, Path("/anchor/pro.pt"))
            self.assertEqual(options.metric_anchor.source_dir, Path("/anchor/source"))
            self.assertEqual((options.metric_anchor.device, options.metric_anchor.input_size), ("cpu", 1024))
            self.assertEqual((options.teacher.device, options.teacher.input_size), ("mps", 0))

    def test_relative_teacher_anchor_uses_its_explicit_depthpro_paths_and_device_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = {"id": "da3", "model": "depth-anything-3", "model_path": "/models/da3",
                       "source_dir": "/sources/da3", "device": "mps", "input_size": 0}
            options = self.call_dataset(root, [teacher], ["--metric-anchor", "depthpro",
                "--anchor-model-path", "/models/pro.pt", "--anchor-source-dir", "/sources/pro",
                "--device", "cpu", "--input-size", "518"])
            self.assertEqual(options.metric_anchor.model, "depthpro")
            self.assertEqual(options.metric_anchor.model_path, Path("/models/pro.pt"))
            self.assertEqual(options.metric_anchor.source_dir, Path("/sources/pro"))
            self.assertEqual((options.metric_anchor.device, options.metric_anchor.input_size), ("cpu", 518))
            self.assertEqual(options.teacher.model_path, Path("/models/da3"))

    def test_per_photo_command_reuses_depthpro_config_and_keeps_relative_anchor_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("depthpro", "depth-anything-3"):
                args = ["--json", "generate-teacher", str(root / "dataset"), "--sample-id", "photo",
                        "--model", model, "--model-path", "/chosen/model", "--source-dir", "/chosen/source",
                        "--device", "cpu", "--input-size", "0", "--metric-anchor", "depthpro"]
                if model != "depthpro":
                    args += ["--anchor-model-path", "/chosen/pro.pt", "--anchor-source-dir", "/chosen/pro-source"]
                output = io.StringIO()
                with self.subTest(model=model), patch("ipde.dataset_teachers.generate_teacher", return_value={}) as generate, \
                        redirect_stdout(output), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(args), 0, output.getvalue())
                    teacher = generate.call_args.args[2]
                    anchor = generate.call_args.kwargs["metric_anchor"]
                    self.assertEqual((anchor.device, anchor.input_size), ("cpu", 0))
                    if model == "depthpro":
                        self.assertIs(anchor, teacher)
                    else:
                        self.assertEqual(anchor.model_path, Path("/chosen/pro.pt"))
                        self.assertEqual(anchor.source_dir, Path("/chosen/pro-source"))
                        self.assertEqual(teacher.model_path, Path("/chosen/model"))


if __name__ == "__main__":
    unittest.main()
