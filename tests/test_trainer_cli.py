"""The trainer is a separate entry point with cheap workspace inventories."""
import json
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipde.trainer_cli import main, workspace_report
from ipde.cli import _parser as extractor_parser


class TrainerCliTests(unittest.TestCase):
    def test_workspace_reads_nested_runs_and_dataset_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "datasets/scene-a"
            data.mkdir(parents=True)
            (data / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1", "samples": [
                {"split": "train", "teacher": {"metadata": {"model_id": "apple/DepthPro"}}},
                {"split": "validation"}]}))
            checkpoint = root / "runs/pilot/arbitrary-name.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture")
            checkpoint.with_suffix(".pth.json").write_text(json.dumps({"best_epoch": 2, "validation": {"mean_absolute_flow_error_pixels": 3.}}))
            report = workspace_report(root)
            self.assertEqual(report["datasets"][0]["sample_count"], 2)
            self.assertEqual(report["datasets"][0]["teacher"], "apple/DepthPro")
            self.assertEqual(report["runs"][0]["path"], str(checkpoint.resolve()))
            self.assertEqual(report["runs"][0]["best_epoch"], 2)

    def test_invalid_group_document_errors_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "groups.json"
            groups.write_text('["not a mapping"]')
            with patch("ipde.dataset.build_dataset") as build, patch("builtins.print") as output:
                self.assertEqual(main(["--json", "dataset", "photo.HEIC", "--output-dir", str(root / "dataset"), "--groups", str(groups)]), 1)
            build.assert_not_called()
            self.assertIn("error", json.loads(output.call_args.args[0]))

    def test_training_options_reach_the_actual_raft_trainer(self):
        with patch("ipde.training.train_dataset", return_value={"status": "experimental"}) as train, patch("builtins.print"):
            self.assertEqual(main(["--json", "train", "/tmp/dataset", "--checkpoint", "/tmp/pilot.pth", "--patch-size", "256", "--epochs", "2", "--steps", "3", "--raft-model", "/tmp/base.pth"]), 0)
        options = train.call_args.args[2]
        self.assertEqual((options.patch_size, options.epochs, options.steps_per_epoch), (256, 2, 3))
        self.assertEqual(options.raft_model, Path("/tmp/base.pth"))

    def test_extract_program_does_not_expose_training_flags(self):
        flags = {flag for action in extractor_parser()._actions for flag in action.option_strings}
        self.assertFalse({"--train-dataset", "--dataset-output", "--learned-model"} & flags)

    def test_model_stdout_logs_do_not_corrupt_gui_json(self):
        def model_work(*args):
            print("[INFO] using SwiGLU layer as FFN")
            return {"source": "photo.HEIC"}

        output, progress = io.StringIO(), io.StringIO()
        with patch("ipde.extractor.extract_file", side_effect=model_work), redirect_stdout(output), redirect_stderr(progress):
            status = main(["--json", "teacher", "photo.HEIC", "--output-dir", "/tmp/teacher", "--model", "depth-anything-3"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), {"sources": [{"source": "photo.HEIC"}]})
        self.assertIn("using SwiGLU layer as FFN", progress.getvalue())
