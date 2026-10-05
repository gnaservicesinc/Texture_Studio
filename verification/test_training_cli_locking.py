"""Published checkpoints remain exportable while a CLI trainer owns its run."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ipde.formats import sha256_file
from ipde.trainer_cli import main
from ipde.workspace_cleanup import cleanup_run
from test_training_progress import _fake_raft_modules


@unittest.skipIf(os.name == "nt", "Windows resource locks conservatively serialize readers")
class TrainingCLILockingTests(unittest.TestCase):
    def test_active_cli_trainer_allows_verified_export_but_blocks_cleanup_and_second_trainer(self):
        import torch
        from ipde.spatial import _model_configuration
        from ipde.training import export_raft_checkpoint
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            workspace = root / "workspace"
            run = workspace / "runs" / "active"
            dataset = workspace / "datasets" / "sample"
            run.mkdir(parents=True)
            dataset.mkdir(parents=True)
            checkpoint = run / "ready-step.pth"
            weights = fixture(None).state_dict()
            training = {"schema": "ipde-raft-training-report-v2", "status": "experimental", "warnings": []}
            torch.save({"state_dict": weights, "ipde_configuration": vars(_model_configuration("sceneflow.pth")),
                "ipde_training": training}, checkpoint)
            checkpoint.with_suffix(".pth.json").write_text(json.dumps({**training,
                "checkpoint_sha256": sha256_file(checkpoint)}))
            (run / "training.log").write_text("retain until trainer exits")
            custom_source = root / "custom-raft-source"
            (custom_source / "core").mkdir(parents=True)
            (custom_source / "core" / "raft_stereo.py").write_text("# fixture source marker\n")
            ready, release = root / "ready", root / "release"
            program = '''
from pathlib import Path
import sys
import time
from ipde import trainer_cli, training
ready, release, dataset, checkpoint = map(Path, sys.argv[1:])
def active_training(*args, **kwargs):
    ready.write_text("locks acquired")
    while not release.exists():
        time.sleep(.02)
    return {"fixture": "finished"}
training.train_dataset = active_training
raise SystemExit(trainer_cli.main(["--json", "train", str(dataset), "--student", "raft", "--checkpoint", str(checkpoint)]))
'''
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            process = subprocess.Popen([sys.executable, "-c", program, str(ready), str(release), str(dataset),
                str(run / "final.pth")], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists(), "Separate CLI trainer failed to acquire its run locks")
                destination = root / "exported"
                output, logs = io.StringIO(), io.StringIO()
                with patch.dict(sys.modules, modules), patch("ipde.training.export_raft_checkpoint", wraps=export_raft_checkpoint) as exporting, \
                        redirect_stdout(output), redirect_stderr(logs):
                    status = main(["--json", "export", str(checkpoint), "--output", str(destination),
                        "--raft-root", str(custom_source)])
                self.assertEqual(status, 0, output.getvalue() + logs.getvalue())
                self.assertEqual(exporting.call_args.kwargs["raft_root"], custom_source)
                report = json.loads(output.getvalue())
                self.assertIn("Strict architecture load", report["weights_verified"])
                saved = torch.load(destination / "raft-model.pth", weights_only=True)
                self.assertTrue(all(torch.equal(weights[key], saved["state_dict"][key]) for key in weights))
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    cleanup_run(checkpoint, workspace, confirm=True)
                with patch("ipde.training.train_dataset", side_effect=AssertionError("second trainer acquired ownership")), \
                        redirect_stdout(io.StringIO()) as rejected:
                    status = main(["--json", "train", str(dataset), "--student", "raft", "--checkpoint", str(run / "second.pth")])
                self.assertEqual(status, 1)
                self.assertIn("in use", json.loads(rejected.getvalue())["error"])
                self.assertEqual((run / "training.log").read_text(), "retain until trainer exits")
            finally:
                release.touch()
                try:
                    stdout, stderr = process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                self.assertEqual(process.returncode, 0, stdout + stderr)
            cleaned = cleanup_run(checkpoint, workspace, confirm=True)
            self.assertEqual(cleaned["removed_files"], 1)
            self.assertFalse((run / "training.log").exists())
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((destination / "raft-model.pth").is_file())


if __name__ == "__main__":
    unittest.main()
