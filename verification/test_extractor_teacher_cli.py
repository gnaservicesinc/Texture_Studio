"""Direct AI exports expose their model choices without requiring RAFT."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ipde import cli
from ipde.extractor import inspect_file


class ExtractorTeacherCLITests(unittest.TestCase):
    def test_explicit_teacher_settings_reach_backend_and_default_to_display_grids(self):
        report = {"source": {"path": "photo.heic"}}
        with patch("ipde.cli.extract_file", return_value=report) as extract, redirect_stdout(io.StringIO()) as output:
            status = cli.main(["photo.heic", "--json", "--learned-depth", "--learned-model", "depth-anything-3",
                               "--learned-model-path", "local-model", "--learned-source-dir", "local-source",
                               "--learned-device", "mps", "--learned-input-size", "1036"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), report)
        options = extract.call_args.args[1]
        self.assertEqual(options.selected_products, ("learned-da3",))
        self.assertEqual((options.learned_model, options.learned_model_path, options.learned_source_dir,
                          options.learned_device, options.learned_input_size),
                         ("depth-anything-3", Path("local-model"), Path("local-source"), "mps", 1036))

    def test_teacher_inspection_ignores_stale_raft_flags(self):
        with patch("ipde.cli.inspect_file", return_value={}) as inspect, redirect_stdout(io.StringIO()):
            status = cli.main(["photo.heic", "--json", "--inspect", "--learned-depth", "--raft-model", "missing.pth"])
        self.assertEqual(status, 0)
        self.assertTrue(inspect.call_args.kwargs["include_learned"])
        self.assertIsNone(inspect.call_args.kwargs["raft_options"])

    def test_backend_teacher_inspection_does_not_resolve_any_raft_checkpoint(self):
        discovery = object()
        with patch("ipde.extractor.inspect_raft_checkpoint", side_effect=AssertionError("stale RAFT settings were used")), \
                patch("ipde.extractor.discover_file", return_value=discovery), \
                patch("ipde.extractor._report", return_value={"fixture": True}) as report:
            self.assertEqual(inspect_file("photo.heic", include_learned=True), {"fixture": True})
        self.assertIsNone(report.call_args.kwargs["selected_model"])

    def test_explicit_product_selection_remains_authoritative(self):
        with patch("ipde.cli.extract_file", return_value={}) as extract, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["photo.heic", "--json", "--learned-depth", "--select", "learned-depthpro", "--select", "learned-da3", "--select", "learned-da2"]), 0)
        self.assertEqual(extract.call_args.args[1].selected_products, ("learned-depthpro", "learned-da3", "learned-da2"))

    def test_checked_models_keep_their_own_local_paths(self):
        settings = {"depth-anything-3": {"model_path": "/models/da3", "source_dir": "/sources/da3"},
                    "depth-anything-v2": {"model_path": "/models/da2.pth"}}
        with patch("ipde.cli.extract_file", return_value={}) as extract, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["photo.heic", "--json", "--learned-depth",
                "--learned-model-settings", json.dumps(settings)]), 0)
        self.assertEqual(extract.call_args.args[1].learned_model_settings, settings)

    def test_model_stdout_is_redirected_away_from_gui_json_response(self):
        def noisy_model(*args):
            print("DA3 constructor diagnostic")
            return {"fixture": True}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("ipde.cli.extract_file", side_effect=noisy_model), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(cli.main(["photo.heic", "--json", "--learned-depth"]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"fixture": True})
        self.assertIn("DA3 constructor diagnostic", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
