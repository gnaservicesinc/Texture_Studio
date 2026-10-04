"""Verify project dataset links are references, including moved-source reports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ipde.trainer_cli import main, workspace_report


class LinkedWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "project" / "workspace"
        self.local = self.workspace / "datasets" / "local"
        self.external = self.root / "other-project" / "dataset"
        for directory in (self.local, self.external):
            directory.mkdir(parents=True)
            (directory / "data.npy").write_bytes(b"preserved raw array fixture\x00\xff")
            (directory / "dataset.json").write_text(json.dumps({
                "schema": "ipde-depth-dataset-v1",
                "name": directory.name,
                "samples": [
                    {"id": "capture-a", "source_sha256": "a", "split": "train", "teacher": {"metadata": {"model_id": "DepthPro"}}},
                    {"id": "capture-b", "source_sha256": "b", "split": "validation", "teacher": {"metadata": {"model_id": "DepthPro"}}},
                ],
            }))

    def test_external_links_preserve_source_and_do_not_copy_arrays(self):
        before = {file: file.read_bytes() for file in self.external.rglob("*") if file.is_file()}
        report = workspace_report(self.workspace, [self.external])
        self.assertEqual(len(report["datasets"]), 2)
        linked = next(item for item in report["datasets"] if item["linked"])
        self.assertEqual(linked["path"], str(self.external.resolve()))
        self.assertEqual((linked["train_count"], linked["validation_count"]), (1, 1))
        self.assertEqual(before, {file: file.read_bytes() for file in self.external.rglob("*") if file.is_file()})
        self.assertEqual(list((self.workspace / "datasets").iterdir()), [self.local])

    def test_canonical_links_and_local_aliases_are_deduplicated(self):
        alias = self.root / "alias"
        alias.symlink_to(self.external, target_is_directory=True)
        report = workspace_report(self.workspace, [alias, self.external, self.local])
        self.assertEqual(len(report["datasets"]), 2)
        self.assertEqual(sum(item["linked"] for item in report["datasets"]), 1)
        self.assertFalse(report["warnings"])

    def test_missing_and_malformed_external_datasets_are_visible_warnings(self):
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "dataset.json").write_text('{"schema": "other"}')
        missing = self.root / "moved-dataset"
        report = workspace_report(self.workspace, [missing, broken])
        self.assertEqual(len(report["datasets"]), 1)
        self.assertEqual(len(report["warnings"]), 2)
        self.assertIn(str(missing), report["warnings"][0])
        self.assertIn(str(broken), report["warnings"][1])

    def test_cli_passes_repeated_links_to_workspace_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--json", "workspace", str(self.workspace),
                         "--linked-dataset", str(self.external), "--linked-dataset", str(self.external)])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(len(report["datasets"]), 2)
        self.assertEqual(report["datasets"][1]["path"], str(self.external.resolve()))

    def test_accidental_symlink_loop_does_not_hide_other_datasets(self):
        loop = self.root / "loop"
        loop.symlink_to(loop, target_is_directory=True)
        report = workspace_report(self.workspace, [loop, self.external])
        self.assertEqual(len(report["datasets"]), 2)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn(str(loop), report["warnings"][0])

    def test_materialized_copies_do_not_claim_inherited_shared_storage(self):
        manifest_path = self.local / "dataset.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["collection"] = {"storage_mode": "shared", "added_storage_bytes": 42,
                                  "reused_array_storage_bytes": 1024}
        manifest_path.write_text(json.dumps(manifest))
        report = workspace_report(self.workspace)
        self.assertEqual(report["datasets"][0]["storage"]["storage_mode"], "shared")
        for operation in ("curation", "storage_compaction"):
            with self.subTest(operation=operation):
                materialized = {**manifest, operation: {"source_manifest_sha256": "fixture"}}
                manifest_path.write_text(json.dumps(materialized))
                report = workspace_report(self.workspace)
                self.assertEqual(report["datasets"][0]["storage"], {})


if __name__ == "__main__":
    unittest.main()
