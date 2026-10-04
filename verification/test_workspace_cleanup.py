"""Explicit cleanup cannot remove models, source photos or active datasets."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ipde.formats import sha256_file
from ipde.resource_lock import resource_lock
from ipde.trainer_cli import main
from ipde.workspace_cleanup import archive_dataset, cleanup_dataset, cleanup_run


class CleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.dataset = self.workspace / "datasets" / "example"
        self.run = self.workspace / "runs" / "example"
        self.dataset.mkdir(parents=True)
        self.run.mkdir(parents=True)
        self.photo = self.root / "original.HEIC"
        self.photo.write_bytes(b"original source photo")
        self.payload = self.dataset / "data.npy"
        self.payload.write_bytes(b"precision preserved dataset payload")
        self.manifest = {"schema": "ipde-depth-dataset-v1", "samples": [{"source_path": str(self.photo)}]}
        (self.dataset / "dataset.json").write_text(json.dumps(self.manifest))
        self.checkpoint = self.run / "checkpoint.pth"
        self.checkpoint.write_bytes(b"trained model")
        self.report = self.run / "checkpoint.pth.json"
        self.report.write_text(json.dumps({"schema": "ipde-raft-training-report-v1",
                                          "checkpoint_sha256": sha256_file(self.checkpoint)}))

    def test_cleanup_needs_confirmation_and_keeps_model_and_source(self):
        for function, path in ((cleanup_dataset, self.dataset), (cleanup_run, self.checkpoint)):
            with self.assertRaisesRegex(ValueError, "confirmation"):
                function(path, self.workspace)
        saved = self.checkpoint.read_bytes(), self.report.read_bytes(), self.photo.read_bytes()
        cleanup_dataset(self.dataset, self.workspace, confirm=True)
        self.assertFalse(self.dataset.exists())
        self.assertEqual(saved, (self.checkpoint.read_bytes(), self.report.read_bytes(), self.photo.read_bytes()))

    def test_external_and_symlinked_dataset_refused(self):
        other = self.root / "external"
        other.mkdir()
        alias = self.workspace / "datasets" / "linked"
        alias.symlink_to(other, target_is_directory=True)
        for path in (other, alias):
            with self.assertRaisesRegex(ValueError, "owned"):
                cleanup_dataset(path, self.workspace, confirm=True)
        self.assertTrue(other.is_dir())

    def test_contained_source_and_model_refused_before_removal(self):
        self.manifest["samples"][0]["source_path"] = str(self.payload)
        (self.dataset / "dataset.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "source"):
            cleanup_dataset(self.dataset, self.workspace, confirm=True)
        self.assertTrue(self.payload.exists())
        self.manifest["samples"][0]["source_path"] = str(self.photo)
        (self.dataset / "dataset.json").write_text(json.dumps(self.manifest))
        (self.dataset / "surprise.pth").write_bytes(b"must retain")
        with self.assertRaisesRegex(ValueError, "model"):
            cleanup_dataset(self.dataset, self.workspace, confirm=True)
        self.assertTrue(self.payload.exists())

    def test_shared_payload_remains_after_source_dataset_cleanup(self):
        import os
        prepared = self.workspace / "datasets" / "prepared"
        prepared.mkdir()
        duplicate = prepared / "payload.npy"
        os.link(self.payload, duplicate)
        (prepared / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1", "samples": [
            {"target": {"path": "payload.npy", "shape": [1], "dtype": "|u1", "array_sha256": "a", "file_sha256": "b"}}
        ]}))
        expected = duplicate.read_bytes()
        cleanup_dataset(self.dataset, self.workspace, confirm=True)
        self.assertEqual(expected, duplicate.read_bytes())

    def test_external_array_dependency_blocks_cleanup(self):
        dependent = self.workspace / "datasets" / "dependent"
        dependent.mkdir()
        (dependent / "dataset.json").write_text(json.dumps({"samples": [
            {"target": {"path": "../example/data.npy", "shape": [1], "dtype": "|u1", "array_sha256": "a", "file_sha256": "b"}}
        ]}))
        with self.assertRaisesRegex(ValueError, "depends"):
            cleanup_dataset(self.dataset, self.workspace, confirm=True)
        self.assertTrue(self.payload.exists())

    def test_cleanup_conflicts_with_active_reader_and_run_writer(self):
        with resource_lock(self.dataset, shared=True):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                cleanup_dataset(self.dataset, self.workspace, confirm=True)
        with resource_lock(self.run):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                cleanup_run(self.checkpoint, self.workspace, confirm=True)
        self.assertTrue(self.payload.exists())
        self.assertTrue(self.checkpoint.exists())

    def test_archive_is_locked_and_preserves_every_byte(self):
        before = {path.name: path.read_bytes() for path in self.dataset.iterdir()}
        with resource_lock(self.dataset, shared=True):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                archive_dataset(self.dataset, self.workspace)
        result = archive_dataset(self.dataset, self.workspace)
        archived = Path(result["archived_dataset"])
        self.assertFalse(self.dataset.exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in archived.iterdir()})

    def test_run_cleanup_retains_all_models_and_reports(self):
        cache = self.run / "cache"
        cache.mkdir()
        (cache / "targets.npy").write_bytes(b"generated targets")
        (self.run / "training.log").write_bytes(b"training logs")
        another = self.run / "another.pth"
        another.write_bytes(b"another output model")
        before = self.checkpoint.read_bytes(), self.report.read_bytes(), another.read_bytes()
        result = cleanup_run(self.checkpoint, self.workspace, confirm=True)
        self.assertEqual(result["removed_files"], 2)
        self.assertFalse(cache.exists())
        self.assertEqual(before, (self.checkpoint.read_bytes(), self.report.read_bytes(), another.read_bytes()))

    def test_bad_checkpoint_hash_and_source_data_block_run_cleanup(self):
        self.checkpoint.write_bytes(b"corrupted")
        log = self.run / "training.log"
        log.write_bytes(b"keep on failure")
        with self.assertRaisesRegex(ValueError, "verification"):
            cleanup_run(self.checkpoint, self.workspace, confirm=True)
        self.assertTrue(log.exists())
        self.report.write_text(json.dumps({"schema": "ipde-raft-training-report-v1",
                                          "checkpoint_sha256": sha256_file(self.checkpoint)}))
        (self.run / "dataset.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source"):
            cleanup_run(self.checkpoint, self.workspace, confirm=True)
        self.assertTrue(log.exists())

    def test_cli_does_not_delete_without_confirmation(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--json", "cleanup-dataset", str(self.dataset), "--workspace", str(self.workspace)])
        self.assertEqual(status, 1)
        self.assertIn("confirmation", json.loads(output.getvalue())["error"])
        self.assertTrue(self.payload.exists())
