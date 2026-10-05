"""Interrupted generation is recovered only with ownership and array proofs."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.dataset_recovery import generation_status, recover_dataset
from ipde.dataset_review import review_dataset
from ipde.formats import sha256_file
from ipde.learned_depth import LearnedDepthConfig
from ipde.resource_lock import resource_lock
from ipde.trainer_cli import main
from verification.test_dataset_edit import _dataset, _bytes


class DatasetRecoveryTests(unittest.TestCase):
    def partial(self, parent, *, final_name="recovered", duplicate=True):
        parent = parent.resolve()
        destination = parent / final_name
        root, samples = _dataset(parent / f".{final_name}-interrupted")
        manifest = json.loads((root / "dataset.json").read_text())
        for sample in manifest["samples"]:
            sample.update(teacher_id="depthpro", teacher_model="depthpro", requested_group=None)
        if duplicate:
            variant = copy.deepcopy(manifest["samples"][0])
            variant.update(id="photo-0-v2", teacher_id="depth-anything-v2", teacher_model="depth-anything-v2")
            manifest["samples"].append(variant)
        manifest.update(generation_state="generating", splits_provisional=True,
                        generation_output_dir=str(destination), validation_fraction=.25, split_seed=41,
                        teachers=[{"id": model, "model": model} for model in ("depthpro", "depth-anything-v2")],
                        summary={"samples": len(manifest["samples"]), "processed_sources": 0, "total_sources": 4},
                        custom_generation_metadata={"keep": "exactly"})
        (root / "dataset.json").write_text(json.dumps(manifest))
        (root / "partial-teacher").mkdir()
        (root / "partial-teacher/incomplete.bin").write_bytes(b"preserve unfinished experiment metadata")
        (root / "events.jsonl").write_text('{"event":"model_loaded"}\n')
        return root, destination, manifest

    def test_verified_completed_samples_promote_without_rewriting_any_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, original = self.partial(Path(directory))
            before = _bytes(root)
            digest = sha256_file(root / "dataset.json")
            self.assertEqual(review_dataset(root)["generation_status"]["status"], "interrupted")
            result = recover_dataset(root, expected_manifest_sha256=digest, workers=1)
            self.assertEqual((result["dataset_path"], result["promoted"], result["recovered_samples"]), (str(destination), True, 5))
            self.assertFalse(root.exists())
            saved = load_dataset(destination)
            self.assertEqual(saved["generation_state"], "complete")
            self.assertFalse(saved["splits_provisional"])
            self.assertEqual(saved["custom_generation_metadata"], original["custom_generation_metadata"])
            self.assertEqual(saved["custom_dataset_metadata"], original["custom_dataset_metadata"])
            self.assertEqual(saved["teachers"], original["teachers"])
            self.assertEqual({name: value for name, value in before.items() if name != "dataset.json"},
                             {name: value for name, value in _bytes(destination).items() if name != "dataset.json"})
            variants = [sample for sample in saved["samples"] if sample["source_sha256"] == "source-0"]
            self.assertEqual(len({(sample["split"], sample["group_id"]) for sample in variants}), 1)
            self.assertEqual((saved["summary"]["train_samples"], saved["summary"]["validation_samples"]), (3, 2))
            self.assertEqual(len({sample["group_id"] for sample in saved["samples"] if sample["split"] == "validation"}), 1)
            self.assertEqual(saved["generation_recovery"]["missing_requested_teacher_pairs"], 3)
            for old, new in zip(original["samples"], saved["samples"]):
                self.assertEqual({key: value for key, value in old.items() if key not in ("group_id", "split")},
                                 {key: value for key, value in new.items() if key not in ("group_id", "split")})

    def test_parent_cli_and_backend_generation_owner_both_block_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, _ = self.partial(Path(directory))
            before = _bytes(root)
            for resource in (destination, destination / ".ipde-generation-owner", root / "dataset.json"):
                with resource_lock(resource):
                    status = generation_status(root)
                    self.assertTrue(status["active"])
                    self.assertFalse(status["recoverable"])
                    with self.assertRaisesRegex(RuntimeError, "in use"):
                        recover_dataset(root, expected_manifest_sha256=sha256_file(root / "dataset.json"))
                self.assertEqual(before, _bytes(root))

    def test_current_root_reader_does_not_masquerade_as_a_live_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.partial(Path(directory))
            with resource_lock(root, shared=True):
                self.assertTrue(generation_status(root)["recoverable"])

    def test_stale_snapshot_and_corrupt_completed_payload_refuse_every_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, _ = self.partial(Path(directory))
            before = _bytes(root)
            with self.assertRaisesRegex(DatasetError, "changed"):
                recover_dataset(root, expected_manifest_sha256="0" * 64)
            self.assertEqual(before, _bytes(root))
            digest = sha256_file(root / "dataset.json")
            target = root / "target-0.npy"
            data = bytearray(target.read_bytes()); data[-1] ^= 1; target.write_bytes(data)
            damaged = _bytes(root)
            with self.assertRaisesRegex(DatasetError, "checksum"):
                recover_dataset(root, expected_manifest_sha256=digest)
            self.assertEqual(damaged, _bytes(root))
            self.assertFalse(destination.exists())

    def test_existing_destination_is_never_replaced_and_replay_requires_content_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, _ = self.partial(Path(directory))
            destination.mkdir(); (destination / "keep.txt").write_text("unrelated user data")
            digest = sha256_file(root / "dataset.json")
            first = recover_dataset(root, expected_manifest_sha256=digest)
            self.assertFalse(first["promoted"])
            before = _bytes(root)
            repeated = recover_dataset(root, expected_manifest_sha256=digest)
            self.assertTrue(repeated["recovered_committed_request"])
            self.assertEqual(before, _bytes(root))
            self.assertEqual((destination / "keep.txt").read_text(), "unrelated user data")
            manifest = json.loads((root / "dataset.json").read_text()); manifest["custom_generation_metadata"]["keep"] = "changed"
            (root / "dataset.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "changed"):
                recover_dataset(root, expected_manifest_sha256=digest)

    def test_atomic_write_failure_keeps_original_manifest_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, _ = self.partial(Path(directory))
            before = _bytes(root)
            with patch("ipde.dataset_recovery.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    recover_dataset(root, expected_manifest_sha256=sha256_file(root / "dataset.json"))
            self.assertEqual(before, _bytes(root))
            self.assertFalse(destination.exists())

    def test_unsafe_recorded_destination_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, manifest = self.partial(Path(directory))
            unrelated = Path(directory) / "outside" / "do-not-create"
            manifest["generation_output_dir"] = str(unrelated)
            (root / "dataset.json").write_text(json.dumps(manifest))
            result = recover_dataset(root, expected_manifest_sha256=sha256_file(root / "dataset.json"))
            self.assertEqual(result["dataset_path"], str(root))
            self.assertFalse(result["promoted"])
            self.assertFalse(unrelated.exists())

    def test_cli_review_and_recover_keep_stdout_json_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root, destination, _ = self.partial(Path(directory))
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--json", "review-dataset", str(root)])
            self.assertEqual(status, 0, output.getvalue())
            reviewed = json.loads(output.getvalue())
            self.assertTrue(reviewed["generation_status"]["recoverable"])
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--json", "recover-dataset", str(root), "--expected-manifest-sha256", reviewed["manifest_sha256"], "--workers", "1"])
            self.assertEqual(status, 0, output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["dataset_path"], str(destination))

    def test_direct_python_generation_holds_owner_for_entire_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            def producer(*args, **kwargs):
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    with resource_lock(destination / ".ipde-generation-owner"):
                        pass
                return {"fixture": True}
            with patch("ipde.dataset._build_dataset", side_effect=producer):
                self.assertEqual(build_dataset([], destination, DatasetOptions(teacher=LearnedDepthConfig())), {"fixture": True})
            with resource_lock(destination / ".ipde-generation-owner"):
                pass


if __name__ == "__main__":
    unittest.main()
