"""Persistent membership changes affect readers without touching scientific data."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np

from ipde.dataset import DatasetError, load_dataset
from ipde.dataset_collection import compose_datasets
from ipde.dataset_review import _array_records, _read_manifest_snapshot, preview_sample, review_dataset
from ipde.training import TrainingError, TrainingOptions, _check_splits, train_dataset, training_target_eligibility
from verification.test_collection_storage import _dataset
from verification.test_training_progress import _dataset as _training_dataset, _fake_raft_modules


class DatasetMembershipConsumerTests(unittest.TestCase):
    def test_review_reopens_exclusions_pending_photos_and_percent_without_array_access(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            path = source / "dataset.json"
            manifest = json.loads(path.read_text())
            manifest["samples"][1]["excluded"] = True
            manifest.update(validation_fraction=1., edit_revision=7,
                            pending_photos=[str(Path(directory) / "new.HEIC")])
            path.write_text(json.dumps(manifest))
            original = path.read_bytes()
            # A removed entry stays restorable even if its generated data has
            # become unavailable; inspection reports errors only on actual use.
            for record in _array_records(manifest["samples"][1]):
                (source / record["path"]).unlink(missing_ok=True)
            with patch("ipde.dataset.read_array", side_effect=AssertionError("Array decoded")), \
                 patch("ipde.dataset_review.read_array", side_effect=AssertionError("Array decoded")), \
                 patch("ipde.dataset_review.sha256_file", side_effect=AssertionError("Payload hashed")):
                first, reopened = review_dataset(source), review_dataset(source)
                snapshot = load_dataset(source, metadata_only=True)
            self.assertEqual(first, reopened)
            self.assertEqual((first["summary"]["samples"], first["excluded_samples"]), (2, 1))
            self.assertEqual((first["summary"]["train_samples"], first["summary"]["validation_samples"]), (1, 1))
            removed = next(sample for sample in first["samples"] if sample["id"] == "sample-1")
            self.assertTrue(removed["excluded"])
            self.assertFalse(removed["included"])
            self.assertEqual(first["validation_fraction"], 1.)
            self.assertEqual(first["pending_photos"], manifest["pending_photos"])
            self.assertEqual(first["manifest_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(first["edit_revision"], 7)
            self.assertEqual(snapshot, manifest)
            self.assertEqual(path.read_bytes(), original)
            load_dataset(source, include_excluded=False)
            with self.assertRaisesRegex(DatasetError, "missing"):
                load_dataset(source)
            with self.assertRaisesRegex(DatasetError, "missing"):
                preview_sample(source, "sample-1", "training", Path(directory) / "preview")

    def test_composition_omits_excluded_sources_and_preserves_remaining_array_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            path = source / "dataset.json"
            manifest = json.loads(path.read_text())
            manifest["samples"][1]["excluded"] = True
            path.write_text(json.dumps(manifest))
            original = {item.name: item.read_bytes() for item in source.iterdir() if item.is_file()}
            for record in _array_records(manifest["samples"][1]):
                (source / record["path"]).unlink(missing_ok=True)
            report = compose_datasets([source], root / "composed", workers=1)
            saved = load_dataset(root / "composed")
            self.assertEqual(report["summary"]["samples"], 2)
            self.assertEqual({sample["collection_provenance"]["source_sample_id"] for sample in saved["samples"]},
                             {"sample-0", "sample-2"})
            self.assertEqual(report["collection"]["sources"][0]["excluded_source_samples"], 1)
            original_hashes = {record["file_sha256"]: original[record["path"]]
                               for sample in manifest["samples"] for record in _array_records(sample)}
            for record in _array_records(saved["samples"]):
                self.assertEqual((root / "composed" / record["path"]).read_bytes(), original_hashes[record["file_sha256"]])

    def test_all_validation_and_all_removed_are_metadata_valid_but_cannot_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            path = source / "dataset.json"
            manifest = json.loads(path.read_text())
            for state in ("validation", "removed"):
                for sample in manifest["samples"]:
                    sample["split"] = "validation"
                    sample["excluded"] = state == "removed"
                manifest["validation_fraction"] = 1.
                path.write_text(json.dumps(manifest))
                with patch("ipde.dataset.read_array", side_effect=AssertionError("Array decoded before rejecting training")):
                    reviewed = review_dataset(source)
                    eligibility = training_target_eligibility(manifest)
                    with self.assertRaisesRegex(TrainingError, "train.*validation"):
                        train_dataset(source, root / f"{state}.pth", TrainingOptions(patch_size=64))
                self.assertFalse(eligibility["trainable"])
                self.assertEqual(eligibility["train_count"], 0)
                self.assertEqual(eligibility["sample_count"], 0 if state == "removed" else 3)
                self.assertEqual(reviewed["summary"]["samples"], 0 if state == "removed" else 3)

    def test_payload_verification_uses_captured_snapshot_after_manifest_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            path = source / "dataset.json"
            _, original, digest = _read_manifest_snapshot(source, validate_files=False)
            updated = json.loads(path.read_text())
            for sample in updated["samples"]:
                sample["excluded"] = True
                sample["split"] = "validation"
            replacement = path.with_suffix(".tmp")
            replacement.write_text(json.dumps(updated))
            replacement.replace(path)
            verified = load_dataset(source, snapshot=original, include_excluded=False)
            self.assertEqual(verified, original)
            self.assertNotEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            train, validation = _check_splits(verified["samples"])
            self.assertEqual((len(train), len(validation)), (2, 1))
            self.assertEqual(load_dataset(source, metadata_only=True), updated)

    def test_training_skips_removed_entries_and_keeps_original_snapshot_provenance(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _training_dataset(root, count=4)
            path = source / "dataset.json"
            manifest = json.loads(path.read_text())
            manifest["edit_revision"] = 3
            for sample in manifest["samples"][2:]:
                sample["excluded"] = True
            path.write_text(json.dumps(manifest))
            original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            arrays_before = {item.name: item.read_bytes() for item in source.iterdir() if item.suffix != ".json"}
            checkpoint = root / "baseline.pth"
            torch.save(fixture(None).state_dict(), checkpoint)

            def verify_snapshot(directory, **options):
                # A second window may atomically edit membership after this
                # run captures it. Array consumption keeps the original state.
                updated = json.loads(path.read_text())
                updated["edit_revision"] = 4
                for sample in updated["samples"]:
                    sample.update(excluded=True, split="validation")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(updated))
                temporary.replace(path)
                return load_dataset(directory, **options)

            from ipde.training import _load_sample
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.load_dataset", side_effect=verify_snapshot), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, checkpoint, None)), \
                 patch("ipde.training._load_sample", wraps=_load_sample) as loading, redirect_stderr(io.StringIO()):
                report = train_dataset(source, root / "trained.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu"))
            self.assertEqual({call.args[1]["id"] for call in loading.call_args_list}, {"sample-0", "sample-1"})
            self.assertEqual((report["sample_count"], report["removed_count"], report["train_count"], report["validation_count"]), (2, 2, 1, 1))
            self.assertEqual(report["dataset_manifest_sha256"], original_digest)
            self.assertEqual(report["dataset_edit_revision"], 3)
            self.assertEqual(json.loads(path.read_text())["edit_revision"], 4)
            self.assertEqual(arrays_before, {item.name: item.read_bytes() for item in source.iterdir() if item.suffix != ".json"})

    def test_metadata_rejects_malformed_membership_and_escaping_paths_without_payload_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            path = source / "dataset.json"
            original = json.loads(path.read_text())
            for field, value, message in (("excluded", "true", "membership"), ("path", "../escaped.npy", "escapes")):
                manifest = json.loads(json.dumps(original))
                if field == "path":
                    manifest["samples"][0]["rgb"][field] = value
                else:
                    manifest["samples"][0][field] = value
                path.write_text(json.dumps(manifest))
                with patch("ipde.dataset.read_array", side_effect=AssertionError("Array decoded")), \
                     self.assertRaisesRegex(DatasetError, message):
                    load_dataset(source, metadata_only=True)

    def test_incomplete_scientific_records_cannot_hide_corrupted_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            path = source / "dataset.json"
            original = json.loads(path.read_text())
            for plane in ("valid_mask", "native_target", "raw_asset", "future_product"):
                with self.subTest(plane=plane):
                    manifest = json.loads(json.dumps(original))
                    sample = manifest["samples"][0]
                    if plane == "raw_asset":
                        record = sample["raw_assets"][0]["storage"]
                    elif plane == "future_product":
                        record = dict(sample["teacher"]["target"])
                        sample["future_product"] = record
                    else:
                        record = sample["teacher"][plane]
                    payload_path = source / record["path"]
                    if plane == "valid_mask":
                        np.save(payload_path, np.zeros(record["shape"], dtype=bool), allow_pickle=False)
                    else:
                        payload_path.write_bytes(b"corrupted scientific plane")
                    record.pop("array_sha256")
                    path.write_text(json.dumps(manifest))
                    before = payload_path.read_bytes()
                    with patch("ipde.dataset.read_array", side_effect=AssertionError("Incomplete record decoded")), \
                         patch("ipde.dataset.sha256_file", side_effect=AssertionError("Incomplete record hashed")):
                        for metadata_only in (False, True):
                            with self.assertRaisesRegex(DatasetError, "incomplete scientific array record"):
                                load_dataset(source, metadata_only=metadata_only)
                    self.assertEqual(payload_path.read_bytes(), before)

    def test_native_and_raw_asset_escaping_paths_are_rejected_before_payload_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            path = source / "dataset.json"
            original = json.loads(path.read_text())
            for plane in ("native_target", "raw_asset"):
                with self.subTest(plane=plane):
                    manifest = json.loads(json.dumps(original))
                    sample = manifest["samples"][0]
                    record = sample["raw_assets"][0]["storage"] if plane == "raw_asset" else sample["teacher"][plane]
                    record["path"] = "../outside.npy"
                    path.write_text(json.dumps(manifest))
                    with patch("ipde.dataset.read_array", side_effect=AssertionError("Escaping record decoded")), \
                         self.assertRaisesRegex(DatasetError, "escapes"):
                        load_dataset(source)


if __name__ == "__main__":
    unittest.main()
