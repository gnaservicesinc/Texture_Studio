"""Training-set preparation reuses verified storage without changing inputs."""
from __future__ import annotations

import errno
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.dataset import DatasetError, load_dataset
from ipde.dataset_collection import CollectionOptions, compose_datasets, compress_dataset
from ipde.dataset_review import _array_records, curate_dataset
from ipde.training import TrainingOptions, _load_sample


def _dataset(root: Path) -> tuple[Path, list[dict]]:
    source = root / "source"
    source.mkdir()
    samples = []
    for index in range(3):
        left = np.random.default_rng(index).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        depth = np.full((64, 96), 2., dtype=">f4")
        # Preserve a particular NaN payload, negative zero and byte order.
        depth.view(">u4")[0, :2] = [0x7fc01234, 0x80000000]
        rgb = array_record(source, source / f"rgb-{index}.npy", left, compressed=False)
        right = array_record(source, source / f"right-{index}.npy", np.roll(left, -2, axis=1), compressed=False)
        target = array_record(source, source / f"target-{index}.npz", depth, compressed=True)
        mask = array_record(source, source / f"mask-{index}.npy", np.isfinite(depth), compressed=False)
        raw = array_record(source, source / f"raw-{index}.npy",
            np.array([[0x7e55, 0x8000, 0x3555]], np.uint16).view(np.float16), compressed=False)
        samples.append({"id": f"sample-{index}", "source_path": f"photo-{index}.HEIC",
            "source_sha256": f"source-{index}", "group_id": f"scene-{index}",
            "split": "validation" if index == 0 else "train", "rgb": rgb,
            "right_rgb": right, "raw_assets": [{"storage": raw}],
            "calibration": {"raft_stereo_ready": True, "rectified_stereo_ready": True,
                "left_camera": {"width": 96, "height": 64, "focal_length_x_pixels": 100.},
                "right_camera": {"width": 96, "height": 64, "focal_length_x_pixels": 100.},
                "focal_length_pixels_for_depth": 100., "baseline_meters": .04,
                "principal_point_delta_x_pixels": .5, "left_image_index": 1, "right_image_index": 2},
            "teacher": {"units": "meters", "target": target, "native_target": target,
                "valid_mask": mask, "metadata": {"checkpoint_sha256": "a" * 64}}})
    (source / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1", "samples": samples}))
    return source, samples


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class CollectionStorageTests(unittest.TestCase):
    def test_default_shares_original_files_without_array_writing_and_survives_source_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root)
            before = _file_bytes(source)
            output, events = root / "prepared", []
            with patch("ipde.dataset_collection._clone_array_file", return_value=False), \
                 patch("ipde.dataset_collection.array_record", side_effect=AssertionError("Unexpected array writer")), \
                 patch("shutil.copyfile", side_effect=AssertionError("Unexpected payload copy")):
                report = compose_datasets([source], output, progress_callback=events.append)
            saved = load_dataset(output)
            representatives = {}
            for record in _array_records(samples):
                representatives.setdefault((record["dtype"], tuple(record["shape"]), record["array_sha256"]), record)
            for record in _array_records(saved["samples"]):
                original = representatives[(record["dtype"], tuple(record["shape"]), record["array_sha256"])]
                src, dst = source / original["path"], output / record["path"]
                self.assertEqual((src.stat().st_dev, src.stat().st_ino), (dst.stat().st_dev, dst.stat().st_ino))
                self.assertEqual(record["file_sha256"], original["file_sha256"])
                self.assertEqual(dst.read_bytes(), before[original["path"]])
                self.assertFalse(dst.is_symlink())
                self.assertTrue(dst.resolve().is_relative_to(output.resolve()))
            self.assertEqual(before, _file_bytes(source))
            self.assertEqual(report["collection"]["storage_mode"], "shared")
            self.assertEqual(report["collection"]["storage_methods"], {"hardlink": len(representatives)})
            self.assertEqual(report["collection"]["added_array_storage_bytes"], 0)
            self.assertEqual(report["collection"]["added_storage_bytes"], (output / "dataset.json").stat().st_size)
            self.assertEqual(report["collection"]["reused_array_storage_bytes"],
                sum(path.stat().st_size for path in (output / "arrays").iterdir()))
            self.assertTrue(all(event["storage_mode"] == "shared" for event in events if event["phase"] == "sample_composed"))

            shutil.rmtree(source)
            saved = load_dataset(output)
            # Existing training preprocessing and materialization readers accept
            # shared files without any special path or storage-mode handling.
            for sample in saved["samples"]:
                self.assertTrue(_load_sample(output, sample, TrainingOptions())["valid"].any())
            curated = root / "curated"
            curate_dataset(output, [sample["id"] for sample in saved["samples"]], curated)
            compact = root / "compact"
            compress_dataset(output, compact)
            for materialized in (curated, compact):
                recovered = load_dataset(materialized)
                for old, new in zip(_array_records(saved["samples"]), _array_records(recovered["samples"])):
                    self.assertEqual(read_array(output / old["path"]).tobytes(), read_array(materialized / new["path"]).tobytes())
                    self.assertEqual(old["dtype"], new["dtype"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS copy-on-write clone support")
    def test_native_clone_keeps_independent_values_after_fixture_source_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root)
            output = root / "prepared"
            report = compose_datasets([source], output)
            if "hardlink" in report["collection"]["storage_methods"]:
                self.skipTest("Fixture filesystem does not support copy-on-write clones")
            self.assertEqual(report["collection"]["storage_methods"], {"clonefile": report["collection"]["unique_arrays"]})
            self.assertEqual(report["collection"]["added_array_storage_bytes"], 0)
            saved = load_dataset(output)
            original = samples[0]
            clone = next(s for s in saved["samples"] if s["collection_provenance"]["source_sample_id"] == original["id"])
            src, dst = source / original["rgb"]["path"], output / clone["rgb"]["path"]
            self.assertNotEqual(src.stat().st_ino, dst.stat().st_ino)
            before = dst.read_bytes()
            changed = np.load(src, mmap_mode="r+", allow_pickle=False)
            changed[0, 0, 0] ^= np.uint8(255)
            changed.flush()
            del changed
            self.assertNotEqual(src.read_bytes(), before)
            self.assertEqual(dst.read_bytes(), before)
            load_dataset(output)

    def test_link_failure_rolls_back_without_falling_back_to_a_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            before = _file_bytes(source)
            output = root / "prepared"
            import os
            link = os.link
            calls = 0
            def failing_link(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError(errno.EXDEV, "Different volume")
                return link(*args, **kwargs)
            with patch("ipde.dataset_collection._clone_array_file", return_value=False), \
                 patch("ipde.dataset_collection.os.link", side_effect=failing_link), \
                 patch("ipde.dataset_collection.array_record", side_effect=AssertionError("Unexpected copy fallback")):
                with self.assertRaisesRegex(DatasetError, "same filesystem.*existing dataset directly"):
                    compose_datasets([source], output)
            self.assertEqual(calls, 2)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [source])
            self.assertEqual(before, _file_bytes(source))
            self.assertTrue(all(path.stat().st_nlink == 1 for path in source.iterdir() if path.is_file()))

    def test_explicit_portable_copy_and_shared_set_preserve_identical_splits_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            before = _file_bytes(source)
            shared, portable = root / "shared", root / "portable"
            compose_datasets([source], shared)
            report = compose_datasets([source], portable, CollectionOptions(storage_mode="copy"))
            left, right = load_dataset(shared), load_dataset(portable)
            self.assertEqual([(s["id"], s["group_id"], s["split"]) for s in left["samples"]],
                [(s["id"], s["group_id"], s["split"]) for s in right["samples"]])
            for old, new in zip(_array_records(left["samples"]), _array_records(right["samples"])):
                self.assertEqual(old["array_sha256"], new["array_sha256"])
                self.assertEqual(read_array(shared / old["path"]).tobytes(), read_array(portable / new["path"]).tobytes())
            self.assertTrue(all(path.suffix == ".npz" for path in (portable / "arrays").iterdir()))
            self.assertEqual(report["collection"]["reused_array_storage_bytes"], 0)
            self.assertEqual(report["collection"]["added_storage_bytes"], sum(len(value) for value in _file_bytes(portable).values()))
            self.assertEqual(before, _file_bytes(source))

    def test_escaping_source_and_corrupted_output_are_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            manifest_path = source / "dataset.json"
            original = manifest_path.read_bytes()
            manifest = json.loads(original)
            manifest["samples"][0]["rgb"]["path"] = "../elsewhere.npy"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "escapes"):
                compose_datasets([source], root / "escaped")
            manifest_path.write_bytes(original)
            before = _file_bytes(source)
            def corrupt_clone(src, dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes()[:-1])
                return "clonefile"
            with patch("ipde.dataset_collection._share_array_file", side_effect=corrupt_clone):
                with self.assertRaises(DatasetError):
                    compose_datasets([source], root / "corrupted")
            self.assertEqual(list(root.iterdir()), [source])
            self.assertEqual(before, _file_bytes(source))


if __name__ == "__main__":
    unittest.main()
