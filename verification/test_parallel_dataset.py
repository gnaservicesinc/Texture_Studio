"""Concurrent scientific file work stays bounded and preserves its evidence."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.concurrency import memory_limited_workers, ordered_map, resolve_workers
from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.dataset_collection import CollectionOptions, compose_datasets, compress_dataset
from ipde.dataset_review import _array_records
from ipde.training import TrainingOptions, _SamplePool
from ipde.learned_depth import LearnedDepthConfig, LearnedDepthResult
from ipde.huggingface_datasets import import_dataset_rows
from ipde.formats import sha256_array
from verification.test_collection_storage import _dataset
from verification.test_spatial_scan import calibrated_capture


class ParallelDatasetTests(unittest.TestCase):
    def test_worker_defaults_validation_and_memory_window(self):
        with patch("ipde.concurrency.available_workers", return_value=12):
            self.assertEqual(resolve_workers(), 12)
            self.assertEqual(resolve_workers(0), 12)
            self.assertEqual(resolve_workers(3), 3)
            self.assertEqual(memory_limited_workers(12, 128 * 1024**2), 4)
        for value in (-1, True, 2.5, "4"):
            with self.assertRaisesRegex(ValueError, "workers"):
                resolve_workers(value)

    def test_ordered_map_runs_multiple_workers_and_bounds_active_tasks(self):
        active = maximum = 0
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def work(index):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            if index < 3:
                barrier.wait(timeout=5)
            with lock:
                active -= 1
            return index * 2

        self.assertEqual(list(ordered_map(work, range(21), workers=3)), [index * 2 for index in range(21)])
        self.assertEqual(maximum, 3)

    def test_verification_reads_each_file_once_and_rejects_conflicting_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            manifest_path = source / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            extra = array_record(source, source / "future.npz", np.array([1, -0.], ">f8"))
            manifest["samples"][0]["future_product"] = extra
            manifest_path.write_text(json.dumps(manifest))
            unique = {record["path"] for record in _array_records(manifest["samples"])}
            with patch("ipde.dataset.read_array", wraps=read_array) as loading:
                self.assertEqual(load_dataset(source, workers=4), manifest)
            self.assertEqual(loading.call_count, len(unique))
            manifest["samples"][0]["future_duplicate"] = {**extra, "array_sha256": "bad"}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "Conflicting dataset array records"):
                load_dataset(source, workers=4)

    def test_serial_and_parallel_compaction_and_composition_preserve_bits_and_owner_callbacks(self):
        owner = threading.get_ident()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            for operation in ("compact", "shared", "portable"):
                outputs = []
                for workers in (1, 4):
                    target = root / f"{operation}-{workers}"
                    callbacks = []
                    def callback(event):
                        self.assertEqual(threading.get_ident(), owner)
                        callbacks.append(event)
                    if operation == "compact":
                        compress_dataset(source, target, workers=workers, progress_callback=callback)
                    else:
                        compose_datasets([source], target,
                            CollectionOptions(storage_mode="copy" if operation == "portable" else "shared"),
                            workers=workers, progress_callback=callback)
                    self.assertTrue(callbacks)
                    outputs.append((target, load_dataset(target)))
                for first, second in zip(_array_records(outputs[0][1]["samples"]), _array_records(outputs[1][1]["samples"])):
                    a, b = read_array(outputs[0][0] / first["path"]), read_array(outputs[1][0] / second["path"])
                    self.assertEqual((a.dtype, a.shape, a.tobytes()), (b.dtype, b.shape, b.tobytes()))
                self.assertEqual([sample["split"] for sample in outputs[0][1]["samples"]],
                                 [sample["split"] for sample in outputs[1][1]["samples"]])
            self.assertEqual(original, {path: path.read_bytes() for path in source.rglob("*") if path.is_file()})

    def test_parallel_storage_failure_finishes_workers_before_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _dataset(root)
            output = root / "failed"
            with patch("ipde.dataset_collection.array_record", side_effect=OSError("Disk full")):
                with self.assertRaisesRegex(OSError, "Disk full"):
                    compress_dataset(source, output, workers=4)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".failed-*")))
            self.assertTrue(load_dataset(source))

    def test_parallel_target_preparation_matches_serial_and_keeps_cache_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = _dataset(Path(directory))
            samples = load_dataset(source)["samples"]
            results = []
            for workers in (1, 4):
                pool = _SamplePool(source, samples, TrainingOptions(patch_size=64, workers=workers, cache_samples=2))
                with redirect_stderr(io.StringIO()):
                    excluded = pool.prepare("training")
                self.assertLessEqual(len(pool.cache), 2)
                self.assertFalse(excluded)
                results.append((pool.details, pool.crop_origins,
                    {index: (value["flow"].tobytes(), value["valid"].tobytes()) for index, value in pool.cache.items()}))
            self.assertEqual(results[0], results[1])

    def test_heic_prefetch_and_raw_writes_keep_generation_journal_on_owner(self):
        owner = threading.get_ident()
        barrier = threading.Barrier(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / f"photo-{index}.heic" for index in range(3)]
            predictions = {}
            for source in sources:
                source.write_bytes(b"capture")
                depth = np.full((3, 4), 2., np.float32)
                predictions[str(source)] = LearnedDepthResult(depth, depth, {
                    "units": "meters", "checkpoint_sha256": "a" * 64,
                    "input_rgb_sha256": sha256_array(calibrated_capture(source).assets[0].array)})
            def discover(source):
                barrier.wait(timeout=5)
                result = calibrated_capture(source)
                result.source_sha256 = hashlib.sha256(source.name.encode()).hexdigest()
                result.spatial_photo["focal_length_pixels_for_depth"] = 20.
                return result
            events = []
            def callback(event):
                self.assertEqual(threading.get_ident(), owner)
                events.append(event)
            with patch("ipde.dataset.discover_file", side_effect=discover), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "generated", DatasetOptions(
                    teacher=LearnedDepthConfig(), include_display_teacher=False, workers=3),
                    teacher_results=predictions, progress_callback=callback)
            self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
            self.assertEqual([sample["source_path"] for sample in manifest["samples"]], [str(source.resolve()) for source in sources])
            verified = load_dataset(root / "generated", workers=3)
            self.assertEqual(verified, manifest)
            for sample in verified["samples"]:
                original = calibrated_capture(Path(sample["source_path"]))
                for asset, record in zip(original.assets, sample["raw_assets"]):
                    self.assertEqual(read_array(root / "generated" / record["storage"]["path"]).tobytes(), asset.array.tobytes())

    def test_huggingface_file_workers_preserve_meter_depth_bits_and_report_bad_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root)
            rows = [{"left": read_array(source / sample["rgb"]["path"]),
                     "right": read_array(source / sample["right_rgb"]["path"]),
                     "depth": read_array(source / sample["teacher"]["target"]["path"]),
                     "calibration": sample["calibration"], "scene": f"scene-{index}"}
                    for index, sample in enumerate(samples)]
            rows.append({**rows[0], "depth": b"not-a-lossless-array"})
            mapping = {"columns": {key: key for key in ("left", "right", "depth", "calibration", "scene")},
                       "depth_units": "meters", "coordinate_reference": "spatial_left", "depth_role": "measured",
                       "verified_scene_groups": True}
            outputs = []
            for workers in (1, 4):
                target = root / f"hf-{workers}"
                report = import_dataset_rows(rows, target, mapping, workers=workers)
                self.assertEqual(report["external_import"]["skipped_rows"][0]["row_index"], 3)
                outputs.append((target, load_dataset(target)))
            for sample_index in range(3):
                first, second = (manifest["samples"][sample_index] for _, manifest in outputs)
                for a, b in zip(_array_records(first), _array_records(second)):
                    x, y = read_array(outputs[0][0] / a["path"]), read_array(outputs[1][0] / b["path"])
                    self.assertEqual((x.dtype.str, x.tobytes()), (y.dtype.str, y.tobytes()))


if __name__ == "__main__":
    unittest.main()
