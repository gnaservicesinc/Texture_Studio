"""Teacher phases never overlap resident models or retain decoded photo arrays."""
from __future__ import annotations

from contextlib import redirect_stderr
import gc
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from ipde.dataset import DatasetError, DatasetOptions, build_dataset, load_dataset
from ipde.dataset_review import _array_records
from ipde.formats import sha256_array
from ipde.learned_depth import LearnedDepthConfig, LearnedDepthPredictor
from verification.test_display_teacher_source import _capture, _prediction


class TeacherBatchingTests(unittest.TestCase):
    def setUp(self):
        # Exercise historical scientific paths independently of the temporary admission policy.
        admission = patch("ipde.dataset.require_dataset_teacher")
        admission.start()
        self.addCleanup(admission.stop)

    def sources(self, root):
        sources = [root / f"capture-{index}.heic" for index in range(3)]
        for source in sources:
            source.write_bytes(b"original spatial photo")
        return sources

    def test_teacher_major_lifetime_decodes_once_and_keeps_stable_photo_major_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self.sources(root)
            calls, active, decoded, events = [], set(), [], []
            rgb_sources = {}
            def capture(source):
                result = _capture(source)
                result.assets[-1].array = np.roll(result.assets[-1].array, int(source.stem[-1]), axis=1).copy()
                rgb_sources[sha256_array(result.assets[-1].array)] = source.name
                decoded.extend(weakref.ref(asset.array) for asset in result.assets)
                return result
            class Predictor:
                def __init__(self, config):
                    gc.collect()
                    self.model = config.model
                    if active:
                        raise AssertionError("Two teachers were resident together")
                    if any(reference() is not None for reference in decoded):
                        raise AssertionError("Decoded source arrays remained resident across teacher phases")
                    active.add(self.model)
                    calls.append(("load", self.model))
                def __call__(self, rgb, **kwargs):
                    calls.append(("infer", self.model, rgb_sources[sha256_array(rgb)]))
                    self.assert_display(kwargs)
                    return _prediction(rgb)
                @staticmethod
                def assert_display(kwargs):
                    if kwargs != {"focal_pixels": None, "reference_label": "display"}:
                        raise AssertionError(f"Unexpected teacher image: {kwargs}")
                def close(self):
                    active.remove(self.model)
                    calls.append(("close", self.model))
            def progress(event):
                events.append(event)
                if event["event"] == "sample_ready":
                    snapshot = load_dataset(event["dataset_dir"])
                    self.assertEqual(snapshot["summary"]["samples"], event["summary"]["samples"])
                    self.assertEqual(snapshot["generation_state"], "generating")
            configs = tuple(LearnedDepthConfig(model=model) for model in ("depthpro", "depth-anything-3", "depth-anything-v2-small"))
            with patch("ipde.dataset.discover_file", side_effect=capture) as discover, \
                    patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=configs[0], additional_teachers=configs[1:], teacher_ids=("dp", "da3", "v2"),
                    workers=2, skip_bad_photos=False), progress_callback=progress)
            expected = []
            for config in configs:
                expected.append(("load", config.model))
                expected.extend(("infer", config.model, source.name) for source in sources)
                expected.append(("close", config.model))
            self.assertEqual(calls, expected)
            self.assertFalse(active)
            self.assertEqual(discover.call_count, len(sources))
            self.assertEqual([(Path(sample["source_path"]).name, sample["teacher_id"]) for sample in manifest["samples"]],
                             [(source.name, teacher) for source in sources for teacher in ("dp", "da3", "v2")])
            self.assertEqual([event["teacher_id"] for event in events if event["event"] == "model_loading"], ["dp", "da3", "v2"])
            self.assertEqual([event["teacher_id"] for event in events if event["event"] == "model_released"], ["dp", "da3", "v2"])
            self.assertEqual(load_dataset(root / "dataset"), manifest)
            self.assertEqual(manifest["summary"]["processed_sources"], 3)
            for source in sources:
                variants = [sample for sample in manifest["samples"] if Path(sample["source_path"]) == source.resolve()]
                self.assertEqual(len({sample["group_id"] for sample in variants}), 1)
                self.assertEqual(len({sample["split"] for sample in variants}), 1)

    def test_metric_anchor_phase_releases_depthpro_before_relative_teacher_and_prunes_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self.sources(root)
            active, calls = set(), []
            class Predictor:
                def __init__(self, config):
                    if active:
                        raise AssertionError("Metric anchor and teacher were resident together")
                    self.model = config.model
                    active.add(self.model)
                    calls.append(("load", self.model))
                def __call__(self, rgb, **kwargs):
                    calls.append(("infer", self.model))
                    relative = np.linspace(.2, 1.8, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
                    return _prediction(rgb, units="meters" if self.model == "depthpro" else "relative_inverse_depth",
                                       depth=1. / (relative * .25 + .1) if self.model == "depthpro" else relative)
                def close(self):
                    active.remove(self.model)
                    calls.append(("close", self.model))
            with patch("ipde.dataset.discover_file", side_effect=lambda source: _capture(source, display_shape=(96, 128))), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=LearnedDepthConfig(model="depth-anything-3"),
                    metric_anchor=LearnedDepthConfig(model="depthpro"), workers=1, skip_bad_photos=False))
            self.assertEqual(calls, [("load", "depthpro"), *[("infer", "depthpro")] * 3, ("close", "depthpro"),
                                    ("load", "depth-anything-3"), *[("infer", "depth-anything-3")] * 3, ("close", "depth-anything-3")])
            self.assertFalse(active)
            self.assertTrue(all(sample["training_target_choice"] == "anchored_display_teacher" for sample in manifest["samples"]))
            referenced = {record["path"] for record in _array_records(manifest["samples"])}
            written = {path.relative_to(root / "dataset").as_posix() for path in (root / "dataset").rglob("*") if path.suffix in {".exr", ".png", ".npz", ".npy"}}
            self.assertEqual(referenced, written)
            self.assertFalse(any("/native." in path for path in written))
            self.assertEqual(load_dataset(root / "dataset"), manifest)

    def test_failed_second_model_load_is_attempted_once_and_publishes_completed_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self.sources(root)
            loads, closes, events = [], [], []
            class Predictor:
                def __init__(self, config):
                    self.model = config.model
                    loads.append(self.model)
                    if self.model == "depth-anything-3":
                        raise RuntimeError("checkpoint is unavailable")
                def __call__(self, rgb, **kwargs):
                    return _prediction(rgb)
                def close(self):
                    closes.append(self.model)
            with patch("ipde.dataset.discover_file", side_effect=_capture), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=LearnedDepthConfig(model="depthpro"), additional_teachers=(LearnedDepthConfig(model="depth-anything-3"),),
                    teacher_ids=("dp", "da3"), workers=1), progress_callback=events.append)
            self.assertEqual(loads, ["depthpro", "depth-anything-3"])
            self.assertEqual(closes, ["depthpro"])
            self.assertEqual(len(manifest["samples"]), 3)
            self.assertTrue(all(sample["teacher_id"] == "dp" for sample in manifest["samples"]))
            self.assertEqual(len(manifest["skipped_sources"]), 3)
            self.assertTrue(all(record["teacher_id"] == "da3" and "checkpoint is unavailable" in record["reason"] for record in manifest["skipped_sources"]))
            self.assertEqual(len([event for event in events if event["event"] == "teacher_failed"]), 1)
            self.assertEqual(load_dataset(root / "dataset"), manifest)
            with patch("ipde.dataset.discover_file", side_effect=_capture), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(DatasetError, "Cannot load teacher da3"):
                    build_dataset(sources, root / "strict", DatasetOptions(
                        teacher=LearnedDepthConfig(model="depthpro"), additional_teachers=(LearnedDepthConfig(model="depth-anything-3"),),
                        teacher_ids=("dp", "da3"), workers=1, skip_bad_photos=False))
            self.assertEqual(closes, ["depthpro", "depthpro"])
            self.assertFalse((root / "strict").exists())
            self.assertFalse(list(root.glob(".strict-*")))
            self.assertEqual(load_dataset(root / "dataset"), manifest)

    def test_identical_depthpro_teacher_and_anchor_reuse_one_prediction_per_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self.sources(root)
            calls = []
            def factory(config):
                calls.append(("load", config.model))
                def predict(rgb, **kwargs):
                    calls.append(("infer", config.model))
                    relative = np.linspace(.2, 1.8, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
                    return _prediction(rgb, units="meters" if config.model == "depthpro" else "relative_inverse_depth",
                                       depth=1. / (relative * .25 + .1) if config.model == "depthpro" else relative)
                predict.close = lambda: calls.append(("close", config.model))
                return predict
            anchor = LearnedDepthConfig(model="depthpro", input_size=0)
            with patch("ipde.dataset.discover_file", side_effect=lambda source: _capture(source, display_shape=(96, 128))), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", side_effect=factory), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=anchor, additional_teachers=(LearnedDepthConfig(model="depth-anything-3"),),
                    metric_anchor=anchor, teacher_ids=("pro", "da3"), workers=1))
            self.assertEqual(calls, [("load", "depthpro"), *[("infer", "depthpro")] * 3, ("close", "depthpro"),
                                    ("load", "depth-anything-3"), *[("infer", "depth-anything-3")] * 3, ("close", "depth-anything-3")])
            self.assertEqual(len(manifest["samples"]), 6)
            for source in sources:
                labels = {sample["teacher_id"]: sample for sample in manifest["samples"] if Path(sample["source_path"]) == source.resolve()}
                self.assertEqual(labels["pro"]["display_teacher"]["target"], labels["da3"]["display_metric_anchor"]["target"])
                self.assertEqual(labels["pro"]["training_target_choice"], "display_teacher")
                self.assertEqual(labels["da3"]["training_target_choice"], "anchored_display_teacher")
            self.assertEqual(load_dataset(root / "dataset"), manifest)

    def test_inference_errors_keep_other_photos_and_release_before_next_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self.sources(root)
            active, closes = set(), []
            hashes = {sha256_array(_capture(source).assets[0].array): source.name for source in sources}
            class Predictor:
                def __init__(self, config):
                    if active:
                        raise AssertionError("The previous teacher was not released")
                    self.model = config.model
                    active.add(self.model)
                def __call__(self, rgb, **kwargs):
                    if self.model == "depthpro" and hashes[sha256_array(rgb)] == sources[1].name:
                        raise RuntimeError("bad inference for one photo")
                    return _prediction(rgb)
                def close(self):
                    active.remove(self.model)
                    closes.append(self.model)
            with patch("ipde.dataset.discover_file", side_effect=_capture), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                manifest = build_dataset(sources, root / "dataset", DatasetOptions(
                    teacher=LearnedDepthConfig(model="depthpro"), additional_teachers=(LearnedDepthConfig(model="depth-anything-3", input_size=28),),
                    teacher_view="stereo-left", include_display_teacher=False, workers=1))
            self.assertEqual(len(manifest["samples"]), 5)
            self.assertEqual(len(manifest["skipped_sources"]), 1)
            self.assertEqual(closes, ["depthpro", "depth-anything-3"])
            self.assertFalse(active)
            self.assertEqual(load_dataset(root / "dataset"), manifest)

    def test_cancel_and_strict_error_release_model_and_remove_staging(self):
        for cancellation in (None, "sample_ready", "model_releasing"):
            with self.subTest(cancellation=cancellation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources = self.sources(root)
                closed = []
                class Predictor:
                    def __init__(self, config):
                        self.model = config.model
                    def __call__(self, rgb, **kwargs):
                        if not cancellation:
                            raise RuntimeError("strict inference failure")
                        return _prediction(rgb)
                    def close(self):
                        closed.append(self.model)
                def progress(event):
                    if cancellation and event["event"] == cancellation:
                        raise KeyboardInterrupt("cancel")
                with patch("ipde.dataset.discover_file", side_effect=_capture), \
                        patch("ipde.learned_depth.LearnedDepthPredictor", Predictor), redirect_stderr(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt if cancellation else RuntimeError):
                        build_dataset(sources, root / "dataset", DatasetOptions(
                            teacher=LearnedDepthConfig(), workers=1, skip_bad_photos=False), progress_callback=progress)
                self.assertEqual(closed, ["depthpro"])
                self.assertFalse((root / "dataset").exists())
                self.assertFalse(list(root.glob(".dataset-*")))

    def test_close_releases_weights_and_only_the_active_accelerator_cache(self):
        for device in ("cpu", "mps", "cuda:0"):
            with self.subTest(device=device):
                predictor = LearnedDepthPredictor.__new__(LearnedDepthPredictor)
                predictor.model = SimpleNamespace(weights=np.ones(4))
                predictor.device = device
                mps, cuda = Mock(), Mock()
                mps.is_available.return_value = cuda.is_available.return_value = True
                predictor.torch = SimpleNamespace(mps=mps, cuda=cuda)
                predictor.close()
                predictor.close()
                self.assertIsNone(predictor.model)
                for name, backend in (("mps", mps), ("cuda", cuda)):
                    if device.startswith(name):
                        backend.synchronize.assert_called_once()
                        backend.empty_cache.assert_called_once()
                    else:
                        backend.empty_cache.assert_not_called()

    def test_removed_teacher_placeholder_still_validates_measured_reference_and_historical_paths(self):
        from ipde.array_storage import array_record, write_array
        from verification.test_collection_storage import _dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root)
            sample = samples[0]
            reference = np.full((64, 96), 3., np.float32)
            record = array_record(source, source / "measured-reference.npz", reference)
            sample["reference"] = {"units": "meters", "target": record,
                                   "valid_mask": dict(sample["teacher"]["valid_mask"])}
            sample.update(excluded=True, teacher_payload_removed=True)
            historical = source / sample["teacher"]["target"]["path"]
            historical.unlink()
            manifest = {"schema": "ipde-depth-dataset-v1", "samples": samples}
            manifest_path = source / "dataset.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(load_dataset(source), manifest)
            for metadata_only in (False, True):
                damaged = json.loads(json.dumps(manifest))
                damaged["samples"][0]["reference"].pop("valid_mask")
                manifest_path.write_text(json.dumps(damaged))
                with self.subTest(metadata_only=metadata_only), self.assertRaisesRegex(DatasetError, "reference.valid_mask"):
                    load_dataset(source, metadata_only=metadata_only)
            for path in ("../escape.exr", "/absolute/escape.exr", "unrelated.txt"):
                damaged = json.loads(json.dumps(manifest))
                damaged["samples"][0]["teacher"]["target"]["path"] = path
                manifest_path.write_text(json.dumps(damaged))
                with self.subTest(path=path), self.assertRaisesRegex(DatasetError, "escapes|NPY"):
                    load_dataset(source, metadata_only=True)
            manifest_path.write_text(json.dumps(manifest))
            write_array(source / record["path"], reference + np.float32(1.))
            with self.assertRaisesRegex(DatasetError, "checksum"):
                load_dataset(source)


if __name__ == "__main__":
    unittest.main()
