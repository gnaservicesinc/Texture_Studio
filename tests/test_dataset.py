from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ipde.dataset import DatasetError, DatasetOptions, assign_grouped_splits, build_dataset, load_dataset, teacher_depth_to_flow
from ipde.extractor import Asset, Discovery
from ipde.formats import arrays_bit_equal, sha256_array


def fixture_discovery(path: Path, seed: int = 0, shape: tuple[int, int] = (64, 64)) -> Discovery:
    rgb = np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)
    left = Asset("spatial_view", 1, 0, rgb, "RGB", 8, "spatial_left")
    right = Asset("spatial_view", 2, 0, np.roll(rgb, -2, axis=1), "RGB", 8, "spatial_right")
    raw_depth = Asset("native_depth", 0, 0, np.array([[1, np.nan], [3, -0.0]], dtype=np.float16), "F", 8, "apple_depth")
    h, w = shape
    calibration = {"raft_stereo_ready": True, "rectified_stereo_ready": True,
        "left_camera": {"width": w, "height": h, "focal_length_x_pixels": 100.0},
        "right_camera": {"width": w, "height": h, "focal_length_x_pixels": 100.0},
        "focal_length_pixels_for_depth": 100.0, "baseline_meters": .04,
        "principal_point_delta_x_pixels": .5, "left_image_index": 1, "right_image_index": 2,
        "group_disparity_adjustment_pixels": 1000.0}
    return Discovery(path, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), "image/heic", 0, [], [raw_depth, left, right], calibration)


def fixture_teacher(discovery: Discovery, *, units: str = "meters") -> SimpleNamespace:
    rgb = next(asset.array for asset in discovery.assets if asset.semantic_name == "spatial_left")
    depth = np.full(rgb.shape[:2], 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    return SimpleNamespace(source_depth=depth, native_depth=np.full((4, 4), 2.0, np.float32), confidence=None,
        metadata={"input_rgb_sha256": sha256_array(rgb), "checkpoint_sha256": "a" * 64, "units": units, "model_id": "test/teacher"})


def build_fixture(root: Path, *, count: int = 2) -> tuple[Path, dict, list[Discovery]]:
    sources, discoveries, teachers, groups = [], [], {}, {}
    for index in range(count):
        source = root / f"capture-{index}.HEIC"
        source.write_bytes(f"original-{index}".encode())
        discovery = fixture_discovery(source, index)
        sources.append(source)
        discoveries.append(discovery)
        teachers[str(source)] = fixture_teacher(discovery)
        groups[str(source)] = f"scene-{index}"
    destination = root / "dataset"
    options = DatasetOptions(SimpleNamespace(model="depthpro"), group_ids=groups, include_display_teacher=False)
    with patch("ipde.dataset.discover_file", side_effect=discoveries):
        manifest = build_dataset(sources, destination, options, teacher_results=teachers)
    return destination, manifest, discoveries


class DatasetTests(unittest.TestCase):
    def test_explicit_metric_anchor_preserves_relative_plane_and_creates_separate_pseudo_meter_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.HEIC"
            source.write_bytes(b"test")
            discovery = fixture_discovery(source)
            yy, xx = np.indices((64, 64), dtype=np.float32)
            depth = np.float32(1) / (np.float32(.2) + xx * np.float32(.0002) + yy * np.float32(.0005))
            relative = (np.float32(1) / depth - np.float32(.12)) / np.float32(.4)
            teacher = fixture_teacher(discovery, units="relative_inverse_depth")
            teacher.source_depth = relative
            teacher.native_depth = relative[:4, :4].copy()
            anchor = fixture_teacher(discovery)
            anchor.source_depth = depth
            anchor.native_depth = depth[:4, :4].copy()
            anchor.metadata["checkpoint_sha256"] = "b" * 64
            with patch("ipde.dataset.discover_file", return_value=discovery):
                manifest = build_dataset([source], root / "dataset", DatasetOptions(None, metric_anchor=object(), include_display_teacher=False),
                    teacher_results={str(source): teacher}, metric_anchor_results={str(source): anchor})
            sample = manifest["samples"][0]
            self.assertEqual(sample["teacher"]["units"], "relative_inverse_depth")
            self.assertEqual(sample["training_target_choice"], "anchored_teacher")
            self.assertTrue(sample["pseudo_calibration"]["accepted"])
            self.assertFalse(sample["pseudo_calibration"]["anchor_is_measured"])
            preserved = np.load(root / "dataset" / sample["teacher"]["target"]["path"])
            self.assertTrue(arrays_bit_equal(preserved, relative))
            self.assertEqual(sample["anchored_teacher"]["metadata"]["metric_anchor_checkpoint_sha256"], "b" * 64)
            load_dataset(root / "dataset")

    def test_rejected_anchor_preserves_inputs_and_never_silently_selects_anchor_as_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.HEIC"
            source.write_bytes(b"test")
            discovery = fixture_discovery(source)
            teacher = fixture_teacher(discovery, units="relative_inverse_depth")
            anchor = fixture_teacher(discovery)
            yy, xx = np.indices((64, 64), dtype=np.float32)
            anchor.source_depth = np.float32(1) / (np.float32(.2) + xx * np.float32(.0002) + yy * np.float32(.0005))
            anchor.metadata["checkpoint_sha256"] = "b" * 64
            with patch("ipde.dataset.discover_file", return_value=discovery):
                manifest = build_dataset([source], root / "dataset", DatasetOptions(None, metric_anchor=object(), include_display_teacher=False),
                    teacher_results={str(source): teacher}, metric_anchor_results={str(source): anchor})
            sample = manifest["samples"][0]
            self.assertFalse(sample["pseudo_calibration"]["accepted"])
            self.assertIn("metric_anchor", sample)
            self.assertNotIn("anchored_teacher", sample)
            self.assertEqual(sample.get("training_target_choice", "teacher"), "teacher")
            self.assertEqual(sample["teacher"]["units"], "relative_inverse_depth")
            load_dataset(root / "dataset")

    def test_preserves_raw_float_bits_and_teacher_grid_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder, manifest, discoveries = build_fixture(Path(temporary))
            loaded = load_dataset(folder)
            self.assertEqual(manifest, loaded)
            self.assertEqual(manifest["summary"]["train_samples"], 1)
            self.assertEqual(manifest["summary"]["validation_samples"], 1)
            for sample, discovery in zip(loaded["samples"], discoveries):
                raw = np.load(folder / sample["raw_assets"][0]["storage"]["path"], allow_pickle=False)
                self.assertTrue(arrays_bit_equal(raw, discovery.assets[0].array))
                self.assertEqual(sample["teacher"]["label_kind"], "pseudo_label")
                self.assertEqual(sample["teacher"]["target"]["shape"], [64, 64])
                self.assertEqual(sample["calibration"], discovery.spatial_photo)
                flow = np.load(folder / sample["raft_target"]["target"]["path"], allow_pickle=False)
                self.assertEqual(float(flow[1, 3]), -1.5)

    def test_flow_conversion_has_correct_sign_offset_and_does_not_use_presentation_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "photo.HEIC"
            source.write_bytes(b"test")
            discovery = fixture_discovery(source)
            depth = np.full((64, 64), 2.0, np.float32)
            flow, valid, details = teacher_depth_to_flow(depth, discovery.spatial_photo)
            self.assertEqual(float(flow[4, 5]), -1.5)
            self.assertFalse(valid[4, 0])
            self.assertTrue(valid[4, 5])
            self.assertTrue(np.isnan(flow[4, 0]))
            self.assertFalse(details["presentation_disparity_adjustment_applied"])
            self.assertTrue(np.all(depth == 2))

    def test_group_union_handles_duplicates_scene_and_burst_transitively(self) -> None:
        samples = [
            {"source_sha256": "a", "rgb": {"array_sha256": "rgb-a"}, "requested_group": "room"},
            {"source_sha256": "b", "rgb": {"array_sha256": "rgb-b"}, "requested_group": "room", "burst_ids": ["burst"]},
            {"source_sha256": "c", "rgb": {"array_sha256": "rgb-c"}, "burst_ids": ["burst"]},
            {"source_sha256": "d", "rgb": {"array_sha256": "rgb-a"}},
            {"source_sha256": "e", "rgb": {"array_sha256": "other"}},
        ]
        groups = assign_grouped_splits(samples, .2, 42)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len({sample["split"] for sample in samples[:4]}), 1)
        self.assertNotEqual(samples[0]["split"], samples[4]["split"])

    def test_detects_corruption_and_manifest_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder, manifest, _ = build_fixture(Path(temporary))
            target = folder / manifest["samples"][0]["teacher"]["target"]["path"]
            values = np.load(target)
            values[1, 1] += 1
            np.save(target, values, allow_pickle=False)
            with self.assertRaisesRegex(DatasetError, "checksum mismatch"):
                load_dataset(folder)
            manifest["samples"][0]["rgb"]["path"] = "../outside.npy"
            (folder / "dataset.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "escapes"):
                load_dataset(folder)

    def test_failed_provenance_never_publishes_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.HEIC"
            source.write_bytes(b"test")
            discovery = fixture_discovery(source)
            teacher = fixture_teacher(discovery)
            teacher.metadata["input_rgb_sha256"] = "wrong"
            with patch("ipde.dataset.discover_file", return_value=discovery), self.assertRaisesRegex(DatasetError, "provenance"):
                build_dataset([source], root / "dataset", DatasetOptions(None, include_display_teacher=False), teacher_results={str(source): teacher})
            self.assertFalse((root / "dataset").exists())
            self.assertFalse(list(root.glob(".dataset-*")))

    def test_cached_display_teacher_rejects_malformed_units_precision_checkpoint_and_values(self) -> None:
        cases = ("units", "precision", "checkpoint", "values")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "photo.HEIC"
                source.write_bytes(b"test")
                discovery = fixture_discovery(source)
                rgb = discovery.assets[1].array.copy()
                discovery.assets.append(Asset("display_view", 0, 0, rgb, "RGB", 8, "display"))
                display_teacher = fixture_teacher(discovery)
                if case == "units":
                    display_teacher.metadata["units"] = "preview_gray"
                elif case == "precision":
                    display_teacher.native_depth = display_teacher.native_depth.astype(np.float16)
                elif case == "checkpoint":
                    display_teacher.metadata["checkpoint_sha256"] = ""
                else:
                    display_teacher.source_depth[:] = np.nan
                with patch("ipde.dataset.discover_file", return_value=discovery), \
                     self.assertRaisesRegex(DatasetError, "Display teacher"):
                    build_dataset([source], root / "dataset", DatasetOptions(None),
                        teacher_results={str(source): fixture_teacher(discovery)},
                        display_teacher_results={str(source): display_teacher})
                self.assertFalse((root / "dataset").exists())
                self.assertFalse(list(root.glob(".dataset-*")))

    def test_rejects_display_preview_inputs_and_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(DatasetError, "original HEIC"):
                build_dataset([root / "preview.png"], root / "new", DatasetOptions(None))
            folder = root / "existing"
            folder.mkdir()
            (folder / "keep").write_text("user data")
            with self.assertRaisesRegex(DatasetError, "already exists"):
                build_dataset([root / "photo.HEIC"], folder, DatasetOptions(None))
            self.assertEqual((folder / "keep").read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
