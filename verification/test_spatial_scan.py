"""Folder discovery feedback and unchanged calibrated photo admission rules."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ipde.extractor import Asset, Discovery
from ipde.spatial_scan import SpatialScanError, scan_spatial_directory


def calibrated_capture(path: Path) -> Discovery:
    left = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    right = left + 1
    camera = {"height": 3, "width": 4, "focal_length_x_pixels": 20.0}
    spatial = {"raft_stereo_ready": True, "left_image_index": 0, "right_image_index": 1,
               "left_camera": camera, "right_camera": camera,
               "baseline_meters": 0.02, "principal_point_delta_x_pixels": 0.0}
    assets = [Asset("spatial_view", index, 0, array, "RGB", 8, role)
              for index, array, role in ((0, left, "spatial_left"), (1, right, "spatial_right"))]
    return Discovery(path, 7, "source-hash", "image/heic", 0,
                     [{"make": "Apple", "model": "iPhone 16 Pro Max"}], assets, spatial)


class SpatialScanTests(unittest.TestCase):
    def test_recursive_mixed_case_and_hif_stream_before_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            child = root / "nested"; child.mkdir()
            first, nested = root / "FIRST.HeIc", child / "nested.HIF"
            first.write_bytes(b"capture"); nested.write_bytes(b"capture")
            (root / "text.jpg").write_bytes(b"ignored")
            events = []
            snapshots = []

            def discover(path):
                snapshots.append((path, [event["event"] for event in events]))
                return calibrated_capture(path)

            with patch("ipde.spatial_scan.discover_file", side_effect=discover) as decoder:
                report = scan_spatial_directory(root, progress_callback=events.append)
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual([Path(record["source_path"]) for record in report["accepted"]], [first, nested])
            self.assertEqual(events[0]["event"], "scan_started")
            self.assertEqual(snapshots[0][1][-1], "scan_photo_started")
            self.assertIn("spatial_photo_found", snapshots[1][1])
            self.assertEqual(report["summary"]["candidates"], 2)
            self.assertEqual(report["summary"]["visited_directories"], 2)
            self.assertEqual(report["accepted"][0]["photo_metadata"]["camera_model"], "iPhone 16 Pro Max")

    def test_nonrecursive_and_empty_folder_report_no_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            child = root / "nested"; child.mkdir()
            (child / "capture.heic").write_bytes(b"capture")
            with patch("ipde.spatial_scan.discover_file") as decoder:
                report = scan_spatial_directory(root, recursive=False)
            decoder.assert_not_called()
            self.assertEqual(report["summary"]["candidates"], 0)
            self.assertEqual(report["accepted"], [])

    def test_bad_capture_is_reported_and_does_not_hide_valid_following_photo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            bad, good = root / "a-bad.heic", root / "b-good.heic"
            bad.write_bytes(b"bad"); good.write_bytes(b"good")
            events = []

            def discover(path):
                if path == bad:
                    raise ValueError("malformed HEIF")
                return calibrated_capture(path)

            with patch("ipde.spatial_scan.discover_file", side_effect=discover):
                report = scan_spatial_directory(root, progress_callback=events.append)
            self.assertEqual(report["summary"]["accepted"], 1)
            self.assertEqual(report["summary"]["skipped"], 1)
            self.assertEqual(report["summary"]["candidates"], 2)
            skipped = next(event for event in events if event["event"] == "photo_skipped")
            self.assertEqual(skipped["reason"], "malformed HEIF")

    def test_unreadable_root_fails_with_selected_folder_instead_of_empty_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch("ipde.spatial_scan.os.scandir", side_effect=PermissionError("Access denied")):
                with self.assertRaisesRegex(SpatialScanError, "Could not read scan directory.*Access denied"):
                    scan_spatial_directory(root)

    def test_unreadable_subfolder_is_reported_without_losing_root_photo(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            child = root / "unreadable"; child.mkdir()
            good = root / "good.heic"; good.write_bytes(b"capture")
            scandir = os.scandir

            def read_directory(path):
                if path == child:
                    raise PermissionError("Access denied")
                return scandir(path)

            with patch("ipde.spatial_scan.os.scandir", side_effect=read_directory), \
                    patch("ipde.spatial_scan.discover_file", side_effect=calibrated_capture):
                report = scan_spatial_directory(root)
            self.assertEqual(report["summary"]["accepted"], 1)
            self.assertEqual(report["skipped"][0]["kind"], "directory_error")

    def test_symbolic_links_stay_excluded_and_are_visible_in_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            good = root / "good.heic"; good.write_bytes(b"capture")
            (root / "linked.heic").symlink_to(good)
            (root / "linked-dir").symlink_to(root, target_is_directory=True)
            events = []
            with patch("ipde.spatial_scan.discover_file", side_effect=calibrated_capture) as decoder:
                report = scan_spatial_directory(root, progress_callback=events.append)
            decoder.assert_called_once_with(good)
            self.assertEqual(report["summary"]["skipped"], 2)
            self.assertEqual(len([event for event in events if event["event"] == "photo_skipped"]), 2)
            with self.assertRaisesRegex(SpatialScanError, "symbolic link"):
                scan_spatial_directory(root / "linked-dir")


if __name__ == "__main__":
    unittest.main()
