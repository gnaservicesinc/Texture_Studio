"""CPU-only original-byte, bounded-download and provenance contract tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_material_resolution as download


def png_bytes(width=2048, height=1024, bits=16, color=0):
    # These fixtures exercise the promised IHDR verification, not pixel decode.
    payload = struct.pack(">IIBBBBB", width, height, bits, color, 0, 0, 0)
    chunk = b"IHDR" + payload
    return download.PNG_SIGNATURE + struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xffffffff)


def api_payload(material="example"):
    result = {}
    for role, provider, suffix, color in (("input", "Diffuse", "diff", 2), ("height", "Displacement", "disp", 0), ("normal", "nor_gl", "nor_gl", 2), ("roughness", "Rough", "rough", 0)):
        data = png_bytes(color=color)
        result[provider] = {"2k": {"png": {"url": f"https://dl.polyhaven.org/file/ph-assets/Textures/png/2k/{material}/{material}_{suffix}_2k.png", "size": len(data), "md5": hashlib.md5(data).hexdigest().upper()}}}
    return result


class MaterialResolutionDownloadTests(unittest.TestCase):
    def test_metadata_published_atomically_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "provenance.json"
            self.assertTrue(download.publish_bytes_once(path, b'{"complete": true}'))
            self.assertFalse(download.publish_bytes_once(path, b"changed"))
            self.assertEqual(path.read_bytes(), b'{"complete": true}')
            with patch.object(download.os, "link", side_effect=PermissionError("simulated publish failure")):
                with self.assertRaises(PermissionError):
                    download.publish_bytes_once(root / "failed.json", b"complete data")
            self.assertFalse((root / "failed.json").exists())
            self.assertEqual(list(root.glob("*.metadata-tmp")), [])

    def test_plan_selects_png_originals_and_opengl_normal(self):
        payload = api_payload()
        payload["nor_dx"] = payload["nor_gl"]
        plan = download.make_plan("example", payload, {"mode": "fresh"}, {}, "2k")
        self.assertEqual(plan["maps"]["normal"]["provider_map_name"], "nor_gl")
        self.assertEqual(plan["published_bytes"], 4 * 33)
        self.assertFalse(plan["resized_from_existing_4k"])
        self.assertFalse(plan["cross_resolution_registration_verified"])
        self.assertFalse(plan["training_split_assigned"])
        self.assertEqual(plan["license"], "CC0-1.0")
        self.assertEqual(plan["maps"]["height"]["published_md5"], payload["Displacement"]["2k"]["png"]["md5"].lower())

    def test_jpeg_is_never_substituted_for_missing_png(self):
        payload = api_payload()
        payload["Displacement"]["2k"]["jpg"] = payload["Displacement"]["2k"].pop("png")
        with self.assertRaisesRegex(ValueError, "Missing published"):
            download.make_plan("example", payload, {}, {}, "2k")

    def test_unofficial_host_and_unsafe_filename_rejected(self):
        for url in ("https://evil.example/file/ph-assets/Textures/png/2k/example/example_disp_2k.png", "http://dl.polyhaven.org/file/ph-assets/Textures/png/2k/example/example_disp_2k.png", "https://dl.polyhaven.org/file/ph-assets/Textures/png/2k/example/../example_disp_2k.png", "https://dl.polyhaven.org/file/ph-assets/Textures/png/2k/example/example_disp_2k.png?redirect=evil"):
            payload = api_payload()
            payload["Displacement"]["2k"]["png"]["url"] = url
            with self.assertRaises(ValueError):
                download.selected_map("example", "height", payload, "2k")

    def test_header_crc_dimensions_and_bitdepth_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.png"
            path.write_bytes(png_bytes())
            self.assertEqual(download.png_header(path)["sample_bits"], 16)
            self.assertEqual(download.png_header(path)["height"], 1024)
            changed = bytearray(path.read_bytes())
            changed[20] ^= 1
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "IHDR"):
                download.png_header(path)

    def test_uint8_height_rejected_without_upconversion(self):
        data = png_bytes(bits=8)
        entry = {"role": "height", "published_bytes": len(data), "published_md5": hashlib.md5(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.png"
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "8-bit"):
                download.verify_map(path, entry, "2k")
            self.assertEqual(path.read_bytes(), data)
            entry["role"] = "input"
            self.assertEqual(download.verify_map(path, entry, "2k")["png_header"]["sample_bits"], 8)

    def test_published_uint8_normal_and_roughness_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            for role, color in (("normal", 2), ("roughness", 0)):
                data = png_bytes(bits=8, color=color)
                path = Path(directory) / f"{role}.png"
                path.write_bytes(data)
                entry = {"role": role, "published_bytes": len(data), "published_md5": hashlib.md5(data).hexdigest()}
                result = download.verify_map(path, entry, "2k")
                self.assertEqual(result["png_header"]["sample_bits"], 8)
                self.assertTrue(result["original_precision_preserved"])
                self.assertIn("without upconversion", result["source_precision_note"])
                self.assertEqual(path.read_bytes(), data)

    def test_existing_changed_file_is_not_overwritten(self):
        plan = download.make_plan("example", api_payload(), {}, {}, "2k")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "example"
            destination.mkdir()
            path = destination / plan["maps"]["input"]["filename"]
            path.write_bytes(b"existing unrelated image")
            with self.assertRaisesRegex(ValueError, "left untouched"), patch.object(download.subprocess, "run") as run:
                download.download_asset(plan, root)
            run.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing unrelated image")

    def test_complete_pair_download_and_idempotent_reuse(self):
        plan = download.make_plan("example", api_payload(), {}, {}, "2k")
        def curl(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            color = 2 if "_diff_" in command[-1] or "_nor_gl_" in command[-1] else 0
            output.write_bytes(png_bytes(color=color))
            return subprocess.CompletedProcess(command, 0, b"", b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(download.subprocess, "run", side_effect=curl) as run:
            root = Path(directory)
            first = download.download_asset(plan, root)
            self.assertEqual(first["actual_native_pixel_dimensions"], [2048, 1024])
            self.assertTrue(first["actual_headers_verified"])
            self.assertEqual(run.call_count, 4)
            manifest = root / "example/material-source.json"
            before = manifest.read_bytes()
            second = download.download_asset(plan, root)
            self.assertEqual(run.call_count, 4)
            self.assertTrue(second["existing_manifest_reused"])
            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(list(root.rglob("*.download")), [])

    def test_mismatched_pair_dimensions_publish_no_new_maps(self):
        payload = api_payload()
        data = png_bytes(height=2048, color=2)
        payload["nor_gl"]["2k"]["png"].update(size=len(data), md5=hashlib.md5(data).hexdigest())
        plan = download.make_plan("example", payload, {}, {}, "2k")
        def curl(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            if "_nor_gl_" in command[-1]:
                output.write_bytes(data)
            else:
                output.write_bytes(png_bytes(color=2 if "_diff_" in command[-1] else 0))
            return subprocess.CompletedProcess(command, 0, b"", b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(download.subprocess, "run", side_effect=curl):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                download.download_asset(plan, root)
            self.assertEqual(list(root.rglob("*.png")), [])
            self.assertEqual(list(root.rglob("*.download")), [])

    def test_partial_failed_download_removes_only_owned_temps(self):
        plan = download.make_plan("example", api_payload(), {}, {}, "2k")
        def curl(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"partial")
            return subprocess.CompletedProcess(command, 28, b"", b"timeout")
        with tempfile.TemporaryDirectory() as directory, patch.object(download.subprocess, "run", side_effect=curl):
            root = Path(directory)
            unrelated = root / "keep.txt"
            unrelated.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "download failed"):
                download.download_asset(plan, root)
            self.assertEqual(list(root.rglob("*.download")), [])
            self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_disk_budget_prevents_any_png_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(download, "fetch_metadata", return_value=(api_payload(), {})), patch.object(download, "download_asset") as hydrate:
            root = Path(directory)
            result = download.main(["--materials", "example", "--download", "--max-bytes", "1", "--destination", str(root / "sources"), "--report", str(root / "report.json")])
            self.assertEqual(result, 1)
            hydrate.assert_not_called()
            self.assertFalse((root / "sources").exists())
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["total_published_bytes"], 132)
            self.assertFalse(report["within_requested_byte_budget"])


if __name__ == "__main__":
    unittest.main()
