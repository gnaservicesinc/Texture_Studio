"""Published-byte audit regressions; no network or user data accesses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_material_sources as audit


def png(value: int) -> bytes:
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    return audit.PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\0" + struct.pack(">H", value))) + chunk(b"IEND", b"")


def metadata(path: Path, data: bytes):
    return {"provenance": {"mode": "cached", "api_url": "https://api.polyhaven.com/files/white_stucco_02"},
            "files": {"Displacement": {"4k": {"png": {"url": "https://dl.polyhaven.org/" + path.name,
                                                           "md5": hashlib.md5(data).hexdigest(), "size": len(data)}}}}}


class MaterialSourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.sources = self.base / "sources"
        folder = self.sources / "White Stucco 02"
        folder.mkdir(parents=True)
        self.path = folder / "white_stucco_02_disp_4k.png"
        self.data = png(12345)
        self.path.write_bytes(self.data)
        self.item = audit.snapshot_sources(self.sources)[0][0]

    def tearDown(self):
        self.temporary.cleanup()

    def test_disp_gl_alias_is_height_without_claiming_published_provenance(self):
        match = audit.PARENT.fullmatch("Wicker013_disp_gl_4k.png")
        self.assertIsNotNone(match)
        self.assertEqual(match["asset"], "Wicker013")
        self.assertEqual(match["role"], "disp_gl")
        self.assertEqual(audit.STANDARD[match["role"]], "height")
        self.assertEqual(match["resolution"], "4k")

    def test_matching_full_bytes_and_renamed_folder_retain_original_inode(self):
        target = self.sources / "white_stucco_02"
        self.path.parent.rename(target)
        result = audit.inspect_source(self.sources, self.item, metadata(self.path, self.data))
        self.assertEqual(result["status"], "matched")
        self.assertTrue(result["path_renamed"])
        self.assertTrue(result["source_stable"])
        self.assertEqual(Path(result["path"]), target / self.path.name)

    def test_same_name_different_numeric_png_is_a_mismatch(self):
        expected = png(23456)
        result = audit.inspect_source(self.sources, self.item, metadata(self.path, expected))
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["published_exact_match"])
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_changed_source_after_snapshot_never_claims_original_bytes(self):
        self.path.write_bytes(png(34567))
        result = audit.inspect_source(self.sources, self.item, metadata(self.path, self.data))
        self.assertEqual(result["status"], "source_changed")
        self.assertFalse(result["source_stable"])
        self.assertFalse(result["published_exact_match"])

    def test_renamed_folder_then_replaced_file_is_reported_changed(self):
        target = self.sources / "white_stucco_02"
        self.path.parent.rename(target)
        replacement = target / "replacement.png"
        replacement.write_bytes(self.data)
        replacement.replace(target / self.path.name)
        result = audit.inspect_source(self.sources, self.item, metadata(self.path, self.data))
        self.assertEqual(result["status"], "source_changed")
        self.assertFalse(result["published_exact_match"])

    def test_changed_source_during_hash_never_claims_original_bytes(self):
        original = audit.stream_md5
        def changed(stream):
            digest = original(stream)
            self.path.write_bytes(png(45678))
            return digest
        with patch.object(audit, "stream_md5", changed):
            result = audit.inspect_source(self.sources, self.item, metadata(self.path, self.data))
        self.assertEqual(result["status"], "source_changed")
        self.assertFalse(result["published_exact_match"])

    def test_existing_api_cache_is_labeled_cached_without_inventing_fetch_time(self):
        cache = self.base / "api"
        cache.mkdir()
        (cache / "white_stucco_02.json").write_text(json.dumps(metadata(self.path, self.data)["files"]))
        with patch.object(audit.subprocess, "run", side_effect=AssertionError("network not allowed")):
            result = audit.api_metadata("white_stucco_02", cache)
        self.assertEqual(result["provenance"]["mode"], "cached")
        self.assertFalse(result["provenance"]["fetch_time_known"])
        self.assertIsNone(result["provenance"]["fetched_utc"])

    def test_fresh_api_fetch_records_provenance_then_reuses_cached_json(self):
        cache = self.base / "api"
        payload = metadata(self.path, self.data)["files"]
        def downloaded(command, **_kwargs):
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(payload))
            return audit.subprocess.CompletedProcess(command, 0, "200", "")
        with patch.object(audit.subprocess, "run", side_effect=downloaded):
            fresh = audit.api_metadata("white_stucco_02", cache)
        self.assertEqual(fresh["provenance"]["mode"], "fresh")
        self.assertTrue(fresh["provenance"]["fetch_time_known"])
        with patch.object(audit.subprocess, "run", side_effect=AssertionError("network not allowed")):
            cached = audit.api_metadata("white_stucco_02", cache)
        self.assertEqual(cached["provenance"]["mode"], "cached")
        self.assertEqual(cached["provenance"]["fetched_utc"], fresh["provenance"]["fetched_utc"])

    def test_short_published_md5_is_rejected_without_padding(self):
        published = metadata(self.path, self.data)
        published["files"]["Displacement"]["4k"]["png"]["md5"] = "1" * 31
        result = audit.inspect_source(self.sources, self.item, published)
        self.assertEqual(result["status"], "published_metadata_invalid")
        self.assertFalse(result["published_exact_match"])

    def test_full_audit_reports_incomplete_maps_without_claiming_training_readiness(self):
        cache = self.base / "api"
        cache.mkdir()
        (cache / "white_stucco_02.json").write_text(json.dumps(metadata(self.path, self.data)["files"]))
        with patch.object(audit.subprocess, "run", side_effect=AssertionError("network not allowed")):
            result = audit.audit(self.sources, cache, api_workers=1, file_workers=1)
        self.assertTrue(result["all_snapshotted_parent_pngs_match"])
        self.assertFalse(result["materials"][0]["complete_standard_map_set"])
        self.assertEqual(result["materials"][0]["missing_required_maps"], ["input", "normal", "roughness"])
        self.assertFalse(result["source_collection_changed_during_audit"])


if __name__ == "__main__":
    unittest.main()
