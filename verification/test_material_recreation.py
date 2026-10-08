from __future__ import annotations

import hashlib
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import struct
import unittest
import zlib

SCRIPT = Path(__file__).parents[1] / "scripts/material_recreation.py"
spec = importlib.util.spec_from_file_location("material_recreation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecreationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"filename": "test_disp_4k.png", "file_bytes": 123, "file_sha256": "a" * 64,
                       "file_md5": "b" * 32,
                       "width": 4096, "height": 4096, "sample_bits": 16, "channels": 1,
                       "suffix": "disp", "resolution_label": "4k"}
        self.entry = {"asset_id": "test", "filename": self.source["filename"], "published_exact_match": True,
                      "local_bytes": 123, "published_bytes": 123, "published_md5": "B" * 32,
                      "local_md5": "b" * 32, "published_url": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/test/test_disp_4k.png",
                      "path": "/old/location/test_disp_4k.png"}
        self.audit = {"snapshot_completed_utc": "2026-10-07T18:00:00Z", "files": [self.entry]}

    def add_sample(self, identity, split):
        folder = self.root / "samples" / identity
        folder.mkdir(parents=True)
        record = {"sample_id": identity, "material_id": "test", "split": split, "status": "prepared",
                  "crop_rectangle_top_left_xywh": [0 if split == "train" else 2048, 0, 1024, 1024],
                  "map_metadata": {"height": {"source": self.source, "transforms": [], "sample_bits": 16}},
                  "source_notes": []}
        (folder / "sample.json").write_text(json.dumps(record))
        return {key: record[key] for key in ("sample_id", "material_id", "split", "status")} | {"path": "samples/" + identity}

    def fixture(self):
        index = {"schema_version": 2, "samples": [self.add_sample("test_auto_001", "train"), self.add_sample("test_auto_003", "validation")]}
        (self.root / "dataset.json").write_text(json.dumps(index))
        audit = self.root / "audit.json"
        audit.write_text(json.dumps(self.audit))
        return audit

    def test_deduplicates_parent_and_embeds_native_metadata_without_image_copy(self):
        result = module.export_recipe(self.root, self.fixture())
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(len(result["sample_records"]), 2)
        self.assertEqual(result["required_parent_bytes"], 123)
        self.assertTrue(result["fully_redownloadable_from_recorded_urls"])
        self.assertFalse(result["image_payloads_copied"])
        self.assertEqual(result["sample_records"]["test_auto_001"]["metadata"]["map_metadata"]["height"]["sample_bits"], 16)

    def test_unknown_source_never_receives_guessed_url(self):
        self.fixture()
        result = module.export_recipe(self.root)
        self.assertEqual(result["unresolved_source_files"], ["test/test_disp_4k.png"])
        self.assertIsNone(result["sources"]["test/test_disp_4k.png"]["download"])

    def test_case_insensitive_complete_md5_and_changed_files_refused(self):
        self.assertIsNotNone(module.download_evidence(self.source, "test", self.audit))
        for change in ({"local_md5": "c" * 32}, {"published_md5": "Z" * 32}, {"published_bytes": 124}, {"published_exact_match": False}, {"source_stable": False}):
            self.assertIsNone(module.download_evidence(self.source, "test", {"files": [self.entry | change]}))

    def test_same_filename_and_size_cannot_attach_stale_audit(self):
        changed = self.source | {"file_md5": "c" * 32}
        self.assertIsNone(module.download_evidence(changed, "test", self.audit))
        missing = {key: value for key, value in self.source.items() if key != "file_md5"}
        self.assertIsNone(module.download_evidence(missing, "test", self.audit))

    def test_legacy_source_requires_the_audited_current_parent_stat(self):
        parent = self.root / self.source["filename"]
        parent.write_bytes(b"a" * 123)
        state = parent.stat()
        recorded = {"device": state.st_dev, "inode": state.st_ino, "size": state.st_size, "mtime_ns": state.st_mtime_ns}
        source = {key: value for key, value in self.source.items() if key != "file_md5"} | {"path": str(parent)}
        audit = {"files": [self.entry | {"current_stat": recorded}]}
        self.assertIsNotNone(module.download_evidence(source, "test", audit))
        parent.write_bytes(b"b" * 123)
        self.assertIsNone(module.download_evidence(source, "test", audit))

    def package_report(self):
        return {"schema": module.PACKAGE_AUDIT_SCHEMA, "provider": "ambientCG", "material_id": "test",
                "asset_id": "Test", "asset_url": "https://ambientcg.com/view?id=Test",
                "package": {"url": "https://ambientcg.com/get?file=Test_4K-PNG.zip", "filename": "Test_4K-PNG.zip",
                            "file_bytes": 999, "sha256": "d" * 64},
                "license": {"spdx": "CC0-1.0", "url": "https://docs.ambientcg.com/license/", "snapshot_sha256": "e" * 64},
                "creation_method": {"id": "PBRPhotogrammetry"},
                "files": [{"source_filename": self.source["filename"], "source_sha256": "a" * 64,
                           "member_sha256": "a" * 64, "source_bytes": 123, "member_bytes": 123,
                           "exact_full_file_match": True, "archive_member": "Test_4K-PNG_Displacement.png"}]}

    def test_package_keeps_archive_member_and_rename_without_fake_per_file_url(self):
        audit = self.package_report()
        result = module.package_download_evidence(self.source, "test", [audit])
        self.assertEqual(result["download_kind"], "zip_archive_member")
        self.assertEqual(result["archive_member"], "Test_4K-PNG_Displacement.png")
        self.assertEqual(result["restore_filename"], self.source["filename"])
        self.assertEqual(result["url"], audit["package"]["url"])
        self.assertEqual(result["license"]["spdx"], "CC0-1.0")

    def test_package_member_or_parent_mismatch_remains_unresolved(self):
        for change in ({"source_sha256": "f" * 64}, {"member_sha256": "f" * 64},
                       {"source_bytes": 124}, {"member_bytes": 124}, {"exact_full_file_match": False},
                       {"archive_member": "Test_4K-PNG_/../secret.png"}):
            audit = self.package_report()
            audit["files"][0].update(change)
            self.assertIsNone(module.package_download_evidence(self.source, "test", [audit]))

    def test_package_untrusted_url_or_missing_license_proof_remains_unresolved(self):
        for url in ("https://evil.example/get?file=Test_4K-PNG.zip", "http://ambientcg.com/get?file=Test_4K-PNG.zip",
                    "https://ambientcg.com/get?file=Wrong_4K-PNG.zip"):
            audit = self.package_report(); audit["package"]["url"] = url
            self.assertIsNone(module.package_download_evidence(self.source, "test", [audit]))
        audit = self.package_report(); audit["license"].pop("snapshot_sha256")
        self.assertIsNone(module.package_download_evidence(self.source, "test", [audit]))

    def test_export_can_include_verified_package_report(self):
        self.fixture()
        package_path = self.root / "package.json"
        package_path.write_text(json.dumps(self.package_report()))
        result = module.export_recipe(self.root, package_audit_paths=[package_path])
        self.assertTrue(result["fully_redownloadable_from_recorded_urls"])
        self.assertIn("package.json", result["package_audit_sha256"])
        self.assertEqual(module.plan(result)["unique_download_count"], 1)
        self.assertEqual(module.plan(result)["recorded_download_bytes"], 999)

    def test_different_package_audits_cannot_overwrite_provenance_identity(self):
        self.fixture()
        paths = []
        for identity in ("one", "two"):
            folder = self.root / identity; folder.mkdir()
            path = folder / "package.json"
            report = self.package_report() | {"completed_utc": identity}
            path.write_text(json.dumps(report)); paths.append(path)
        with self.assertRaisesRegex(ValueError, "share a filename"):
            module.export_recipe(self.root, package_audit_paths=paths)

    def test_untrusted_download_host_not_recorded(self):
        for url in ("https://dl.polyhaven.org.evil.test/file/ph-assets/Textures/test_disp_4k.png", "http://dl.polyhaven.org/file/ph-assets/Textures/test_disp_4k.png"):
            self.assertIsNone(module.download_evidence(self.source, "test", {"files": [self.entry | {"published_url": url}]}))

    def test_index_metadata_disagreement_refused(self):
        self.fixture()
        p = self.root / "samples/test_auto_001/sample.json"
        d = json.loads(p.read_text()); d["split"] = "validation"; p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "disagreement"):
            module.export_recipe(self.root)

    def test_path_escape_refused(self):
        with self.assertRaises(ValueError):
            module.relative_path(self.root, "../outside/sample.json")

    def test_source_note_kept_exact_and_changed_note_refused(self):
        self.fixture()
        p = self.root / "samples/test_auto_001/sample.json"
        d = json.loads(p.read_text()); payload = b"Original scale note\n"
        note = p.parent / "note.txt"; note.write_bytes(payload)
        d["source_notes"] = [{"path": "note.txt", "sha256": hashlib.sha256(payload).hexdigest()}]
        p.write_text(json.dumps(d))
        result = module.export_recipe(self.root)
        self.assertEqual(result["source_notes"]["test/note.txt"]["sha256"], hashlib.sha256(payload).hexdigest())
        note.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed source note"):
            module.export_recipe(self.root)

    def duplicate_note_fixture(self):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
        png = module.PNG_SIGNATURE + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 1, 16, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\0\x7f\xff\x80\x00')) + chunk(b'IEND', b'')
        parent = self.root / 'originals' / self.source['filename']
        parent.parent.mkdir(); parent.write_bytes(png)
        checksum, md5 = module.sha256(png), hashlib.md5(png).hexdigest()
        self.source.update(path=str(parent), file_bytes=len(png), file_sha256=checksum, file_md5=md5, width=2, height=1)
        self.entry.update(local_bytes=len(png), published_bytes=len(png), local_md5=md5, published_md5=md5)
        audit = self.fixture()
        path = self.root / 'samples/test_auto_001/sample.json'
        record = json.loads(path.read_text())
        note = path.parent / 'test_disp_4k.txt'; note.write_bytes(png)
        record['source_notes'] = [{'path': note.name, 'sha256': checksum}]
        path.write_text(json.dumps(record))
        return audit, parent, note, png

    def test_exact_png_duplicate_note_is_parent_reference_without_payload(self):
        audit, parent, note, payload = self.duplicate_note_fixture()
        original_metadata = (note.parent / 'sample.json').read_bytes()
        recipe = module.export_recipe(self.root, audit)
        entry = recipe['source_notes']['test/test_disp_4k.txt']
        self.assertNotIn('data_base64', entry)
        self.assertEqual(entry['parent_reference'], {'source_key': 'test/test_disp_4k.png',
            'sha256': module.sha256(payload), 'file_bytes': len(payload)})
        self.assertNotIn(base64.b64encode(payload).decode(), json.dumps(recipe['source_notes']))
        self.assertEqual(module.plan(recipe)['parent_reference_source_note_count'], 1)
        self.assertEqual(module.plan(recipe)['parent_reference_source_note_bytes'], len(payload))
        self.assertEqual(module.plan(recipe)['embedded_source_note_bytes'], 0)
        self.assertEqual(parent.read_bytes(), payload)
        self.assertEqual(note.read_bytes(), payload)
        self.assertEqual((note.parent / 'sample.json').read_bytes(), original_metadata)

    def test_changed_matching_recorded_parent_refused_and_missing_parent_embedded(self):
        audit, parent, note, payload = self.duplicate_note_fixture()
        parent.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'Changed parent'):
            module.export_recipe(self.root, audit)
        parent.unlink()
        recipe = module.export_recipe(self.root, audit)
        entry = recipe['source_notes']['test/test_disp_4k.txt']
        self.assertNotIn('parent_reference', entry)
        self.assertEqual(base64.b64decode(entry['data_base64']), payload)
        self.assertEqual(note.read_bytes(), payload)

    def test_unique_png_note_and_legacy_text_note_stay_embedded(self):
        audit, parent, note, payload = self.duplicate_note_fixture()
        unique = payload + b'Unique note content'
        note.write_bytes(unique)
        metadata_path = note.parent / 'sample.json'
        record = json.loads(metadata_path.read_text()); record['source_notes'][0]['sha256'] = module.sha256(unique)
        metadata_path.write_text(json.dumps(record))
        recipe = module.export_recipe(self.root, audit)
        self.assertEqual(base64.b64decode(recipe['source_notes']['test/test_disp_4k.txt']['data_base64']), unique)
        self.assertTrue(recipe['image_payloads_copied'])
        self.assertEqual(recipe['embedded_png_source_note_count'], 1)
        self.assertIn('image bytes', recipe['scope'])
        self.assertTrue(module.plan(recipe)['image_payloads_copied'])
        legacy_png = deepcopy(recipe)
        legacy_png['source_notes'] = {'test/old-png.txt': {'sha256': module.sha256(payload), 'data_base64': base64.b64encode(payload).decode()}}
        legacy_png['image_payloads_copied'] = False  # Historical recipe flag remains untouched.
        self.assertTrue(module.plan(legacy_png)['image_payloads_copied'])
        legacy = b'Original old recipe note\n'
        recipe['source_notes'] = {'test/old-note.txt': {'sha256': module.sha256(legacy), 'data_base64': base64.b64encode(legacy).decode()}}
        self.assertEqual(module.plan(recipe)['embedded_source_note_bytes'], len(legacy))
        destination = self.root / 'legacy-restore'
        module.restore_source_notes(recipe, destination)
        self.assertEqual((destination / 'test/old-note.txt').read_bytes(), legacy)
        self.assertEqual(parent.read_bytes(), payload)

    def test_exact_restoration_is_idempotent_and_never_overwrites_parent_or_note(self):
        audit, parent, note, payload = self.duplicate_note_fixture()
        recipe = module.export_recipe(self.root, audit)
        destination = self.root / 'restored'; available = destination / 'test/test_disp_4k.png'
        available.parent.mkdir(parents=True); available.write_bytes(payload)
        result = module.restore_source_notes(recipe, destination)
        self.assertEqual(result['restored_source_notes'], ['test/test_disp_4k.txt'])
        self.assertEqual((destination / 'test/test_disp_4k.txt').read_bytes(), payload)
        self.assertEqual(available.read_bytes(), payload)
        self.assertEqual(module.restore_source_notes(recipe, destination)['already_present_source_notes'], ['test/test_disp_4k.txt'])
        (destination / 'test/test_disp_4k.txt').write_bytes(b'Different existing note')
        with self.assertRaisesRegex(ValueError, 'no files overwritten'):
            module.restore_source_notes(recipe, destination)
        self.assertEqual((destination / 'test/test_disp_4k.txt').read_bytes(), b'Different existing note')
        self.assertEqual(available.read_bytes(), payload)
        self.assertFalse(list(destination.rglob('.restore-source-note-*')))

    def test_changed_available_parent_rejects_all_note_writes(self):
        audit, _, _, payload = self.duplicate_note_fixture()
        recipe = module.export_recipe(self.root, audit)
        text = b'Unique text'
        recipe['source_notes'] = {'test/text.txt': {'sha256': module.sha256(text), 'data_base64': base64.b64encode(text).decode()}, **recipe['source_notes']}
        destination = self.root / 'restored'; parent = destination / 'test/test_disp_4k.png'
        parent.parent.mkdir(parents=True); parent.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        before = parent.read_bytes()
        with self.assertRaisesRegex(ValueError, 'parent changed'):
            module.restore_source_notes(recipe, destination)
        self.assertFalse((parent.parent / 'text.txt').exists())
        self.assertFalse((parent.parent / 'test_disp_4k.txt').exists())
        self.assertEqual(parent.read_bytes(), before)

    def test_malformed_unknown_or_mismatched_parent_references_fail_validation(self):
        audit, _, _, _ = self.duplicate_note_fixture()
        recipe = module.export_recipe(self.root, audit)
        for change in ({'source_key': 'test/unknown.png'}, {'sha256': 'f' * 64},
                       {'file_bytes': 1}, {'file_bytes': True}, {'source_key': '../outside.png'},
                       {'unexpected': 'extra'}):
            with self.subTest(change=change):
                altered = deepcopy(recipe)
                altered['source_notes']['test/test_disp_4k.txt']['parent_reference'].update(change)
                with self.assertRaises(ValueError): module.plan(altered)
                with self.assertRaises(ValueError): module.restore_source_notes(altered, self.root / 'never-created')
        for change in ({'data_base64': ''}, {'file_bytes': False}, {'parent_reference': None}):
            with self.subTest(change=change):
                altered = deepcopy(recipe); altered['source_notes']['test/test_disp_4k.txt'].update(change)
                with self.assertRaises(ValueError): module.plan(altered)
        altered = deepcopy(recipe); altered['sources']['test/test_disp_4k.png']['source']['file_sha256'] = 'f' * 64
        with self.assertRaises(ValueError): module.plan(altered)
        for bad_source in ('not a source dictionary', {'source': 'not source metadata'}):
            altered = deepcopy(recipe); altered['sources']['test/test_disp_4k.png'] = bad_source
            with self.assertRaises(ValueError): module.plan(altered)
        for bad_key in ('test/../test_disp_4k.png', 'test/nested/test_disp_4k.png', './test/test_disp_4k.png', 'test//test_disp_4k.png', 'test/\0bad.png'):
            altered = deepcopy(recipe)
            altered['sources'][bad_key] = altered['sources'].pop('test/test_disp_4k.png')
            altered['source_notes']['test/test_disp_4k.txt']['parent_reference']['source_key'] = bad_key
            with self.assertRaises(ValueError): module.plan(altered)
        self.assertFalse((self.root / 'never-created').exists())
