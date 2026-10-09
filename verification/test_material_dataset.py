"""Numerical and filesystem contracts for native material dataset preparation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock
import zlib

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "material_dataset.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("material_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def independent_png(path: Path, values: np.ndarray, gamma: float | None = None, compression_level: int = 6):
    """Construct an unfiltered PNG without the production decoder/encoder."""
    bits = values.dtype.itemsize * 8
    height, width, channels = values.shape
    color = {1: 0, 3: 2, 4: 6}[channels]
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, bits, color, 0, 0, 0)
    rows = b"".join(b"\0" + row.astype(values.dtype.newbyteorder(">"), copy=False).tobytes() for row in values)
    png = MODULE.PNG_SIGNATURE + chunk(b"IHDR", header)
    if gamma is not None:
        png += chunk(b"gAMA", struct.pack(">I", round(gamma * 100000)))
    png += chunk(b"IDAT", zlib.compress(rows, level=compression_level)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


class PNGPrecisionTests(unittest.TestCase):
    def test_fast_header_checks_actual_dimensions_and_ihdr_crc_without_pixel_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "normal.png"
            independent_png(source, np.zeros((16, 32, 3), dtype=np.uint16))
            with mock.patch.object(MODULE.cv2, "imdecode", side_effect=AssertionError("Header inspection must not decode pixels")):
                self.assertEqual(MODULE.png_image_header(source),
                                 {"width": 32, "height": 16, "sample_bits": 16, "channels": 3})
            corrupted = bytearray(source.read_bytes())
            corrupted[19] ^= 1
            source.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "IHDR CRC"):
                MODULE.png_image_header(source)

    def test_adjacent_rgb16_and_scalar_codes_survive_decode_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            # Adjacent low bits and deliberately different RGB values expose RGB8
            # truncation, endian swapping and BGR-order mistakes independently.
            rgb = np.array([[[0, 1, 2], [32767, 32768, 32769]], [[65533, 65534, 65535], [259, 260, 261]]], dtype=np.uint16)
            for name, values in (("rgb", rgb), ("gray", rgb[:, :, 1:2])):
                source = base / (name + ".png")
                independent_png(source, values)
                decoded, metadata = MODULE.read_png(source)
                self.assertEqual(decoded.dtype, np.uint16)
                np.testing.assert_array_equal(decoded, values)
                target = base / (name + "-roundtrip.png")
                MODULE.write_png(target, decoded, metadata)
                actual, actual_metadata = MODULE.read_png(target)
                self.assertEqual(actual_metadata["sample_bits"], 16)
                np.testing.assert_array_equal(actual, values)

    def test_color_gamma_tag_does_not_change_numeric_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tagged.png"
            values = np.array([[[0], [32768], [65535]]], dtype=np.uint16)
            independent_png(source, values, .45455)
            actual, metadata = MODULE.read_png(source)
            np.testing.assert_array_equal(actual, values)
            self.assertEqual(metadata["png_gamma"], .45455)

    def test_crc_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.png"
            independent_png(source, np.zeros((2, 2, 1), dtype=np.uint16))
            encoded = bytearray(source.read_bytes())
            encoded[30] ^= 1
            source.write_bytes(encoded)
            with self.assertRaisesRegex(ValueError, "CRC"):
                MODULE.read_png(source)


class SourceCacheTests(unittest.TestCase):
    def test_cached_parent_change_is_invalidated_even_when_mtime_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.png"
            values = np.arange(256, dtype=np.uint16).reshape(16, 16, 1)
            independent_png(path, values, compression_level=0)
            cache = MODULE.SourceDecodeCache(100000)
            first, metadata = cache.read(path)
            first_state = path.stat()
            cached, same_metadata = cache.read(path)
            self.assertIs(first, cached)
            self.assertEqual(cache.hits, 1)
            independent_png(path, values + 1, compression_level=0)
            os.utime(path, ns=(first_state.st_atime_ns, first_state.st_mtime_ns))
            self.assertEqual(path.stat().st_size, first_state.st_size)
            self.assertEqual(path.stat().st_mtime_ns, first_state.st_mtime_ns)
            self.assertNotEqual(path.stat().st_ctime_ns, first_state.st_ctime_ns)
            actual, changed_metadata = cache.read(path)
            np.testing.assert_array_equal(actual, values + 1)
            self.assertNotEqual(metadata["file_sha256"], changed_metadata["file_sha256"])
            self.assertEqual(cache.misses, 2)
            self.assertEqual(len(cache.entries), 1)
            self.assertGreaterEqual(cache.evictions, 1)

    def test_cache_eviction_is_bounded_and_oversized_sources_are_not_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"parent-{index}.png" for index in range(3)]
            for index, path in enumerate(paths):
                independent_png(path, np.full((32, 32, 3), index, dtype=np.uint16))
            unlimited = MODULE.SourceDecodeCache(100000)
            unlimited.read(paths[0])
            one_entry_bytes = unlimited.retained_bytes
            cache = MODULE.SourceDecodeCache(one_entry_bytes + 100)
            for path in paths:
                cache.read(path)
                self.assertLessEqual(cache.retained_bytes, cache.max_bytes)
                self.assertEqual(len(cache.entries), 1)
            self.assertEqual(cache.evictions, 2)
            self.assertLessEqual(cache.peak_retained_bytes, cache.max_bytes)
            cache.read(paths[0])
            self.assertEqual(cache.misses, 4)
            tiny = MODULE.SourceDecodeCache(1)
            actual, _ = tiny.read(paths[0])
            self.assertEqual(actual.shape, (32, 32, 3))
            self.assertEqual(tiny.retained_bytes, 0)
            self.assertFalse(tiny.entries)

    def test_source_changing_during_decode_is_rejected_and_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.png"
            values = np.zeros((4, 4, 1), dtype=np.uint16)
            independent_png(path, values)
            cache = MODULE.SourceDecodeCache(100000)
            original_decoder = MODULE.read_png
            def mutate_after_decode(filename):
                result = original_decoder(filename)
                independent_png(path, values + 1)
                return result
            with mock.patch.object(MODULE, "read_png", side_effect=mutate_after_decode):
                with self.assertRaisesRegex(ValueError, "changed while decoding"):
                    cache.read(path)
            self.assertEqual(cache.retained_bytes, 0)
            self.assertFalse(cache.entries)


class MaterialPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.output = self.root / "dataset"
        self.sources.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def material(self, name="Test Surface", identity="test_surface", shape=(4, 8), dx=False, mismatch=False):
        folder = self.sources / name
        folder.mkdir()
        height, width = shape
        scalar = np.arange(height * width, dtype=np.uint16).reshape(height, width, 1) * 2000 + 17
        rgb = np.concatenate([scalar, scalar + 1, scalar + 2], axis=2)
        maps = {"diff": rgb, "disp": scalar, "nor_dx" if dx else "nor_gl": rgb, "rough": scalar,
                "anisotropy_rotation": scalar}
        for suffix, values in maps.items():
            if suffix == "rough" and mismatch:
                values = values[:-1]
            independent_png(folder / f"{identity}_{suffix}_4k.png", values, .45455)
        (folder / "material-notes.txt").write_text("Unique source note\n")
        # Existing handmade crop in the source folder must not become a parent.
        independent_png(folder / f"{identity}_001_diff.png", rgb[:2, :2])
        return folder, maps

    def prepare(self, **kwargs):
        materials = MODULE.discover_materials(self.sources, 2, kwargs.pop("dimension_policy", "strict"))
        self.assertEqual(len(materials), 1)
        self.assertTrue(materials[0]["ready"], materials[0]["problems"])
        return materials[0], MODULE.prepare_material(materials[0], self.output, kwargs.pop("overrides", {}), True, .2)

    def test_native_two_corner_crops_preserve_all_maps_and_source_notes(self):
        folder, maps = self.material()
        original_hashes = {path.name: MODULE.file_sha256(path) for path in folder.iterdir()}
        material, records = self.prepare()
        self.assertEqual(material["crop_rectangles_top_left_xywh"], [[0, 0, 2, 2], [6, 2, 2, 2]])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["split"], records[1]["split"])
        self.assertEqual(records[0]["status"], "approved")
        self.assertTrue(records[0]["encoding_warnings"])
        for record in records:
            sample = self.output / "samples" / record["sample_id"]
            self.assertEqual(MODULE.verify_sample(sample, check_sources=True), [])
            x, y, width, height = record["crop_rectangle_top_left_xywh"]
            actual, header = MODULE.read_png(sample / "displacement.png")
            np.testing.assert_array_equal(actual, maps["disp"][y:y + height, x:x + width])
            self.assertEqual(header["sample_bits"], 16)
            self.assertEqual(record["map_metadata"]["input"]["encoding"], "source_srgb_assumed")
            self.assertEqual(record["map_metadata"]["height"]["encoding"], "linear_data")
            self.assertTrue((sample / "anisotropy_rotation.png").exists())
            self.assertEqual((sample / "source_notes/material-notes.txt").read_text(), "Unique source note\n")
        self.assertEqual(original_hashes, {path.name: MODULE.file_sha256(path) for path in folder.iterdir()})

    def test_manual_samples_not_overwritten_and_reruns_are_idempotent(self):
        self.material()
        manual = self.output / "samples/test_surface_001"
        manual.mkdir(parents=True)
        (manual / "handmade.txt").write_text("Keep this")
        _, first = self.prepare()
        first_hashes = {path: MODULE.file_sha256(path) for path in (self.output / "samples").rglob("*") if path.is_file()}
        _, second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual((manual / "handmade.txt").read_text(), "Keep this")
        self.assertEqual(first_hashes, {path: MODULE.file_sha256(path) for path in (self.output / "samples").rglob("*") if path.is_file()})

    def test_directx_normal_conversion_is_exact_reversible_and_rgb16(self):
        _, maps = self.material(dx=True)
        _, records = self.prepare()
        crop, header = MODULE.read_png(self.output / "samples" / records[0]["sample_id"] / "normal.png")
        expected = maps["nor_dx"][:2, :2].copy()
        expected[:, :, 1] = 65535 - expected[:, :, 1]
        np.testing.assert_array_equal(crop, expected)
        self.assertEqual(header["sample_bits"], 16)
        self.assertEqual(records[0]["normal_convention"], "OpenGL +Y")
        self.assertFalse(records[0]["map_metadata"]["normal"]["exact_source_crop"])
        self.assertEqual(MODULE.verify_sample(self.output / "samples" / records[0]["sample_id"], True), [])

    def test_dimension_mismatch_requires_explicit_registration_policy(self):
        self.material(mismatch=True)
        strict = MODULE.discover_materials(self.sources, 2)
        self.assertFalse(strict[0]["ready"])
        self.assertIn("registration", " ".join(strict[0]["problems"]))
        material, records = self.prepare(dimension_policy="common-intersection")
        self.assertEqual(material["crop_rectangles_top_left_xywh"][1], [6, 1, 2, 2])
        self.assertEqual(records[0]["dimension_policy"], "common-intersection")
        self.assertEqual(MODULE.main(["prepare", "--sources", str(self.sources), "--output", str(self.output), "--dimension-policy", "common-intersection"]), 1)

    def test_explicit_target_transfer_conversion_keeps_bitdepth_and_diffuse(self):
        _, maps = self.material()
        _, records = self.prepare(overrides={"test_surface": {"height": "srgb_to_linear"}})
        sample = self.output / "samples" / records[0]["sample_id"]
        actual, metadata = MODULE.read_png(sample / "displacement.png")
        normalized = maps["disp"][:2, :2].astype(np.float64) / 65535
        expected = np.rint(np.where(normalized <= .04045, normalized / 12.92, ((normalized + .055) / 1.055) ** 2.4) * 65535).astype(np.uint16)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(metadata["sample_bits"], 16)
        self.assertEqual(metadata["png_gamma"], 1)
        diffuse, _ = MODULE.read_png(sample / "diffuse.png")
        np.testing.assert_array_equal(diffuse, maps["diff"][:2, :2])
        self.assertEqual(MODULE.verify_sample(sample, True), [])

    def test_changed_source_or_tampered_crop_cannot_be_overwritten(self):
        folder, maps = self.material()
        _, records = self.prepare()
        target = self.output / "samples" / records[0]["sample_id"] / "displacement.png"
        independent_png(target, np.zeros((2, 2, 1), dtype=np.uint16))
        self.assertTrue(MODULE.verify_sample(target.parent))
        with self.assertRaisesRegex(ValueError, "failed verification"):
            self.prepare()
        independent_png(folder / "test_surface_disp_4k.png", maps["disp"] + 1)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.prepare()

    def test_normalization_plans_only_folders_and_rejects_collisions(self):
        folder, _ = self.material()
        materials = MODULE.discover_materials(self.sources, 2)
        self.assertEqual(MODULE.normalization_plan(materials), [{"from": str(folder.resolve()), "to": str((self.sources / "test_surface").resolve())}])
        self.assertTrue(folder.exists())
        (self.sources / "test_surface").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            MODULE.normalization_plan(materials)

    def test_manifest_path_escape_and_map_disagreement_are_rejected(self):
        self.material()
        _, records = self.prepare()
        sample = self.output / "samples" / records[0]["sample_id"]
        metadata_path = sample / "sample.json"
        record = json.loads(metadata_path.read_text())
        record["maps"]["height"] = "../../outside.png"
        record["map_metadata"]["height"]["filename"] = "../../outside.png"
        metadata_path.write_text(json.dumps(record))
        self.assertIn("contained", " ".join(MODULE.verify_sample(sample)))
        record["maps"]["height"] = "displacement.png"
        metadata_path.write_text(json.dumps(record))
        self.assertIn("disagrees", " ".join(MODULE.verify_sample(sample)))

    def test_symlink_escape_and_dtype_manifest_mismatch_are_rejected(self):
        self.material()
        _, records = self.prepare()
        sample = self.output / "samples" / records[0]["sample_id"]
        outside = self.root / "external.png"
        target = sample / "displacement.png"
        target.rename(outside)
        target.symlink_to(outside)
        self.assertIn("escapes", " ".join(MODULE.verify_sample(sample)))
        target.unlink()
        outside.rename(target)
        record = json.loads((sample / "sample.json").read_text())
        record["map_metadata"]["height"]["sample_dtype"] = "uint8"
        (sample / "sample.json").write_text(json.dumps(record))
        self.assertIn("dtype", " ".join(MODULE.verify_sample(sample)))

    def test_foreign_dataset_index_and_output_under_sources_are_preserved(self):
        self.material()
        self.output.mkdir()
        index = self.output / "dataset.json"
        original = '{"schema_version":1,"generator":"another-workflow","note":"keep"}\n'
        index.write_text(original)
        self.assertEqual(MODULE.main(["prepare", "--sources", str(self.sources), "--output", str(self.output), "--crop-size", "2"]), 1)
        self.assertEqual(index.read_text(), original)
        for output in (self.sources, self.sources / "generated"):
            self.assertEqual(MODULE.main(["prepare", "--sources", str(self.sources), "--output", str(output), "--crop-size", "2"]), 1)
        self.assertFalse((self.sources / "generated").exists())

    def test_changed_approval_is_rejected_without_modifying_existing_metadata(self):
        self.material()
        _, records = self.prepare()
        sample = self.output / "samples" / records[0]["sample_id"] / "sample.json"
        original = sample.read_bytes()
        material = MODULE.discover_materials(self.sources, 2)[0]
        with self.assertRaisesRegex(ValueError, "approval or split"):
            MODULE.prepare_material(material, self.output, {}, False, .2)
        self.assertEqual(sample.read_bytes(), original)

    def test_partial_folder_normalization_rolls_back_without_touching_files(self):
        first = self.sources / "First Name"
        second = self.sources / "Second Name"
        first.mkdir()
        second.mkdir()
        (first / "keep.txt").write_text("original bytes")
        plan = [{"from": str(first), "to": str(self.sources / "first_name")},
                {"from": str(second), "to": str(self.sources / "second_name")}]
        real_rename = MODULE.os.rename
        calls = 0
        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Injected second rename failure")
            return real_rename(source, target)
        with mock.patch.object(MODULE.os, "rename", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "rolled back"):
                MODULE.apply_normalization(plan)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual((first / "keep.txt").read_text(), "original bytes")
        self.assertFalse((self.sources / "first_name").exists())

    def test_cli_roundtrip_index_verification_and_changed_options_preserve_index(self):
        self.material()
        command = ["prepare", "--sources", str(self.sources), "--output", str(self.output), "--crop-size", "2", "--approve"]
        self.assertEqual(MODULE.main(command), 0)
        self.assertEqual(MODULE.main(["verify", "--dataset", str(self.output), "--check-sources"]), 0)
        index = self.output / "dataset.json"
        original = index.read_bytes()
        self.assertEqual(MODULE.main(command[:-1]), 1)
        self.assertEqual(index.read_bytes(), original)
        with mock.patch.object(MODULE, "prepare_material", side_effect=ValueError("Injected crop failure")):
            self.assertEqual(MODULE.main(command), 1)
        self.assertEqual(index.read_bytes(), original)

    def test_zero_validation_fraction_preserves_whole_maps_for_fitting_pilot(self):
        folder, maps = self.material(shape=(2, 2))
        originals = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        command = ["prepare", "--sources", str(self.sources), "--output", str(self.output),
                   "--crop-size", "2", "--validation-fraction", "0", "--approve"]
        self.assertEqual(MODULE.main(command), 0)
        index = json.loads((self.output / "dataset.json").read_text())
        self.assertEqual(index["validation_fraction"], 0)
        self.assertEqual(len(index["samples"]), 1)
        sample = index["samples"][0]
        self.assertEqual(sample["split"], "train")
        metadata = json.loads((self.output / sample["path"] / "sample.json").read_text())
        self.assertEqual(metadata["crop_rectangle_top_left_xywh"], [0, 0, 2, 2])
        actual, header = MODULE.read_png(MODULE.resolve_map_path(self.output / sample["path"], metadata, "height"))
        self.assertTrue(index["original_sources_required_for_training"])
        self.assertFalse(list((self.output / sample["path"]).glob("*.png")))
        np.testing.assert_array_equal(actual, maps["disp"])
        self.assertEqual(header["sample_bits"], 16)
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        self.assertEqual(MODULE.main(["verify", "--dataset", str(self.output), "--check-sources"]), 0)

    def test_whole_map_preparation_preserves_rectangular_boundaries_and_all_16bit_codes(self):
        folder, maps = self.material(shape=(2, 4))
        originals = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        command = ["prepare", "--sources", str(self.sources), "--output", str(self.output),
                   "--crop-size", "4", "--whole-maps", "--validation-fraction", "0", "--approve"]
        self.assertEqual(MODULE.main(command), 0)
        index = json.loads((self.output / "dataset.json").read_text())
        self.assertTrue(index["whole_source_maps"])
        self.assertEqual(len(index["samples"]), 1)
        sample = self.output / index["samples"][0]["path"]
        metadata = json.loads((sample / "sample.json").read_text())
        self.assertEqual(metadata["sample_pixel_dimensions"], [4, 2])
        self.assertEqual(metadata["crop_rectangle_top_left_xywh"], [0, 0, 4, 2])
        for role, source_key in (("input", "diff"), ("height", "disp"), ("normal", "nor_gl"), ("roughness", "rough")):
            actual, header = MODULE.read_png(sample / metadata["maps"][role])
            np.testing.assert_array_equal(actual, maps[source_key])
            self.assertEqual(header["sample_bits"], 16)
        self.assertEqual(MODULE.verify_sample(sample, check_sources=True), [])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        before = (self.output / "dataset.json").read_bytes()
        self.assertEqual(MODULE.main([arg for arg in command if arg != "--whole-maps"]), 1)
        self.assertEqual((self.output / "dataset.json").read_bytes(), before)

    def test_parallel_material_preparation_keeps_exact_pairs_and_deterministic_index(self):
        self.material("First", "first", shape=(2, 2))
        self.material("Second", "second", shape=(2, 2))
        barrier = threading.Barrier(2)
        original = MODULE.prepare_material
        def synchronized(*args):
            barrier.wait(timeout=5)
            return original(*args)
        with mock.patch.object(MODULE, "source_cache_budget", return_value=1024**3), mock.patch.object(MODULE.os, "cpu_count", return_value=4), mock.patch.object(MODULE, "prepare_material", side_effect=synchronized):
            self.assertEqual(MODULE.main(["prepare", "--sources", str(self.sources), "--output", str(self.output),
                                         "--crop-size", "2", "--workers", "2", "--validation-fraction", "0"]), 0)
        index = json.loads((self.output / "dataset.json").read_text())
        self.assertEqual([row["material_id"] for row in index["samples"]], ["first", "second"])
        for row in index["samples"]:
            self.assertEqual(MODULE.verify_sample(self.output / row["path"], check_sources=True), [])

    def test_dataset_index_path_escape_is_rejected_before_reading_sample(self):
        self.output.mkdir()
        document = {"generator": MODULE.GENERATOR, "schema_version": 2,
                    "samples": [{"sample_id": "external", "path": "../external"}]}
        (self.output / "dataset.json").write_text(json.dumps(document))
        self.assertEqual(MODULE.main(["verify", "--dataset", str(self.output)]), 1)

    def test_cached_source_verification_rechecks_expected_hash_on_every_use(self):
        folder, maps = self.material()
        _, records = self.prepare()
        sample = self.output / "samples" / records[0]["sample_id"]
        cache = MODULE.SourceDecodeCache(100000)
        self.assertEqual(MODULE.verify_sample(sample, True, cache), [])
        misses = cache.misses
        self.assertEqual(MODULE.verify_sample(sample, True, cache), [])
        self.assertEqual(cache.misses, misses)
        self.assertGreater(cache.hits, 0)
        metadata_path = sample / "sample.json"
        record = json.loads(metadata_path.read_text())
        record["map_metadata"]["height"]["source"]["file_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(record))
        self.assertIn("source checksum changed", " ".join(MODULE.verify_sample(sample, True, cache)))
        independent_png(folder / "test_surface_disp_4k.png", maps["disp"] + 1)
        self.assertIn("source crop values changed", " ".join(MODULE.verify_sample(sample, True, cache)))
        self.assertGreater(cache.misses, misses)

    def test_published_license_provenance_requires_actual_matching_parent_bytes(self):
        self.material()
        material = MODULE.discover_materials(self.sources, 2)[0]
        source = material["maps"]["height"]
        audit = {"timestamp_utc": "2026-10-07T00:00:00Z", "files": [{
            "asset_id": "test_surface", "filename": source["filename"], "published_exact_match": True,
            "published_md5": source["file_md5"], "published_bytes": source["file_bytes"],
            "api_url": "https://api.polyhaven.com/files/test_surface",
            "published_url": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/test_surface/test_surface_disp_4k.png"}]}
        provenance = MODULE.published_source_provenance(source, "test_surface", audit)
        self.assertEqual(provenance["license"], "CC0-1.0")
        self.assertEqual(provenance["provider"], "Poly Haven")
        self.assertEqual(MODULE.published_source_provenance(source, "other_material", audit), {})
        changed = {**source, "file_md5": "0" * 32}
        self.assertEqual(MODULE.published_source_provenance(changed, "test_surface", audit), {})
        self.assertEqual(MODULE.published_source_provenance(source, "test_surface", None), {})

    def test_disp_gl_alias_is_raw_height_and_unknown_origin_stays_unverified(self):
        folder, maps = self.material(name="wicker013", identity="Wicker013")
        (folder / "Wicker013_disp_4k.png").rename(folder / "Wicker013_disp_gl_4k.png")
        material, records = self.prepare()
        self.assertEqual(material["material_id"], "wicker013")
        self.assertEqual(material["maps"]["height"]["suffix"], "disp_gl")
        for record in records:
            self.assertIsNone(record["source_url"])
            self.assertEqual(record["source_origin"], "unverified")
            self.assertNotIn("source_license", record)
            self.assertNotIn("provider", record["map_metadata"]["height"]["source"])
            self.assertEqual(record["map_metadata"]["height"]["transforms"], [])
            x, y, width, height = record["crop_rectangle_top_left_xywh"]
            sample = self.output / "samples" / record["sample_id"]
            actual, metadata = MODULE.read_png(sample / "displacement.png")
            np.testing.assert_array_equal(actual, maps["disp"][y:y + height, x:x + width])
            self.assertEqual(metadata["sample_bits"], 16)
            self.assertEqual(MODULE.verify_sample(sample, True), [])

    def test_nested_api_audit_provenance_preserves_current_byte_checks(self):
        self.material()
        material = MODULE.discover_materials(self.sources, 2)[0]
        audit = {"snapshot_completed_utc": "2026-10-07T18:18:39Z", "files": []}
        for source in material["maps"].values():
            audit["files"].append({"asset_id": "test_surface", "filename": source["filename"],
                "published_exact_match": True, "published_md5": source["file_md5"],
                "published_bytes": source["file_bytes"],
                "api_provenance": {"api_url": "https://api.polyhaven.com/files/test_surface"},
                "published_url": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/test_surface/" + source["filename"]})
        provenance = MODULE.published_source_provenance(material["maps"]["height"], "test_surface", audit)
        self.assertEqual(provenance["license"], "CC0-1.0")
        self.assertEqual(provenance["published_audit_timestamp_utc"], audit["snapshot_completed_utc"])
        self.assertEqual(provenance["published_api_url"], "https://api.polyhaven.com/files/test_surface")
        records = MODULE.prepare_material(material, self.output, {}, True, .2, audit)
        self.assertEqual(records[0]["source_url"], "https://polyhaven.com/a/test_surface")
        self.assertEqual(records[0]["source_license"], "CC0-1.0")
        self.assertEqual(records[0]["source_origin"], "all_maps_match_published_polyhaven_files")
        changed = {**material["maps"]["height"], "file_bytes": material["maps"]["height"]["file_bytes"] + 1}
        self.assertEqual(MODULE.published_source_provenance(changed, "test_surface", audit), {})

    def test_published_md5_case_is_normalized_only_after_valid_full_hex_check(self):
        self.material()
        material = MODULE.discover_materials(self.sources, 2)[0]
        source = material["maps"]["normal"]
        uppercase = source["file_md5"].upper()
        entry = {"asset_id": "test_surface", "filename": source["filename"],
                 "published_exact_match": True, "published_md5": uppercase,
                 "published_bytes": source["file_bytes"],
                 "api_url": "https://api.polyhaven.com/files/test_surface",
                 "published_url": "https://dl.polyhaven.org/file/ph-assets/Textures/png/4k/test_surface/" + source["filename"]}
        audit = {"timestamp_utc": "2026-10-07T00:00:00Z", "files": [entry]}
        provenance = MODULE.published_source_provenance(source, "test_surface", audit)
        self.assertEqual(provenance["license"], "CC0-1.0")
        self.assertEqual(provenance["published_md5"], uppercase)
        self.assertEqual(MODULE.published_source_provenance({**source, "file_md5": uppercase}, "test_surface", audit)["license"], "CC0-1.0")
        for invalid in ("0" * 32, "g" * 32, "d" * 31, " " + uppercase, None):
            candidate = {"files": [{**entry, "published_md5": invalid}]}
            self.assertEqual(MODULE.published_source_provenance(source, "test_surface", candidate), {})
        self.assertEqual(MODULE.published_source_provenance({**source, "file_md5": "g" * 32}, "test_surface", {"files": [{**entry, "published_md5": "G" * 32}]}), {})

    def duplicate_sources(self, canonical_exists=True):
        first, _ = self.material()
        second = self.sources / ("test_surface" if canonical_exists else "Test-Surface")
        shutil.copytree(first, second)
        (first / ".DS_Store").write_bytes(b"incoming hidden metadata")
        (second / ".DS_Store").write_bytes(b"canonical hidden metadata")
        return first, second

    def test_identical_duplicate_archive_preserves_all_files_and_restoration_hash_log(self):
        incoming, canonical = self.duplicate_sources()
        incoming_hashes = {path.name: MODULE.file_sha256(path) for path in incoming.iterdir()}
        canonical_hashes = {path.name: MODULE.file_sha256(path) for path in canonical.iterdir()}
        materials = MODULE.discover_materials(self.sources, 2)
        with self.assertRaisesRegex(ValueError, "collision"):
            MODULE.normalization_plan(materials)
        archive = self.root / "archived-duplicates"
        plan = MODULE.normalization_plan(materials, archive)
        self.assertEqual(len(plan), 1)
        operation = plan[0]
        self.assertEqual(operation["kind"], "archive_identical_duplicate")
        self.assertFalse(archive.exists())
        MODULE.apply_normalization(plan)
        self.assertFalse(incoming.exists())
        archived = Path(operation["to"])
        self.assertEqual(incoming_hashes, {path.name: MODULE.file_sha256(path) for path in archived.iterdir()})
        self.assertEqual(canonical_hashes, {path.name: MODULE.file_sha256(path) for path in canonical.iterdir()})
        log = json.loads(Path(operation["archive_log_path"]).read_text())
        self.assertEqual(log["status"], "archived")
        self.assertEqual(log["operation"]["restoration"], {"from": str(archived), "to": str(incoming.resolve())})
        self.assertTrue(any(item["hidden"] for item in log["operation"]["incoming_files"]))
        self.assertEqual({item["relative_path"]: item["sha256"] for item in log["operation"]["incoming_files"]}, incoming_hashes)
        # Restoring the complete folder is lossless and does not merge files.
        os.rename(archived, incoming)
        self.assertEqual(incoming_hashes, {path.name: MODULE.file_sha256(path) for path in incoming.iterdir()})
        next_plan = MODULE.normalization_plan(materials, archive)
        self.assertNotEqual(next_plan[0]["to"], operation["to"])

    def test_duplicate_archive_refuses_changed_maps_unique_notes_and_symlinks(self):
        incoming, canonical = self.duplicate_sources()
        materials = MODULE.discover_materials(self.sources, 2)
        archive = self.root / "archived-duplicates"
        changed = incoming / "test_surface_disp_4k.png"
        original = changed.read_bytes()
        values, _ = MODULE.read_png(changed)
        independent_png(changed, values + 1)
        with self.assertRaisesRegex(ValueError, "differs"):
            MODULE.normalization_plan(materials, archive)
        changed.write_bytes(original)
        note = incoming / "unique-note.txt"
        note.write_text("Never discard this")
        with self.assertRaisesRegex(ValueError, "unique notes"):
            MODULE.normalization_plan(materials, archive)
        note.unlink()
        (incoming / ".hidden-symlink").symlink_to(canonical / "test_surface_disp_4k.png")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            MODULE.normalization_plan(materials, archive)
        self.assertTrue(incoming.exists())
        self.assertTrue(canonical.exists())
        self.assertFalse(archive.exists())

    def test_duplicate_archive_outside_sources_and_new_primary_group(self):
        first, second = self.duplicate_sources(canonical_exists=False)
        materials = MODULE.discover_materials(self.sources, 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.normalization_plan(materials, self.sources / "archive")
        plan = MODULE.normalization_plan(materials, self.root / "archive")
        self.assertEqual(len(plan), 2)
        self.assertNotIn("kind", plan[0])
        self.assertEqual(plan[1]["kind"], "archive_identical_duplicate")
        MODULE.apply_normalization(plan)
        self.assertTrue((self.sources / "test_surface").is_dir())
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        self.assertTrue(Path(plan[1]["to"]).is_dir())

    def test_archive_and_prior_renames_rollback_on_later_failure(self):
        first, second = self.duplicate_sources(canonical_exists=False)
        materials = MODULE.discover_materials(self.sources, 2)
        plan = MODULE.normalization_plan(materials, self.root / "archive")
        third = self.sources / "Third Folder"
        third.mkdir()
        plan.append({"from": str(third), "to": str(self.sources / "third_folder")})
        original_rename = MODULE.os.rename
        calls = 0
        def fail_third(source, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("Injected rename failure after archival")
            original_rename(source, target)
        with mock.patch.object(MODULE.os, "rename", side_effect=fail_third):
            with self.assertRaisesRegex(ValueError, "rolled back"):
                MODULE.apply_normalization(plan)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertTrue(third.exists())
        self.assertFalse((self.sources / "test_surface").exists())
        self.assertFalse(Path(plan[1]["to"]).exists())
        self.assertEqual(json.loads(Path(plan[1]["archive_log_path"]).read_text())["status"], "rolled_back")

    def test_duplicate_change_after_preflight_prevents_archive(self):
        incoming, canonical = self.duplicate_sources()
        plan = MODULE.normalization_plan(MODULE.discover_materials(self.sources, 2), self.root / "archive")
        (incoming / "new-note.txt").write_text("Arrived after plan")
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            MODULE.apply_normalization(plan)
        self.assertTrue(incoming.exists())
        self.assertTrue(canonical.exists())
        self.assertFalse(Path(plan[0]["to"]).exists())

    def test_region_geometry_native_1k_prefers_disjoint_center(self):
        regions = MODULE.heldout_regions(4096, 4096, 1024)
        self.assertEqual([item["split"] for item in regions], ["train", "train", "validation"])
        self.assertEqual(regions[2]["rectangle"], [1536, 1536, 1024, 1024])
        self.assertEqual(regions[2]["region"], "center")
        for first in regions:
            for second in regions:
                if first is not second:
                    self.assertFalse(MODULE.rectangles_overlap(first["rectangle"], second["rectangle"]))

    def test_region_geometry_native_2k_rectangle_has_one_train_one_validation(self):
        regions = MODULE.heldout_regions(2048, 4096, 2048)
        self.assertEqual(regions, [
            {"ordinal": 1, "split": "train", "region": "top_left", "rectangle": [0, 0, 2048, 2048]},
            {"ordinal": 2, "split": "validation", "region": "opposite_corner", "rectangle": [0, 2048, 2048, 2048]}])
        self.assertFalse(MODULE.rectangles_overlap(regions[0]["rectangle"], regions[1]["rectangle"]))
        square = MODULE.heldout_regions(4096, 4096, 2048)
        self.assertEqual(square[2]["region"], "bottom_left")
        self.assertEqual(square[2]["rectangle"], [0, 2048, 2048, 2048])
        with self.assertRaisesRegex(ValueError, "no disjoint"):
            MODULE.heldout_regions(2048, 2048, 2048)

    def test_selected_region_subset_preserves_sources_and_cannot_drop_existing_materials(self):
        self.material("First", "first", shape=(8, 8))
        self.material("Second", "second", shape=(8, 8))
        original = {path: path.read_bytes() for path in self.sources.rglob("*") if path.is_file()}
        report = MODULE.prepare_region_splits(self.output, self.sources, 2, material_ids=["second"])
        self.assertEqual(report["requested_material_ids"], ["second"])
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual({item["material_id"] for item in json.loads((self.output / "dataset.json").read_text())["samples"]}, {"second"})
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        before = (self.output / "dataset.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "omit existing"):
            MODULE.prepare_region_splits(self.output, self.sources, 2, material_ids=["first"])
        self.assertEqual((self.output / "dataset.json").read_bytes(), before)
        self.assertEqual(MODULE.prepare_region_splits(self.output, self.sources, 2, material_ids=["second"])["new_sample_count"], 0)

    def test_region_preparation_decodes_each_parent_once_for_all_native_crops(self):
        self.material(shape=(8, 8))
        original_decoder = MODULE.read_png
        parent_decodes = []
        def count_parent_reads(path):
            if Path(path).resolve().is_relative_to(self.sources.resolve()):
                parent_decodes.append(Path(path).resolve())
            return original_decoder(path)
        with mock.patch.object(MODULE, "read_png", side_effect=count_parent_reads):
            report = MODULE.prepare_region_splits(self.output, self.sources, 2)
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(len(parent_decodes), 5)
        self.assertEqual(len(set(parent_decodes)), 5)
        index = json.loads((self.output / "dataset.json").read_text())
        self.assertEqual([entry["split"] for entry in index["samples"]], ["train", "train", "validation"])
        for entry in index["samples"]:
            self.assertEqual(MODULE.verify_sample(self.output / entry["path"], True), [])

    def test_unknown_material_selection_fails_before_creating_dataset(self):
        self.material()
        with self.assertRaisesRegex(ValueError, "Unknown or empty"):
            MODULE.prepare_region_splits(self.output, self.sources, 2, material_ids=["typo"])
        self.assertFalse(self.output.exists())

    def legacy_index(self, records):
        document = {"schema_version": 2, "generator": MODULE.GENERATOR, "crop_size": 2,
                    "validation_fraction": .2, "split_policy": "heldout materials",
                    "samples": [{"sample_id": item["sample_id"], "material_id": item["material_id"],
                                 "path": "samples/" + item["sample_id"], "split": item["split"], "status": item["status"]} for item in records]}
        MODULE.write_json(self.output / "dataset.json", document)
        return document

    def test_explicit_region_migration_preserves_pngs_backs_up_metadata_and_reruns(self):
        self.material(shape=(8, 8))
        _, old_records = self.prepare()
        self.legacy_index(old_records)
        index_path = self.output / "dataset.json"
        old_index_bytes = index_path.read_bytes()
        old_metadata = {item["sample_id"]: (self.output / "samples" / item["sample_id"] / "sample.json").read_bytes() for item in old_records}
        old_png_hashes = {path: MODULE.file_sha256(path) for path in (self.output / "samples").rglob("*.png")}
        with self.assertRaisesRegex(ValueError, "migrate-split-policy"):
            MODULE.prepare_region_splits(self.output, self.sources, 2)
        self.assertEqual(index_path.read_bytes(), old_index_bytes)
        report = MODULE.prepare_region_splits(self.output, self.sources, 2, True)
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["train_samples"], 2)
        self.assertEqual(report["validation_samples"], 1)
        backup = Path(report["metadata_backup"])
        self.assertTrue(backup.is_relative_to((self.output / "metadata-backups").resolve()))
        self.assertEqual((backup / "dataset.json").read_bytes(), old_index_bytes)
        for identity, raw in old_metadata.items():
            self.assertEqual((backup / "samples" / identity / "sample.json").read_bytes(), raw)
        self.assertEqual(old_png_hashes, {path: MODULE.file_sha256(path) for path in old_png_hashes})
        index = json.loads(index_path.read_text())
        self.assertEqual(index["split_strategy"], "heldout-region-v1")
        self.assertEqual(index["validation_scope"], "unseen_regions_of_known_materials")
        for entry in index["samples"]:
            sample = self.output / entry["path"]
            self.assertEqual(MODULE.verify_sample(sample, True), [])
        second_report = MODULE.prepare_region_splits(self.output, self.sources, 2)
        self.assertIsNone(second_report["metadata_backup"])
        self.assertEqual(second_report["new_sample_count"], 0)

    def test_region_migration_changes_only_split_for_two_crop_rectangle(self):
        self.material(shape=(4, 2))
        _, old_records = self.prepare()
        self.legacy_index(old_records)
        old_png_hashes = {path: MODULE.file_sha256(path) for path in (self.output / "samples").rglob("*.png")}
        report = MODULE.prepare_region_splits(self.output, self.sources, 2, True)
        self.assertEqual(report["new_sample_count"], 0)
        self.assertEqual(report["train_samples"], 1)
        self.assertEqual(report["validation_samples"], 1)
        index = json.loads((self.output / "dataset.json").read_text())
        self.assertEqual([item["split"] for item in index["samples"]], ["train", "validation"])
        self.assertEqual(old_png_hashes, {path: MODULE.file_sha256(path) for path in old_png_hashes})

    def test_region_migration_rolls_back_metadata_and_new_crops_after_publication_failure(self):
        self.material(shape=(8, 8))
        _, records = self.prepare()
        self.legacy_index(records)
        old_index = (self.output / "dataset.json").read_bytes()
        old_samples = {path: path.read_bytes() for path in (self.output / "samples").rglob("sample.json")}
        original_write_json = MODULE.write_json
        def fail_index(path, document):
            if path == self.output.resolve() / "dataset.json" and document.get("split_strategy") == "heldout-region-v1":
                raise OSError("Injected index publication failure")
            return original_write_json(path, document)
        with mock.patch.object(MODULE, "write_json", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "Injected"):
                MODULE.prepare_region_splits(self.output, self.sources, 2, True)
        self.assertEqual((self.output / "dataset.json").read_bytes(), old_index)
        for path, raw in old_samples.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertFalse((self.output / "samples/test_surface_auto_003").exists())

    def test_region_migration_refuses_tampered_source_or_existing_crop(self):
        folder, maps = self.material(shape=(8, 8))
        _, records = self.prepare()
        self.legacy_index(records)
        index_bytes = (self.output / "dataset.json").read_bytes()
        source = folder / "test_surface_disp_4k.png"
        source_bytes = source.read_bytes()
        independent_png(source, maps["disp"] + 1)
        with self.assertRaisesRegex(ValueError, "unverified sample"):
            MODULE.prepare_region_splits(self.output, self.sources, 2, True)
        source.write_bytes(source_bytes)
        crop = self.output / "samples" / records[0]["sample_id"] / "displacement.png"
        independent_png(crop, np.zeros((2, 2, 1), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "unverified sample"):
            MODULE.prepare_region_splits(self.output, self.sources, 2, True)
        self.assertEqual((self.output / "dataset.json").read_bytes(), index_bytes)


if __name__ == "__main__":
    unittest.main()
