"""Precision, fair-comparison and review safety tests; no GPU jobs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("review_material_quality", SCRIPTS / "review_material_quality.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterialQualityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        diffuse = np.full((8, 8, 3), .25, dtype=np.float32)
        np.save(self.sources / "diffuse.npy", diffuse)
        codes = np.arange(64, dtype=np.uint16).reshape(8, 8) + 32700
        np.save(self.sources / "target.npy", codes)
        np.save(self.sources / "prediction.npy", np.full((8, 8), .1, dtype=np.float32))
        self.manifest = {"schema": MODULE.SCHEMA, "materials": [{
            "material_id": "surface", "diffuse": "sources/diffuse.npy", "diffuse_encoding": "linear",
            "height_gain": 1.4, "height_offset": -.2, "displacement_scale_m": .025,
            "variants": [{"name": "target", "height": "sources/target.npy"},
                         {"name": "prediction", "height": "sources/prediction.npy", "diagnostic_metrics": {"mae": .002}}]}]}

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, manifest=None, output_name="review"):
        return MODULE.prepare_manifest(manifest or self.manifest, self.root, self.root / output_name, 64, 1)

    def test_uint16_adjacent_codes_stay_distinct_and_are_not_gamma_transformed(self):
        raw = np.load(self.sources / "target.npy")
        values, information = MODULE.load_numeric(self.sources / "target.npy")
        expected = raw.astype(np.float32) / 65535
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(len(np.unique(values)), 64)
        self.assertEqual(information["source_dtype"], "uint16")
        self.assertNotEqual(float(values.min()), 0)
        self.assertNotEqual(float(values.max()), 1)

    def test_numeric_float_values_and_range_survive_exr_roundtrip_without_clipping(self):
        values = np.array([[-.2, .1234567], [.9876543, 1.4]], dtype=np.float32)
        path = self.sources / "unbounded.npy"
        np.save(path, values)
        actual, information = MODULE.load_numeric(path)
        np.testing.assert_array_equal(actual, values)
        exr = self.root / "numeric.exr"
        MODULE.write_exr(exr, actual)
        roundtrip, _ = MODULE.load_numeric(exr)
        np.testing.assert_array_equal(roundtrip, values)
        self.assertEqual(information["outside_0_1_samples"], 2)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            MODULE.write_exr(exr, values)

    def test_shared_gain_scale_camera_lights_and_raw_variants_preserve_sources(self):
        before = {path: MODULE.digest(path) for path in self.sources.iterdir()}
        prepared = self.prepare()
        group = prepared["materials"][0]
        self.assertEqual(group["surface_mode"], "geometric_displacement")
        self.assertEqual(group["height_gain"], 1.4)
        self.assertEqual(group["height_offset"], -.2)
        self.assertEqual(group["displacement_scale_m"], .025)
        self.assertEqual(group["geometry_edges_per_width"], 1024)
        self.assertEqual(set(prepared["lighting"]), {"neutral", "grazing"})
        self.assertEqual(prepared["render_modes"], ["material", "clay"])
        target, _ = MODULE.load_numeric(self.root / "review" / group["variants"][0]["height"])
        predicted, _ = MODULE.load_numeric(self.root / "review" / group["variants"][1]["height"])
        np.testing.assert_array_equal(target, np.load(self.sources / "target.npy").astype(np.float32) / 65535)
        np.testing.assert_array_equal(predicted, np.full((8, 8), .1, dtype=np.float32))
        self.assertEqual(before, {path: MODULE.digest(path) for path in self.sources.iterdir()})
        self.assertNotIn("height_gain", group["variants"][0])

    def test_height_rgb_channel_disagreement_and_meaningful_alpha_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "averaging"):
            MODULE.scalar_map(np.array([[[.1, .2, .1]]], dtype=np.float32), "Height")
        gray_alpha = np.array([[[.5, 65527 / 65535], [.6, 1]]], dtype=np.float32)
        np.testing.assert_array_equal(MODULE.scalar_map(gray_alpha, "Height", "uint16"), gray_alpha[:, :, 0])
        with self.assertRaisesRegex(ValueError, "non-opaque"):
            MODULE.scalar_map(gray_alpha, "Height", "uint8")
        gray_alpha[0, 0, 1] = 65526 / 65535
        with self.assertRaisesRegex(ValueError, "non-opaque"):
            MODULE.scalar_map(gray_alpha, "Height", "uint16")

    def test_nearopaque_uint16_rgb_is_not_alpha_scaled(self):
        values = np.array([[[.1, .2, .3, 65527 / 65535]]], dtype=np.float32)
        actual = MODULE.rgb_map(values, "Diffuse", "uint16")
        np.testing.assert_array_equal(actual, values[:, :, :3])
        with self.assertRaisesRegex(ValueError, "non-opaque"):
            MODULE.rgb_map(values, "Diffuse", "float32")

    def test_explicit_diffuse_transfer_and_aligned_native_dimensions_are_required(self):
        manifest = json.loads(json.dumps(self.manifest))
        del manifest["materials"][0]["diffuse_encoding"]
        with self.assertRaisesRegex(ValueError, "diffuse_encoding"):
            self.prepare(manifest)
        np.save(self.sources / "small.npy", np.zeros((4, 4), dtype=np.float32))
        manifest = json.loads(json.dumps(self.manifest))
        manifest["materials"][0]["variants"][1]["height"] = "sources/small.npy"
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.prepare(manifest, "second")

    def test_existing_review_and_unsafe_names_are_not_overwritten(self):
        output = self.root / "review"
        output.mkdir()
        original = b'{"decision":"keep my ratings"}'
        (output / "review.json").write_bytes(original)
        with self.assertRaisesRegex(ValueError, "preserve"):
            self.prepare()
        self.assertEqual((output / "review.json").read_bytes(), original)
        manifest = json.loads(json.dumps(self.manifest))
        manifest["materials"][0]["material_id"] = "../outside"
        with self.assertRaisesRegex(ValueError, "names"):
            self.prepare(manifest, "safe")

    def test_normal_only_requires_independent_opengl_normal_for_every_variant(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["materials"][0]["surface_mode"] = "normal_only"
        with self.assertRaisesRegex(ValueError, "independent OpenGL normal"):
            self.prepare(manifest)
        normal = np.tile(np.array([.5, .5, 1], dtype=np.float32), (8, 8, 1))
        np.save(self.sources / "normal.npy", normal)
        for item in manifest["materials"][0]["variants"]:
            item["normal"] = "sources/normal.npy"
            item["normal_convention"] = "DirectX"
        with self.assertRaisesRegex(ValueError, "OpenGL"):
            self.prepare(manifest, "second")

    def test_nonfinite_float64_precision_loss_and_pickle_arrays_are_rejected(self):
        for name, values, message in (("nan", np.array([[0, 0], [0, np.nan]], dtype=np.float32), "Non-finite"),
                                      ("float64", np.full((2, 2), .1, dtype=np.float64), "preserved exactly"),
                                      ("objects", np.full((2, 2), {}, dtype=object), "Object arrays")):
            path = self.sources / (name + ".npy")
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, message):
                MODULE.load_numeric(path)

    def test_manual_review_does_not_promote_metric_winner_and_html_is_editable(self):
        prepared = self.prepare()
        template = MODULE.review_template(prepared)
        self.assertEqual(len(template["entries"]), 2)
        for entry in template["entries"]:
            self.assertEqual(entry["decision"], "unreviewed")
            self.assertTrue(all(value is None for value in entry["ratings"].values()))
            self.assertTrue(entry["diagnostic_metrics_do_not_rank"])
            self.assertEqual(len(entry["renders"]), 4)
        MODULE.write_html(self.root / "review", template)
        page = (self.root / "review/index.html").read_text()
        self.assertIn("Save edited review JSON", page)
        self.assertIn("data-rating", page)
        self.assertIn("Clay views isolate relief", page)
        self.assertIn("0 = poor, 5 = good", page)
        self.assertIn("noise cleanliness", page)
        self.assertIn("higher means fewer artifacts", page)
        compile(MODULE.BLENDER_SCRIPT, "render_material_quality.py", "exec")

    def test_checkpoint_snapshot_mismatch_is_rejected_without_modifying_checkpoint(self):
        checkpoint = self.sources / "checkpoint.pt"
        checkpoint.write_bytes(b"immutable checkpoint bytes")
        manifest = json.loads(json.dumps(self.manifest))
        manifest["materials"][0]["variants"][1].update(checkpoint="sources/checkpoint.pt", checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "declared snapshot"):
            self.prepare(manifest)
        self.assertEqual(checkpoint.read_bytes(), b"immutable checkpoint bytes")
        manifest["materials"][0]["variants"][1]["checkpoint_sha256"] = MODULE.digest(checkpoint)
        prepared = self.prepare(manifest, "valid")
        self.assertEqual(prepared["materials"][0]["variants"][1]["checkpoint_provenance"]["sha256"], MODULE.digest(checkpoint))

    def roughness_manifest(self):
        np.save(self.sources / "rough-low.npy", np.full((8, 8), .12, dtype=np.float32))
        np.save(self.sources / "rough-high.npy", np.full((8, 8), .86, dtype=np.float32))
        manifest = json.loads(json.dumps(self.manifest))
        group = manifest["materials"][0]
        group.update(comparison_target="roughness", height="sources/target.npy")
        group["variants"] = [{"name": "low", "roughness": "sources/rough-low.npy"},
                             {"name": "high", "roughness": "sources/rough-high.npy"}]
        return manifest

    def test_roughness_comparison_uses_identical_height_with_raw_per_variant_roughness(self):
        prepared = self.prepare(self.roughness_manifest())
        group = prepared["materials"][0]
        self.assertEqual(group["comparison_target"], "roughness")
        self.assertEqual(group["variants"][0]["height"], group["variants"][1]["height"])
        self.assertEqual(group["variants"][0]["height_provenance"], group["variants"][1]["height_provenance"])
        for item, expected in zip(group["variants"], (.12, .86)):
            roughness, _ = MODULE.load_numeric(self.root / "review" / item["roughness"])
            np.testing.assert_array_equal(roughness, np.full((8, 8), expected, dtype=np.float32))
            height, _ = MODULE.load_numeric(self.root / "review" / item["height"])
            np.testing.assert_array_equal(height, np.load(self.sources / "target.npy").astype(np.float32) / 65535)
        template = MODULE.review_template(prepared)
        self.assertTrue(all(item["comparison_target"] == "roughness" for item in template["entries"]))

    def test_variant_roughness_cannot_confound_height_or_normal_comparisons(self):
        np.save(self.sources / "roughness.npy", np.full((8, 8), .2, dtype=np.float32))
        for target, name in (("height", "review"), ("normal", "normal-review")):
            manifest = json.loads(json.dumps(self.manifest))
            manifest["materials"][0]["comparison_target"] = target
            manifest["materials"][0]["variants"][0]["roughness"] = "sources/roughness.npy"
            with self.assertRaisesRegex(ValueError, "allowed only for roughness comparison"):
                self.prepare(manifest, name)

    def test_roughness_comparison_requires_shared_height_and_rejects_variant_height(self):
        manifest = self.roughness_manifest()
        del manifest["materials"][0]["height"]
        with self.assertRaisesRegex(ValueError, "one shared group height"):
            self.prepare(manifest)
        manifest = self.roughness_manifest()
        manifest["materials"][0]["variants"][0]["height"] = "sources/prediction.npy"
        with self.assertRaisesRegex(ValueError, "forbids per-variant height"):
            self.prepare(manifest, "second")

    def test_normal_comparison_defaults_normal_only_without_height_doubling(self):
        normal = np.tile(np.array([.5, .5, 1], dtype=np.float32), (8, 8, 1))
        np.save(self.sources / "normal.npy", normal)
        manifest = json.loads(json.dumps(self.manifest))
        group = manifest["materials"][0]
        group["comparison_target"] = "normal"
        for item in group["variants"]:
            item.update(normal="sources/normal.npy", normal_convention="OpenGL +Y")
        prepared = self.prepare(manifest)
        self.assertEqual(prepared["materials"][0]["surface_mode"], "normal_only")
        self.assertTrue(all(item["normal_used"] for item in prepared["materials"][0]["variants"]))
        group["surface_mode"] = "geometric_displacement"
        with self.assertRaisesRegex(ValueError, "not doubled"):
            self.prepare(manifest, "second")


if __name__ == "__main__":
    unittest.main()
