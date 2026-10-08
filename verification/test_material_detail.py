"""CPU-only detail planning tests; importing the helper never imports bpy."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("render_material_detail", Path(__file__).resolve().parents[1] / "scripts/render_material_detail.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterialDetailTests(unittest.TestCase):
    def setUp(self):
        self.prepared = {"materials": [{"material_id": name, "physical_width_m": 1,
                                       "tiling_repeats": 1, "variants": [{"name": variant,
                                       "height_provenance": {"source_shape": [2048, 2048]}}
                                       for variant in ("target", "starting_head", "trained_2k")]}
                                      for name in ("stucco", "soil")]}

    def test_two_material_three_variant_two_mode_native_footprint(self):
        jobs = MODULE.plan_jobs(self.prepared, ["stucco", "soil"], ["target", "starting_head", "trained_2k"], 512)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["path"] for job in jobs}), 12)
        self.assertTrue(all(job["ortho_scale_m"] == .25 for job in jobs))
        self.assertTrue(all(job["scene"].endswith("__grazing") for job in jobs))
        self.assertTrue(all(job["camera"].endswith(" native detail camera") for job in jobs))

    def test_tiled_maps_account_for_one_native_tile_without_modifying_manifest(self):
        self.prepared["materials"][0]["tiling_repeats"] = 2
        jobs = MODULE.plan_jobs(self.prepared, ["stucco"], ["target"], 512)
        self.assertEqual(jobs[0]["ortho_scale_m"], .125)
        self.assertEqual(self.prepared["materials"][0]["variants"][0]["height_provenance"]["source_shape"], [2048, 2048])

    def test_unknown_or_unsafe_selection_and_unbounded_size_are_rejected(self):
        for names in (["../outside"], ["missing"]):
            with self.assertRaises(ValueError):
                MODULE.plan_jobs(self.prepared, names, ["target"], 512)
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            MODULE.plan_jobs(self.prepared, ["stucco"], ["missing"], 512)
        with self.assertRaisesRegex(ValueError, "Preview"):
            MODULE.plan_jobs(self.prepared, ["stucco"], ["target"], 8192)


if __name__ == "__main__":
    unittest.main()
