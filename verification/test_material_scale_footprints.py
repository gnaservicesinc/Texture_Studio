import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('material_scales', Path(__file__).parents[1] / 'scripts/plan_material_scales.py')
scales = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scales)


class ScaleFootprintTests(unittest.TestCase):
    def test_equal_surface_rectangles_different_resolution_overlap(self):
        self.assertEqual(scales.footprint([0, 0, 1024, 1024], [2048, 2048]),
                         scales.footprint([0, 0, 2048, 2048], [4096, 4096]))

    def test_existing_center_blocks_every_published_2k_corner(self):
        heldout = [{'sample_id': 'center', 'rectangle': [1536, 1536, 1024, 1024], 'dimensions': [4096, 4096]}]
        candidates = scales.check_candidates([2048, 2048], 1024, heldout)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(c['overlapping_validation_sample_ids'] == ['center'] for c in candidates))

    def test_touching_surface_boundaries_share_no_pixels(self):
        self.assertFalse(scales.overlaps(scales.footprint([0, 0, 1024, 1024], [2048, 2048]),
                                         scales.footprint([2048, 0, 1024, 1024], [4096, 4096])))

    def test_bottom_left_holdout_allows_upper_2k_regions(self):
        heldout = [{'sample_id': 'bottom_left', 'rectangle': [0, 3072, 1024, 1024], 'dimensions': [4096, 4096]}]
        candidates = scales.check_candidates([2048, 2048], 1024, heldout)
        statuses = {item['region']: item['geometry_disjoint_from_validation'] for item in candidates}
        self.assertTrue(statuses['top_left'])
        self.assertTrue(statuses['top_right'])
        self.assertFalse(statuses['bottom_left'])
        self.assertTrue(all(not c['training_assigned'] for c in candidates))

    def test_non_square_parent_uses_independent_axis_scale(self):
        self.assertEqual(scales.footprint([0, 512, 1024, 512], [2048, 1024]),
                         scales.footprint([0, 1024, 2048, 1024], [4096, 2048]))

    def test_invalid_native_rectangles_fail(self):
        for rectangle in ([0, 0, 0, 1024], [-1, 0, 1024, 1024], [1025, 0, 1024, 1024], [0.0, 0, 1024, 1024]):
            with self.assertRaises(ValueError):
                scales.footprint(rectangle, [2048, 2048])
        with self.assertRaises(ValueError):
            scales.candidate_rectangles([512, 1024], 1024)


if __name__ == '__main__':
    unittest.main()
