"""Teacher resolution contracts and full-size scientific EXR preservation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from ipde.learned_depth import LearnedDepthConfig, LearnedDepthPredictor
from ipde.formats import write_exr, verify_exr


class ResolutionPrecisionTests(unittest.TestCase):
    def predictor(self, model, size):
        import torch
        predictor = LearnedDepthPredictor.__new__(LearnedDepthPredictor)
        predictor.config = LearnedDepthConfig(model=model, input_size=size)
        predictor.torch, predictor.device = torch, 'cpu'
        return predictor

    def test_da3_requested_size_reaches_processor_and_native_mode_uses_longest_side(self):
        import torch
        rgb = np.zeros((42, 56, 3), np.uint8)
        original = rgb.copy()
        for size, effective in ((28, 28), (0, 56)):
            predictor = self.predictor('depth-anything-3', size)
            calls = []
            def processor(images, **options):
                calls.append(options)
                return torch.zeros(1, 3, 14, 28), None, None
            predictor.model = SimpleNamespace(input_processor=processor,
                model=lambda *_: {'depth': torch.full((1, 1, 14, 28), 2.25)})
            native, source, metadata, _ = predictor._depth_anything_3(rgb, 255)
            self.assertEqual(calls[0]['process_res'], effective)
            self.assertEqual(metadata['requested_input_size'], size)
            self.assertEqual(metadata['model_input_shape'], [14, 28])
            self.assertEqual(tuple(source.shape[-2:]), (42, 56))
            np.testing.assert_array_equal(rgb, original)
            self.assertTrue(bool((native == 2.25).all()))

    def test_v2_native_mode_uses_shortest_side_and_records_actual_grid(self):
        import torch
        rgb = np.zeros((42, 56, 3), np.float32)
        for size, effective in ((28, 28), (0, 42)):
            predictor = self.predictor('depth-anything-v2', size)
            sizes = []
            def resize(**options):
                sizes.append((options['width'], options['height']))
                return lambda sample: sample
            predictor.module = SimpleNamespace(Resize=resize,
                NormalizeImage=lambda **_: lambda sample: sample,
                PrepareForNet=lambda: lambda sample: {'image': np.moveaxis(sample['image'], -1, 0)})
            predictor.model = lambda tensor: torch.full((1, 42, 56), 3.5)
            native, source, metadata = predictor._depth_anything_v2(rgb)
            self.assertEqual(sizes, [(effective, effective)])
            self.assertEqual(metadata['requested_input_size'], size)
            self.assertEqual(metadata['model_input_shape'], [42, 56])
            self.assertEqual(tuple(source.shape[-2:]), (42, 56))
            self.assertTrue(bool((native == 3.5).all()))

    def test_5712_by_4284_single_float_channel_zip_round_trip_preserves_bits(self):
        import OpenEXR
        # Repeated rows compress well; include bit patterns lost by screen conversions.
        row = np.arange(5712, dtype=np.float32) * np.float32(0.125) - np.float32(17.5)
        array = np.broadcast_to(row, (4284, 5712)).copy()
        array.view(np.uint32)[0, :5] = [0x80000000, 0x7fc00123, 0x7f800000, 0xff800000, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.exr'
            write_exr(path, array)
            verify_exr(path, array)
            with OpenEXR.File(str(path), separate_channels=True) as image:
                self.assertEqual(list(image.channels()), ['Y'])
                self.assertEqual(image.channels()['Y'].pixels.dtype, np.dtype('float32'))
                self.assertEqual(image.channels()['Y'].pixels.shape, (4284, 5712))
                self.assertEqual(image.header()['compression'], OpenEXR.ZIP_COMPRESSION)


if __name__ == '__main__':
    unittest.main()
