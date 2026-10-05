"""Scientific image storage changes containers, never pixels or array bits."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import OpenEXR

from ipde.array_storage import array_record, read_array, write_array
from ipde.dataset import DatasetError, load_dataset
from ipde.dataset_collection import CollectionOptions, compose_datasets, compress_dataset
from ipde.dataset_review import _array_records, _array_path, curate_dataset
from ipde.formats import FormatError, arrays_bit_equal, write_exr, write_png
from ipde.training import TrainingOptions, _load_sample
from verification.test_collection_storage import _dataset


class DatasetImageStorageTests(unittest.TestCase):
    def test_float32_exr_preserves_nan_payloads_signed_zero_and_channels(self):
        bits = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                         0x7fc01234, 0x7f801234, 0xffc05555, 0x00800001], np.uint32)
        value = bits.view(np.float32).reshape(2, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for channels in (None, 1, 2, 3, 4):
                array = value if channels is None else np.repeat(value[:, :, None], channels, axis=2)
                with self.subTest(channels=channels):
                    path = write_array(root / f"depth-{channels}.npz", array, storage="images")
                    self.assertEqual(path.suffix, ".exr")
                    self.assertTrue(arrays_bit_equal(array, read_array(path)))
                    with OpenEXR.File(str(path), separate_channels=True) as image:
                        self.assertEqual(image.header()["compression"], OpenEXR.ZIP_COMPRESSION)
                        self.assertEqual(json.loads(image.header()["ipdeArrayShape"]), list(array.shape))
                        self.assertTrue(all(channel.pixels.dtype == np.float32 for channel in image.channels().values()))
                        if channels in (None, 1):
                            self.assertEqual(set(image.channels()), {"Y"})

    def test_native_float16_exr_preserves_every_bit_pattern(self):
        # Includes every half-precision NaN payload, subnormal and signed zero.
        value = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256)
        with tempfile.TemporaryDirectory() as directory:
            path = write_array(Path(directory) / "native.npz", value, storage="images")
            self.assertEqual(path.suffix, ".exr")
            self.assertTrue(arrays_bit_equal(value, read_array(path)))
            with OpenEXR.File(str(path), separate_channels=True) as image:
                self.assertEqual(image.channels()["Y"].pixels.dtype, np.float16)

    def test_png_preserves_integer_codes_and_singleton_channel_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dtype in (np.uint8, np.uint16):
                for channels in (None, 1, 2, 3, 4):
                    value = np.random.default_rng(6).integers(0, np.iinfo(dtype).max + 1, (8, 13), dtype=dtype)
                    if channels is not None:
                        value = np.repeat(value[:, :, None], channels, axis=2)
                    with self.subTest(dtype=dtype, channels=channels):
                        path = write_array(root / f"rgb-{dtype}-{channels}.npz", value, storage="images")
                        self.assertEqual(path.suffix, ".png")
                        self.assertTrue(arrays_bit_equal(value, read_array(path)))

    def test_fallback_preserves_unrepresentable_arrays_and_explicit_numpy_policy(self):
        arrays = [np.array([[True, False]], np.bool_), np.ones((2, 3), np.float64),
                  np.array([[1., -0.]], ">f4"), np.ones((2, 3, 5), np.float32),
                  np.arange(4, dtype=np.float32), np.ones((2, 0), np.uint8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, value in enumerate(arrays):
                path = write_array(root / f"fallback-{index}", value, storage="images")
                self.assertEqual(path.suffix, ".npz")
                self.assertTrue(arrays_bit_equal(value, read_array(path)))
            value = np.ones((2, 3), np.float32)
            self.assertEqual(write_array(root / "legacy", value).suffix, ".npz")
            self.assertEqual(write_array(root / "uncompressed", value, storage="images", compressed=False).suffix, ".npy")
            with self.assertRaisesRegex(ValueError, "numpy or images"):
                write_array(root / "invalid", value, storage="half")

    def test_readers_reject_image_metadata_conflicts_and_lossy_exr(self):
        value = np.ones((2, 3), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, metadata in (("shape", {"ipdeArrayShape": "[3,2]"}),
                                   ("dtype", {"ipdeArrayDtype": "<f2"}),
                                   ("invalid", {"ipdeArrayShape": "[true,3]"})):
                path = root / f"{name}.exr"
                write_exr(path, value, attributes=metadata)
                with self.assertRaisesRegex(FormatError, "metadata"):
                    read_array(path)
            path = root / "lossy.exr"
            with OpenEXR.File({"type": OpenEXR.scanlineimage, "compression": OpenEXR.DWAA_COMPRESSION}, {"Y": value}) as image:
                image.write(str(path))
            with self.assertRaisesRegex(FormatError, "lossless ZIP"):
                read_array(path)
            path = root / "badshape.png"
            write_png(path, np.ones((2, 3), np.uint16), attributes={"ipdeArrayShape": "[2,3,2]"})
            with self.assertRaisesRegex(FormatError, "metadata"):
                read_array(path)
            with self.assertRaisesRegex(DatasetError, "escapes"):
                _array_path(root, {"path": "../depth.exr"})

    def test_compaction_converts_depth_once_preserves_legacy_sources_and_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, samples = _dataset(root)
            # Give legacy native float32 teacher arrays the production codec.
            for sample in samples:
                old = read_array(source / sample["teacher"]["target"]["path"])
                native = old.byteswap().view(old.dtype.newbyteorder("="))
                record = array_record(source, source / f"native-{sample['id']}.npz", native)
                sample["teacher"]["target"] = sample["teacher"]["native_target"] = record
            (source / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1", "samples": samples}))
            before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            compact = root / "compact"
            report = compress_dataset(source, compact, workers=2)
            saved = load_dataset(compact, workers=2)
            for old, new in zip(_array_records(samples), _array_records(saved["samples"])):
                self.assertEqual((old["shape"], old["dtype"], old["array_sha256"]),
                                 (new["shape"], new["dtype"], new["array_sha256"]))
                self.assertTrue(arrays_bit_equal(read_array(source / old["path"]), read_array(compact / new["path"])))
            for sample in saved["samples"]:
                self.assertTrue(sample["teacher"]["target"]["path"].endswith(".exr"))
                self.assertEqual(sample["teacher"]["target"], sample["teacher"]["native_target"])
                self.assertTrue(sample["teacher"]["valid_mask"]["path"].endswith(".npz"))
                self.assertTrue(sample["rgb"]["path"].endswith(".png"))
                self.assertTrue(_load_sample(compact, sample, TrainingOptions())["valid"].any())
            self.assertEqual(report["storage_compaction"]["unique_arrays"], len(list((compact / "arrays").iterdir())))
            self.assertEqual(before, {path: path.read_bytes() for path in source.rglob("*") if path.is_file()})
            curate_dataset(compact, [sample["id"] for sample in saved["samples"]], root / "reviewed")
            compose_datasets([compact], root / "prepared", CollectionOptions(storage_mode="copy"))
            self.assertTrue(load_dataset(root / "reviewed"))
            self.assertTrue(load_dataset(root / "prepared"))
            compress_dataset(compact, root / "numpy-copy", storage="numpy")
            self.assertTrue(all(path.suffix == ".npz" for path in (root / "numpy-copy" / "arrays").iterdir()))


if __name__ == "__main__":
    unittest.main()
