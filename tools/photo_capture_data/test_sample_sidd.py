import io
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from sample_sidd import capture_pairs, select_variants, copy_stream, native_crop


def files(instance, frame="001"):
    prefix = "SRGB/train/Data/" + instance + "/" + instance[:4]
    return [{"rfilename": prefix + "_" + role + "_SRGB_" + frame + ".PNG"}
            for role in ["GT", "NOISY"]]


class CaptureDataTests(unittest.TestCase):
    def test_same_frame_pairing_and_physical_scene_group_cross_camera_iso_and_frame(self):
        source = files("0001_001_S6_00100_00060_3200_L") + files("0010_001_IP_03200_00020_5500_N", "145")
        pairs = capture_pairs(source)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["group"], pairs[1]["group"])
        self.assertIn("_145.PNG", pairs[1]["reference"]["rfilename"])
        self.assertNotEqual(pairs[0]["capture"]["camera"], pairs[1]["capture"]["camera"])
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            capture_pairs([source[1]])

    def test_sampling_does_not_drop_the_last_available_quality_when_asking_for_more(self):
        pairs = capture_pairs(files("0001_001_S6_00100_00060_3200_L") + files("0002_001_S6_03200_00020_3200_N"))
        chosen = select_variants(pairs, scenes=1, variants=3)
        self.assertEqual([int(x["capture"]["iso"]) for x in chosen], [100, 3200])
        with self.assertRaisesRegex(ValueError, "not available"):
            select_variants(pairs, 1, 1, scene_ids=["008"])

    def test_budget_hash_failure_and_existing_files_never_publish_bad_sources(self):
        data = b"camera-original"
        expected = {"size": len(data), "lfs": {"sha256": hashlib.sha256(data).hexdigest()}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input.PNG"
            with self.assertRaises(ValueError): copy_stream(io.BytesIO(data), path, expected, 2)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError): copy_stream(io.BytesIO(b"x"*len(data)), path, expected, 100)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".PNG.pending").exists())
            copy_stream(io.BytesIO(data), path, expected, 100)
            self.assertEqual(copy_stream(io.BytesIO(), path, expected, 100), 0)
            path.write_bytes(b"valuable previous result")
            with self.assertRaises(ValueError): copy_stream(io.BytesIO(data), path, expected, 100)
            self.assertEqual(path.read_bytes(), b"valuable previous result")

    def test_exact_native_coordinates_and_source_bytes_survive_crop_and_review(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder);a=p/"a.png";b=p/"b.png"
            pixels = bytes(i % 256 for i in range(75*55*3))
            Image.frombytes("RGB", (75,55), pixels).save(a)
            Image.frombytes("RGB", (75,55), bytes(255-x for x in pixels)).save(b)
            originals = [a.read_bytes(), b.read_bytes()]
            receipt = native_crop(a, b, p/"crop", size=32)
            self.assertEqual((receipt["x"],receipt["y"]), (21,11))
            with Image.open(a) as src, Image.open(p/"crop/input.png") as crop:
                self.assertEqual(crop.size, (32,32))
                self.assertEqual(crop.tobytes(), src.crop((21,11,53,43)).tobytes())
            self.assertEqual([a.read_bytes(), b.read_bytes()], originals)
            self.assertFalse(receipt["resampled"])
            with self.assertRaises(ValueError): native_crop(a,b,p/"too-large",size=1024)


if __name__ == "__main__":
    unittest.main()
