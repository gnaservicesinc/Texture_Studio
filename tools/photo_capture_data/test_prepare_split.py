import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from sample_sidd import REPO, REVISION, sha, native_crop
from prepare_split import prepare


def review(root, captures):
    root.mkdir()
    pairs = []
    for instance, scene in captures:
        folder = root / (instance + "-001")
        folder.mkdir()
        files = {}
        for role, color in [("input", (92, 97, 101)), ("reference", (93, 97, 100))]:
            path = folder / (role + ".PNG")
            Image.new("RGB", (1030, 1026), color).save(path)
            files[role] = {"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}
        crop = native_crop(folder / "input.PNG", folder / "reference.PNG", folder / "native-1024")
        pairs.append({"id": instance + "-001", "frame": "001", "group": "sidd-scene-" + scene,
                      "capture": {"instance": instance, "scene": scene}, "crop": crop, "files": files})
    (root / "receipt.json").write_text(json.dumps({"schema": "photo-capture-review-v1", "dataset": REPO,
        "revision": REVISION, "diffuse_target_claim": False, "pairs": pairs}))
    return root


class DiagnosticSplitTests(unittest.TestCase):
    def test_variants_in_different_receipts_cannot_leak_into_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            a = review(p / "a", [("0001", "001"), ("0002", "008")])
            b = review(p / "b", [("0003", "008"), ("0004", "002")])
            before = (a / "receipt.json").read_bytes()
            result = prepare([a, b], ["sidd-scene-008"], ["sidd-scene-002"])
            self.assertEqual(result["counts"]["validation"], {"pairs": 2, "physical_scenes": 1})
            self.assertEqual({row["group"] for row in result["pairs"] if row["split"] == "train"}, {"sidd-scene-001"})
            self.assertFalse(result["ready_for_training"])
            self.assertEqual((a / "receipt.json").read_bytes(), before)

    def test_empty_training_duplicate_and_missing_or_overlapping_holdouts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = review(Path(temporary) / "a", [("0001", "001"), ("0002", "008")])
            for validation, test in [(["sidd-scene-001", "sidd-scene-008"], []),
                                     (["sidd-scene-999"], []), (["sidd-scene-008"], ["sidd-scene-008"])]:
                with self.assertRaises(ValueError): prepare([root], validation, test)
            with self.assertRaisesRegex(ValueError, "Duplicate capture"):
                prepare([root, root], ["sidd-scene-008"], [])

    def test_changed_original_and_changed_native_crop_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = review(Path(temporary) / "a", [("0001", "001"), ("0002", "008")])
            path = root / "0001-001/input.PNG"
            original = path.read_bytes()
            path.write_bytes(b"changed original")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare([root], ["sidd-scene-008"], [])
            path.write_bytes(original)
            Image.new("RGB", (1024, 1024), "red").save(root / "0001-001/native-1024/input.png")
            with self.assertRaisesRegex(ValueError, "crop no longer matches"):
                prepare([root], ["sidd-scene-008"], [])

    def test_false_scene_identity_and_external_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = review(Path(temporary) / "a", [("0001", "001"), ("0002", "008")])
            path = root / "receipt.json"
            data = json.loads(path.read_text())
            data["pairs"][0]["group"] = "sidd-scene-009"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                prepare([root], ["sidd-scene-008"], [])
            data["pairs"][0]["group"] = "sidd-scene-001"
            data["pairs"][0]["files"]["input"]["file"] = "../outside.PNG"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare([root], ["sidd-scene-008"], [])


if __name__ == "__main__":
    unittest.main()
