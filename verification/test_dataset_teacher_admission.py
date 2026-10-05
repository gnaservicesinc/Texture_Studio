"""Dataset admission refuses expensive teachers before touching any payload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipde.dataset import DatasetError, DatasetOptions, build_dataset, require_dataset_teacher
from ipde.dataset_teachers import generate_teacher
from ipde.learned_depth import LearnedDepthConfig


class DatasetTeacherAdmissionTests(unittest.TestCase):
    def test_new_dataset_refuses_primary_and_additional_depth_anything_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.heic"
            source.write_bytes(b"unchanged original")
            for model in ("depth-anything-3", "depth-anything-v2"):
                for options in (DatasetOptions(teacher=LearnedDepthConfig(model=model)),
                                DatasetOptions(teacher=LearnedDepthConfig(), additional_teachers=(LearnedDepthConfig(model=model),))):
                    with patch("ipde.dataset.discover_file") as discover, patch("ipde.dataset._build_dataset") as build:
                        with self.assertRaisesRegex(DatasetError, "temporarily disabled"):
                            build_dataset([source], root / "dataset", options)
                        discover.assert_not_called()
                        build.assert_not_called()
                    self.assertFalse((root / "dataset").exists())
                    self.assertEqual(source.read_bytes(), b"unchanged original")

    def test_existing_dataset_refuses_depth_anything_generation_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "dataset.json"
            manifest.write_text(json.dumps({"samples": [{"id": "existing"}]}))
            raw = root / "depth.npy"
            raw.write_bytes(b"existing scientific payload")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            for model in ("depth-anything-3", "depth-anything-v2"):
                with patch("ipde.learned_depth.LearnedDepthPredictor") as predictor:
                    with self.assertRaisesRegex(DatasetError, "existing dataset maps remain usable"):
                        generate_teacher(root, ["existing"], LearnedDepthConfig(model=model), regenerate=True)
                    predictor.assert_not_called()
                self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_depthpro_dataset_generation_remains_available(self):
        require_dataset_teacher("depthpro")
        with tempfile.TemporaryDirectory() as directory:
            with patch("ipde.dataset._build_dataset", return_value={"ok": True}) as build:
                self.assertEqual(build_dataset(["photo.heic"], Path(directory) / "dataset", DatasetOptions(teacher=LearnedDepthConfig())), {"ok": True})
                build.assert_called_once()
