"""Dataset inspection exports preserve the original grid and floating bits."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.dataset import DatasetError
from ipde.dataset_review import export_sample
from ipde.formats import sha256_array
from ipde.trainer_cli import main
from verification import test_dataset_teachers as teacher_fixture


class DatasetDepthExportTests(unittest.TestCase):
    def _dataset(self, root, *, shape=(96, 128)):
        dataset, manifest = teacher_fixture.DatasetTeacherTests()._dataset(root)
        depth = np.linspace(.123456, 789.1234, np.prod(shape), dtype=np.float32).reshape(shape)
        depth.view(np.uint32).flat[:4] = [0x7FC01234, 0x80000000, 0x00000001, 0x7F800000]
        label = manifest["samples"][0]["display_teacher"]
        record = array_record(dataset, dataset / "inspection-depth", depth, compressed=False)
        label["target"] = record
        manifest["samples"][0]["teacher"]["target"] = record
        (dataset / "dataset.json").write_text(json.dumps(manifest))
        return dataset, manifest["samples"][0]["id"], depth

    def test_exr_and_npy_preserve_float_bits_and_dataset_files_without_rgb_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, identity, depth = self._dataset(root)
            before = {path.relative_to(dataset): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
            for suffix in (".exr", ".npy"):
                output = root / ("saved-depth" + suffix)
                with patch("ipde.dataset_review._verified_array", wraps=__import__("ipde.dataset_review", fromlist=["_verified_array"])._verified_array) as verified:
                    report = export_sample(dataset, identity, output)
                self.assertEqual(verified.call_count, 1)
                saved = read_array(output)
                self.assertEqual((saved.shape, saved.dtype, saved.tobytes()), (depth.shape, depth.dtype, depth.tobytes()))
                self.assertEqual(report["array_sha256"], sha256_array(depth))
                self.assertEqual(report["shape"], list(depth.shape))
                self.assertEqual(report["label"], "display_teacher")
            self.assertEqual(before, {path.relative_to(dataset): path.read_bytes() for path in dataset.rglob("*") if path.is_file()})

    def test_full_5712_by_4284_display_grid_is_saved_without_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, identity, depth = self._dataset(root, shape=(4284, 5712))
            output = root / "full-display.npy"
            report = export_sample(dataset, identity, output)
            self.assertEqual(report["shape"], [4284, 5712])
            self.assertEqual(sha256_array(read_array(output)), sha256_array(depth))

    def test_refuses_overwrite_dataset_paths_discarded_raw_and_unknown_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, identity, depth = self._dataset(root)
            output = root / "existing.npy"
            output.write_bytes(b"keep existing file")
            with self.assertRaisesRegex(DatasetError, "already exists"):
                export_sample(dataset, identity, output)
            self.assertEqual(output.read_bytes(), b"keep existing file")
            export_sample(dataset, identity, output, replace_existing=True)
            self.assertEqual(read_array(output).tobytes(), depth.tobytes())
            with self.assertRaisesRegex(DatasetError, "outside"):
                export_sample(dataset, identity, dataset / "forbidden.npy")
            with self.assertRaisesRegex(DatasetError, "EXR or NPY"):
                export_sample(dataset, identity, root / "depth.png")
            with self.assertRaisesRegex(DatasetError, "Unknown"):
                export_sample(dataset, "unknown", root / "unknown.npy")
            manifest = json.loads((dataset / "dataset.json").read_text())
            manifest["samples"][0]["original_teacher_provenance"] = {"payload_retained": False}
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DatasetError, "Regenerate"):
                export_sample(dataset, identity, root / "discarded.npy")

    def test_checksum_or_roundtrip_failure_cannot_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, identity, depth = self._dataset(root)
            output = root / "failed.npy"
            with patch("ipde.dataset_review.read_array", side_effect=[depth, depth + np.float32(1)]):
                with self.assertRaisesRegex(DatasetError, "round-trip"):
                    export_sample(dataset, identity, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".depth-export-*")))
            manifest = json.loads((dataset / "dataset.json").read_text())
            path = dataset / manifest["samples"][0]["display_teacher"]["target"]["path"]
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(DatasetError, "Cannot read|checksum"):
                export_sample(dataset, identity, output)
            self.assertFalse(output.exists())

    def test_json_cli_exports_without_inference_or_preview_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, identity, depth = self._dataset(root)
            output = root / "cli.npy"
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()), \
                    patch("ipde.dataset_review.preview_sample", side_effect=AssertionError("preview called")), \
                    patch("ipde.learned_depth.LearnedDepthPredictor", side_effect=AssertionError("inference called")):
                result = main(["--json", "export-sample", str(dataset), "--sample", identity, "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["output_path"], str(output.resolve()))
            self.assertEqual(read_array(output).tobytes(), depth.tobytes())


if __name__ == "__main__":
    unittest.main()
