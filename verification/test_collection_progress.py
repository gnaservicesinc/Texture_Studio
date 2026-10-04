"""Composition progress must stay visible without changing scientific arrays."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ipde.array_storage import array_record, read_array
from ipde.dataset_collection import compose_datasets
from ipde.dataset_review import _array_records


class CollectionProgressTests(unittest.TestCase):
    def test_progress_counts_and_lossless_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            samples = []
            for index in range(3):
                rgb = array_record(source, source / f"rgb-{index}.npy", np.full((2, 3, 3), index, dtype=np.uint16), compressed=False)
                # Preserve negative zero, NaN, and float precision in targets.
                target = array_record(source, source / f"target-{index}.npy", np.array([[-0., np.nan, 1.234567], [2., 3., 4.]], dtype=np.float32), compressed=False)
                mask = array_record(source, source / f"mask-{index}.npy", np.array([[True, False, True], [True, True, True]]), compressed=False)
                samples.append({"id": f"sample-{index}", "source_path": f"photo-{index}.HEIC", "source_sha256": f"source-{index}", "group_id": f"group-{index}",
                                "split": "validation" if index == 0 else "train", "rgb": rgb, "right_rgb": rgb, "raw_assets": [],
                                "teacher": {"units": "meters", "target": target, "native_target": target, "valid_mask": mask}})
            manifest = {"schema": "ipde-depth-dataset-v1", "samples": samples}
            (source / "dataset.json").write_text(json.dumps(manifest))
            original = {file: file.read_bytes() for file in source.iterdir()}
            events = []
            output = root / "training-set"
            report = compose_datasets([source], output, progress_callback=events.append)
            self.assertEqual(events[0]["phase"], "verifying_dataset")
            self.assertEqual((events[0]["processed"], events[0]["total"]), (1, 1))
            copied = [event for event in events if event["phase"] == "sample_composed"]
            self.assertEqual([(event["processed"], event["total"]) for event in copied], [(1, 3), (2, 3), (3, 3)])
            self.assertEqual([event["phase"] for event in events[-2:]], ["verifying_output", "publishing_dataset"])
            self.assertEqual(report["summary"]["samples"], 3)
            saved = json.loads((output / "dataset.json").read_text())
            for sample in saved["samples"]:
                source_sample = next(item for item in samples if item["id"] == sample["collection_provenance"]["source_sample_id"])
                for before, after in zip(_array_records(source_sample), _array_records(sample)):
                    self.assertEqual(before["array_sha256"], after["array_sha256"])
                    self.assertEqual(read_array(source / before["path"]).tobytes(), read_array(output / after["path"]).tobytes())
            self.assertEqual(original, {file: file.read_bytes() for file in source.iterdir()})


if __name__ == "__main__":
    unittest.main()
