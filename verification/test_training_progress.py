"""Live progress reports real training updates without changing saved data."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from ipde.array_storage import array_record
from ipde import trainer_cli
from ipde.training import TrainingError, TrainingOptions, _SamplePool, _training_crop_origin, train_dataset, training_target_eligibility


def _dataset(root: Path, count: int = 2, *, height: int = 64, width: int = 96) -> Path:
    destination = root / "dataset"
    destination.mkdir()
    samples = []
    for index in range(count):
        left = np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), 2., np.float32)
        depth[0, 0] = np.nan
        rgb = array_record(destination, destination / f"rgb-{index}.npy", left, compressed=False)
        right = array_record(destination, destination / f"right-{index}.npy", np.roll(left, -2, axis=1), compressed=False)
        target = array_record(destination, destination / f"depth-{index}.npy", depth, compressed=False)
        mask = array_record(destination, destination / f"mask-{index}.npy", np.isfinite(depth), compressed=False)
        raw = array_record(destination, destination / f"raw-{index}.npy",
            np.array([[0x7e55, 0x8000, 0x3555]], np.uint16).view(np.float16), compressed=False)
        samples.append({"id": f"sample-{index}", "source_path": f"photo-{index}.HEIC", "source_sha256": f"source-{index}",
            "group_id": f"scene-{index}", "split": "train" if index % 2 == 0 else "validation",
            "rgb": rgb, "right_rgb": right, "raw_assets": [{"storage": raw}],
            "calibration": {"raft_stereo_ready": True, "rectified_stereo_ready": True,
                "left_camera": {"width": width, "height": height, "focal_length_x_pixels": 100.},
                "right_camera": {"width": width, "height": height, "focal_length_x_pixels": 100.},
                "focal_length_pixels_for_depth": 100., "baseline_meters": .04,
                "principal_point_delta_x_pixels": .5, "left_image_index": 1, "right_image_index": 2},
            "teacher": {"units": "meters", "target": target, "native_target": target,
                "valid_mask": mask, "metadata": {"checkpoint_sha256": "a" * 64}}})
    (destination / "dataset.json").write_text(json.dumps({"schema": "ipde-depth-dataset-v1",
        "samples": samples, "group_ids": [f"scene-{index}" for index in range(count)], "explicit_scene_groups": True}))
    return destination


def _fake_raft_modules(torch):
    # PyTorch registers process-wide dispatcher libraries during this lazy
    # import. Load them before patch.dict restores the module inventory, so a
    # second test never tries to register the same libraries again.
    import torch._dynamo
    # Exercise our actual optimizer and checkpoint code with a tiny, real,
    # differentiable model; do not load or run the user's pretrained model.
    class FixtureRAFT(torch.nn.Module):
        def __init__(self, configuration):
            super().__init__()
            self.update_block = torch.nn.Conv2d(6, 1, 1)
            with torch.no_grad():
                self.update_block.weight.zero_()
                self.update_block.bias.zero_()

        def freeze_bn(self):
            pass

        def forward(self, left, right, iters=1, test_mode=False):
            prediction = self.update_block(torch.cat((left, right), 1) / 255)
            return (prediction, prediction) if test_mode else [prediction] * iters

    class FixturePadder:
        def __init__(self, shape, divis_by):
            pass

        def pad(self, *values):
            return values

        def unpad(self, value):
            return value

    core, raft, utils, padder = (ModuleType(name) for name in
        ("core", "core.raft_stereo", "core.utils", "core.utils.utils"))
    core.__path__, utils.__path__ = [], []
    raft.RAFTStereo, padder.InputPadder = FixtureRAFT, FixturePadder
    return FixtureRAFT, {"core": core, "core.raft_stereo": raft, "core.utils": utils, "core.utils.utils": padder}


class TrainingProgressTests(unittest.TestCase):
    def test_metadata_eligibility_honors_selected_labels_registration_and_mode(self):
        teacher = {"units": "meters", "metadata": {"checkpoint_sha256": "a" * 64}}
        manifest = {"samples": [
            {"id": "good", "split": "train", "teacher": teacher},
            {"id": "relative", "split": "train", "teacher": {**teacher, "units": "relative_depth"},
                "metric_anchor": teacher, "pseudo_calibration": {"accepted": False}},
            {"id": "right", "split": "validation", "teacher": teacher,
                "training_target_choice": "registered_display_teacher",
                "registered_display_teacher": {**teacher, "reference_role": "right"}},
            {"id": "measured", "split": "validation", "training_target_choice": "reference",
                "reference": {"units": "meters"}},
        ]}
        original = json.dumps(manifest, sort_keys=True)
        with patch("ipde.training.read_array", side_effect=AssertionError("metadata inspection must not decode arrays")), \
             patch("ipde.training.load_dataset", side_effect=AssertionError("metadata inspection must not load files")):
            summary = training_target_eligibility(manifest)
            supervised = training_target_eligibility(manifest, mode="supervised")
            distillation = training_target_eligibility(manifest, mode="distillation")
        self.assertEqual(summary["mode"], "mixed")
        self.assertEqual((summary["sample_count"], summary["eligible_count"], summary["excluded_count"]), (4, 2, 2))
        self.assertEqual((summary["train_count"], summary["validation_count"]), (1, 1))
        self.assertEqual([item["sample_id"] for item in summary["excluded"]], ["relative", "right"])
        self.assertEqual((supervised["eligible_count"], distillation["eligible_count"]), (1, 1))
        self.assertEqual(json.dumps(manifest, sort_keys=True), original)

    def test_six_rejected_relative_targets_skip_without_editing_or_changing_anchor(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=8)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            for index, sample in enumerate(manifest["samples"]):
                if index < 2:
                    sample["training_target_choice"] = "anchored_teacher"
                    sample["anchored_teacher"] = {**sample["teacher"], "metadata": {
                        "checkpoint_sha256": "a" * 64, "metric_anchor_checkpoint_sha256": "b" * 64}}
                else:
                    sample["teacher"]["units"] = "relative_depth"
                    # Another meter label is deliberately present. A rejected
                    # anchor never authorizes substituting this target.
                    sample["metric_anchor"] = {**sample["teacher"], "units": "meters",
                        "metadata": {"checkpoint_sha256": "c" * 64}}
                    sample["pseudo_calibration"] = {"accepted": False}
            manifest_path.write_text(json.dumps(manifest))
            original_data = {path: path.read_bytes() for path in dataset.iterdir()}
            summary = training_target_eligibility(manifest)
            self.assertEqual((summary["eligible_count"], summary["excluded_count"]), (2, 6))
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)
            from ipde.training import _load_sample
            events = []
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), \
                 patch("ipde.training._load_sample", wraps=_load_sample) as loading, redirect_stderr(io.StringIO()):
                report = train_dataset(dataset, root / "filtered.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu", mode="auto"),
                    progress_callback=events.append)
            self.assertEqual([call.args[1]["id"] for call in loading.call_args_list], ["sample-0", "sample-1"])
            self.assertEqual((report["sample_count"], report["eligible_count"], report["excluded_count"]), (8, 2, 6))
            self.assertEqual(report["eligible_sample_ids"], ["sample-0", "sample-1"])
            self.assertEqual([item["sample_id"] for item in report["excluded_samples"]], [f"sample-{index}" for index in range(2, 8)])
            self.assertEqual(report["metric_anchor_checkpoint_sha256"], ["b" * 64])
            self.assertTrue(all(item["target_choice"] == "anchored_teacher" for item in report["label_provenance"]))
            self.assertEqual(sum(event["stage"] == "skipped_sample" for event in events), 6)
            self.assertEqual(original_data, {path: path.read_bytes() for path in dataset.iterdir()})
            self.assertTrue((root / "filtered.pth").is_file())

    def test_geometry_and_support_failures_skip_with_actual_splits_in_checkpoint(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=6)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"][4]["calibration"]["baseline_meters"] = -1.
            sample = manifest["samples"][5]
            sample["teacher"]["valid_mask"] = array_record(dataset,
                dataset / sample["teacher"]["valid_mask"]["path"], np.zeros((64, 96), bool), compressed=False)
            manifest_path.write_text(json.dumps(manifest))
            original_data = {path: path.read_bytes() for path in dataset.iterdir()}
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), redirect_stderr(io.StringIO()):
                report = train_dataset(dataset, root / "usable.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu"))
            self.assertEqual((report["train_count"], report["validation_count"], report["excluded_count"]), (2, 2, 2))
            self.assertEqual(report["train_sample_ids"], ["sample-0", "sample-2"])
            self.assertEqual(report["validation_sample_ids"], ["sample-1", "sample-3"])
            self.assertEqual([item["sample_id"] for item in report["excluded_samples"]], ["sample-4", "sample-5"])
            self.assertIn("calibration/grid", report["excluded_samples"][0]["reason"])
            self.assertIn("no visible", report["excluded_samples"][1]["reason"])
            self.assertEqual(original_data, {path: path.read_bytes() for path in dataset.iterdir()})
            saved = torch.load(root / "usable.pth", weights_only=True)
            self.assertEqual(saved["ipde_training"]["eligible_sample_ids"], [f"sample-{index}" for index in range(4)])

    def test_preparation_remaps_bounded_cache_without_extra_decodes(self):
        samples = [{"id": f"sample-{index}", "teacher": {"units": "meters"}} for index in range(6)]
        pool = _SamplePool(Path("unused"), samples, TrainingOptions(cache_samples=2))
        def loading(_root, sample, _options):
            if sample["id"] in {"sample-1", "sample-3"}:
                raise TrainingError("no visible correspondence")
            return {"details": {"sample_id": sample["id"]}, "array": np.ones((1, 1)),
                "flow": np.zeros((8, 8), np.float32), "valid": np.ones((8, 8), bool),
                "left": np.zeros((8, 8, 3), np.uint8), "right": np.zeros((8, 8, 3), np.uint8)}
        with patch("ipde.training._load_sample", side_effect=loading) as loading_mock, redirect_stderr(io.StringIO()):
            excluded = pool.prepare("training")
            self.assertEqual([sample["id"] for sample in pool.samples], ["sample-0", "sample-2", "sample-4", "sample-5"])
            self.assertEqual(list(pool.cache), [2, 3])
            self.assertEqual(sorted(pool.details), [0, 1, 2, 3])
            self.assertEqual(sorted(pool.crop_origins), [0, 1, 2, 3])
            self.assertIs(pool[2], pool.cache[2])
            self.assertIs(pool[3], pool.cache[3])
            self.assertEqual(loading_mock.call_count, 6)
        self.assertEqual([item["sample_id"] for item in excluded], ["sample-1", "sample-3"])

    def test_invalid_targets_do_not_hide_integrity_or_group_leakage_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["samples"][1]["teacher"]["units"] = "relative_depth"
            manifest["samples"][1]["group_id"] = manifest["samples"][0]["group_id"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TrainingError, "leakage"):
                train_dataset(dataset, root / "leaked.pth")
            manifest["samples"][1]["group_id"] = "scene-1"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TrainingError, "No usable training and held-out validation split"):
                train_dataset(dataset, root / "empty-split.pth")
            raw_path = dataset / manifest["samples"][1]["raw_assets"][0]["storage"]["path"]
            raw_path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(TrainingError, "Cannot read dataset array"):
                train_dataset(dataset, root / "corrupt.pth")

    def test_checkpoint_destination_cannot_write_into_input_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            original = {path: path.read_bytes() for path in dataset.iterdir()}
            with self.assertRaisesRegex(TrainingError, "outside the input dataset"):
                train_dataset(dataset, dataset / "model.pth")
            self.assertEqual(original, {path: path.read_bytes() for path in dataset.iterdir()})

    def test_full_image_correspondences_without_usable_requested_crops_are_skipped(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, count=4, width=192)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            for sample in manifest["samples"][2:]:
                depth = np.full((64, 192), np.float32(1 / 24), np.float32)
                target = array_record(dataset, dataset / sample["teacher"]["target"]["path"], depth, compressed=False)
                sample["teacher"]["target"] = sample["teacher"]["native_target"] = target
            manifest_path.write_text(json.dumps(manifest))
            original_data = {path: path.read_bytes() for path in dataset.iterdir()}
            from ipde.training import _load_sample
            bad = _load_sample(dataset, manifest["samples"][2], TrainingOptions(patch_size=64))
            self.assertTrue(bad["valid"].any())
            self.assertIsNone(_training_crop_origin(bad, 64))
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), redirect_stderr(io.StringIO()):
                report = train_dataset(dataset, root / "usable-crops.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu"))
            self.assertEqual((report["train_count"], report["validation_count"], report["excluded_count"]), (1, 1, 2))
            self.assertEqual(report["eligible_sample_ids"], ["sample-0", "sample-1"])
            self.assertIn("64-pixel training crop", report["excluded_samples"][0]["reason"])
            self.assertIn("Fixed validation crops", report["excluded_samples"][1]["reason"])
            self.assertEqual(report["total_steps"], 1)
            self.assertEqual(original_data, {path: path.read_bytes() for path in dataset.iterdir()})

    def test_sparse_usable_training_crop_survives_exhausted_random_draws(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root, height=256, width=256)
            manifest_path = dataset / "dataset.json"
            manifest = json.loads(manifest_path.read_text())
            sample = manifest["samples"][0]
            sparse = np.zeros((256, 256), bool)
            sparse[128, 128] = True
            sample["teacher"]["valid_mask"] = array_record(dataset,
                dataset / sample["teacher"]["valid_mask"]["path"], sparse, compressed=False)
            manifest_path.write_text(json.dumps(manifest))
            original_data = {path: path.read_bytes() for path in dataset.iterdir()}
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)

            class MissEveryRandomCrop:
                calls = 0
                def integers(self, _maximum):
                    self.calls += 1
                    return 0
            misses = MissEveryRandomCrop()
            actual_factory = np.random.default_rng
            def random_factory(seed):
                return misses if seed == 99 else actual_factory(seed)
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), \
                 patch("ipde.training.np.random.default_rng", side_effect=random_factory), redirect_stderr(io.StringIO()):
                report = train_dataset(dataset, root / "sparse.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu", seed=99))
            self.assertEqual(misses.calls, 1 + 2 * 128)
            self.assertEqual((report["total_steps"], report["fallback_training_crops"], report["excluded_count"]), (1, 1, 0))
            self.assertEqual(report["train_sample_ids"], ["sample-0"])
            self.assertTrue((root / "sparse.pth").is_file())
            self.assertEqual(original_data, {path: path.read_bytes() for path in dataset.iterdir()})

    def test_training_crop_support_applies_loss_limit_while_validation_keeps_its_scope(self):
        flow = np.full((8, 1024), -700., np.float32)
        valid = np.zeros(flow.shape, bool)
        valid[:, 700:] = True
        value = {"details": {"sample_id": "wide"}, "flow": flow, "valid": valid,
            "left": np.zeros((8, 1024, 3), np.uint8), "right": np.zeros((8, 1024, 3), np.uint8)}
        sample = {"id": "wide", "teacher": {"units": "meters"}}
        with patch("ipde.training._load_sample", return_value=value), redirect_stderr(io.StringIO()):
            training = _SamplePool(Path("unused"), [sample], TrainingOptions(patch_size=736))
            validation = _SamplePool(Path("unused"), [sample], TrainingOptions(patch_size=736))
            excluded = training.prepare("training")
            self.assertEqual(len(training), 0)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(validation.prepare("validation"), [])
            self.assertEqual(len(validation), 1)

    def test_real_updates_stages_and_lossless_data_match_without_callback(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            original_data = {path: path.read_bytes() for path in dataset.iterdir()}
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)
            options = TrainingOptions(epochs=2, steps_per_epoch=2, patch_size=64,
                iterations=1, learning_rate=1e-3, device="cpu", raft_root=root, raft_model=original)
            events, logs = [], io.StringIO()
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), \
                 redirect_stderr(logs):
                report = train_dataset(dataset, root / "progress.pth", options, progress_callback=events.append)
                reference = train_dataset(dataset, root / "reference.pth", options)

            self.assertEqual({path: path.read_bytes() for path in dataset.iterdir()}, original_data)
            payload = torch.load(root / "progress.pth", map_location="cpu", weights_only=True)
            comparison = torch.load(root / "reference.pth", map_location="cpu", weights_only=True)
            for key, value in payload["state_dict"].items():
                self.assertTrue(torch.equal(value, comparison["state_dict"][key]))
            self.assertEqual(report["history"], reference["history"])
            self.assertEqual(report["validation"], reference["validation"])
            self.assertEqual(report["total_steps"], 4)
            self.assertEqual(report["epochs_completed"], 2)
            self.assertIn("Epoch 1/2: loss", logs.getvalue())
            self.assertIn("Epoch 2/2: loss", logs.getvalue())
            self.assertTrue(all(event["phase"] == "training_progress" for event in events))
            self.assertTrue(all(event["total_steps"] == 4 for event in events))
            json.dumps(events, allow_nan=False)
            stages = list(dict.fromkeys(event["stage"] for event in events))
            self.assertEqual(stages, ["checking_dataset", "filtering_targets", "preparing_targets", "model_setup",
                "baseline_validation", "epoch_step", "epoch_validation", "final_validation", "writing_checkpoint", "completed"])
            started = [event for event in events if event["stage"] == "epoch_step" and event["status"] == "started"]
            finished = [event for event in events if event["stage"] == "epoch_step" and event["status"] == "finished"]
            self.assertEqual([(event["epoch"], event["step"], event["completed_steps"]) for event in started],
                [(1, 1, 0), (1, 2, 1), (2, 1, 2), (2, 2, 3)])
            self.assertEqual([event["completed_steps"] for event in finished], [1, 2, 3, 4])
            self.assertTrue(all(np.isfinite(event["loss"]) for event in finished))
            prepared = [event for event in events if event["stage"] == "preparing_targets"]
            self.assertEqual((prepared[0]["processed"], prepared[-1]["processed"], prepared[-1]["total"]), (0, 2, 2))
            validations = [event for event in events if event["stage"] in {"baseline_validation", "epoch_validation", "final_validation"}
                and event["status"] == "finished"]
            self.assertEqual(len(validations), 4)
            self.assertTrue(all(event["evaluated_patches"] == 3 for event in validations))
            self.assertEqual(events[-1]["completed_steps"], 4)
            self.assertEqual(events[-1]["checkpoint_path"], str((root / "progress.pth").resolve()))
            self.assertTrue((root / "progress.pth.json").is_file())

    def test_failed_step_is_never_counted_or_published_as_completed(self):
        import torch
        fixture, modules = _fake_raft_modules(torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            original = root / "original.pth"
            torch.save(fixture(None).state_dict(), original)
            events = []
            with patch.dict(sys.modules, modules), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)), \
                 patch("ipde.training.sequence_loss", return_value=torch.tensor(float("nan"))), \
                 redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(TrainingError, "loss became nonfinite"):
                    train_dataset(dataset, root / "failed.pth", TrainingOptions(
                        epochs=1, steps_per_epoch=1, patch_size=64, iterations=1, device="cpu"),
                        progress_callback=events.append)
            self.assertEqual(events[-1]["stage"], "epoch_step")
            self.assertEqual(events[-1]["status"], "started")
            self.assertTrue(all(event["completed_steps"] == 0 for event in events))
            self.assertFalse((root / "failed.pth").exists())
            self.assertFalse((root / "failed.pth.json").exists())

    def test_checking_stage_precedes_expensive_verification_and_failure_never_completes(self):
        events = []
        def fail_loading(_directory, *, validate_files=True):
            self.assertEqual(events[0]["stage"], "checking_dataset")
            self.assertEqual(events[0]["completed_steps"], 0)
            raise TrainingError("invalid dataset")
        with patch("ipde.dataset_review._read_manifest_snapshot", side_effect=fail_loading):
            with self.assertRaisesRegex(TrainingError, "invalid dataset"):
                train_dataset("unused-dataset", "unused-checkpoint.pth", progress_callback=events.append)
        self.assertEqual(len(events), 1)

    def test_cli_streams_callback_events_to_stderr_and_preserves_json_response(self):
        event = {"phase": "training_progress", "stage": "epoch_step", "epoch": 2,
            "epochs": 3, "step": 1, "steps_per_epoch": 4, "completed_steps": 5,
            "total_steps": 12, "status": "finished", "loss": .25}
        stdout, stderr = io.StringIO(), io.StringIO()
        def training(_dataset, _checkpoint, _options, *, progress_callback):
            progress_callback(event)
            return {"total_steps": 12, "best_epoch": 2}
        with patch("ipde.training.train_dataset", side_effect=training), redirect_stdout(stdout), redirect_stderr(stderr):
            result = trainer_cli.main(["--json", "train", "dataset", "--checkpoint", "output.pth"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"total_steps": 12, "best_epoch": 2})
        self.assertEqual(json.loads(stderr.getvalue().removeprefix("IPDE_EVENT ")), event)


if __name__ == "__main__":
    unittest.main()
