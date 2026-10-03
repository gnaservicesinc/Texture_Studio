from __future__ import annotations

import tempfile
import sys
import json
from types import ModuleType
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ipde.spatial import RaftStereoError, _model_configuration, checkpoint_model_configuration
from ipde.training import TrainingError, TrainingOptions, _check_splits, _patch, export_raft_checkpoint, sequence_loss, train_dataset
from test_dataset import build_fixture


class TrainingTests(unittest.TestCase):
    def test_consistency_gate_checks_the_actually_selected_display_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest, _ = build_fixture(root)
            selected = manifest["samples"][0]
            selected["training_target_choice"] = "registered_display_teacher"
            selected["registered_display_teacher"] = {
                **selected["teacher"], "reference_role": "left",
                "metadata": {**selected["teacher"]["metadata"], "checkpoint_sha256": "b" * 64},
            }
            (dataset / "dataset.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TrainingError, "consistent teacher checkpoint"), \
                 patch("ipde.training.resolve_raft_resources") as resources:
                train_dataset(dataset, root / "unused.pth", TrainingOptions())
            resources.assert_not_called()

    def test_one_iteration_loss_masks_nan_and_backpropagates_correct_signed_target(self) -> None:
        import torch
        prediction = torch.zeros((1, 1, 2, 2), requires_grad=True)
        target = torch.tensor([[[[-2.0, float("nan")], [-3.0, 999.0]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        loss = sequence_loss(torch, [prediction], target, valid)
        self.assertEqual(float(loss.detach()), 2.5)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 0, 0, 0]), .5)
        self.assertEqual(float(prediction.grad[0, 0, 0, 1]), 0)
        self.assertEqual(float(prediction.grad[0, 0, 1, 1]), 0)

    def test_crop_preserves_flow_but_excludes_out_of_crop_correspondence(self) -> None:
        flow = np.full((64, 128), -3.0, np.float32)
        sample = {"flow": flow, "valid": np.ones(flow.shape, bool),
            "left": np.zeros((*flow.shape, 3), np.uint8), "right": np.zeros((*flow.shape, 3), np.uint8)}
        _, _, cropped_flow, valid = _patch(sample, 0, 32, 64)
        self.assertFalse(valid[:, :3].any())
        self.assertTrue(valid[:, 3:].all())
        self.assertTrue(np.all(cropped_flow[:, 3:] == -3))
        self.assertTrue(np.all(flow == -3))

    def test_rejects_same_scene_split_leakage_and_single_group(self) -> None:
        samples = [
            {"split": "train", "group_id": "a", "source_sha256": "a", "requested_group": "room", "rgb": {"array_sha256": "a"}},
            {"split": "validation", "group_id": "b", "source_sha256": "b", "requested_group": "room", "rgb": {"array_sha256": "b"}},
        ]
        with self.assertRaisesRegex(TrainingError, "leakage"):
            _check_splits(samples)
        with self.assertRaisesRegex(TrainingError, "held-out"):
            _check_splits(samples[:1])

    def test_checkpoint_architecture_survives_arbitrary_filename_and_rejects_untrusted_settings(self) -> None:
        configuration = vars(_model_configuration("realtime.pth"))
        restored = checkpoint_model_configuration({"ipde_configuration": configuration}, "renamed.pth")
        self.assertTrue(restored.shared_backbone)
        self.assertEqual(restored.n_downsample, 3)
        for updates in ({"mixed_precision": True}, {"n_downsample": 999}, {"hidden_dims": [128]}, {"corr_implementation": "alt_cuda"}):
            with self.assertRaises(RaftStereoError):
                checkpoint_model_configuration({"ipde_configuration": {**configuration, **updates}}, "anything.pth")
        with self.assertRaises(RaftStereoError):
            checkpoint_model_configuration({"ipde_configuration": {**configuration, "arbitrary": "ignored"}}, "x.pth")

    def test_real_optimizer_loop_saves_reloadable_raft_weights_and_experimental_report(self) -> None:
        import torch
        # Tiny differentiable stand-in isolates our optimizer/masking/checkpoint
        # logic from upstream GPU cost. The actual RAFT model is exercised by the
        # local two-photo experiment, not by substituting this test model in apps.
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

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _, _ = build_fixture(root)
            original = root / "original.pth"
            torch.save(FixtureRAFT(None).state_dict(), original)
            class FixturePadder:
                def __init__(self, shape, divis_by):
                    pass
                def pad(self, *values):
                    return values
                def unpad(self, value):
                    return value
            core = ModuleType("core")
            core.__path__ = []
            raft_module = ModuleType("core.raft_stereo")
            raft_module.RAFTStereo = FixtureRAFT
            utils = ModuleType("core.utils")
            utils.__path__ = []
            utils_module = ModuleType("core.utils.utils")
            utils_module.InputPadder = FixturePadder
            with patch.dict(sys.modules, {"core": core, "core.raft_stereo": raft_module, "core.utils": utils, "core.utils.utils": utils_module}), \
                 patch("ipde.training.resolve_raft_resources", return_value=(root, original, None)):
                report = train_dataset(dataset, root / "trained.pth", TrainingOptions(
                    epochs=1, steps_per_epoch=2, iterations=1, learning_rate=1e-3,
                    device="cpu", raft_model=original, raft_root=root, patch_size=64))
                exported = export_raft_checkpoint(root / "trained.pth", root / "export", raft_root=root)
                self.assertFalse(exported["source_photos_included"])
                self.assertFalse(exported["installed_as_default"])
                self.assertTrue((root / "export" / "raft-model.pth").is_file())
                exported_payload = torch.load(root / "export" / "raft-model.pth", weights_only=True)
                self.assertNotIn("options", exported_payload["ipde_training"])
                with self.assertRaisesRegex(TrainingError, "already exists"):
                    export_raft_checkpoint(root / "trained.pth", root / "export", raft_root=root)
            checkpoint = torch.load(root / "trained.pth", map_location="cpu", weights_only=True)
            self.assertIn("state_dict", checkpoint)
            self.assertIn("ipde_configuration", checkpoint)
            restored = FixtureRAFT(None)
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            self.assertEqual(report["model"], "RAFT-Stereo")
            self.assertEqual(report["status"], "experimental")
            self.assertEqual(report["best_epoch"], 1)
            self.assertTrue(set(report["train_group_ids"]).isdisjoint(report["validation_group_ids"]))
            self.assertTrue((root / "trained.pth.json").is_file())
            self.assertLess(report["validation"]["mean_absolute_flow_error_pixels"], report["baseline_validation"]["mean_absolute_flow_error_pixels"])


if __name__ == "__main__":
    unittest.main()
