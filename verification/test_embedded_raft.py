"""Native diagnostics must use the RAFT component of the selected checkpoint."""
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from ipde.spatial import RaftStereoError, RaftStereoOptions, run_raft_stereo


class FixtureRAFT(torch.nn.Module):
    configurations = []
    calls = []

    def __init__(self, configuration):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(-.5))
        self.configurations.append(deepcopy(vars(configuration)))

    def forward(self, left, right, *, iters, test_mode):
        self.calls.append((left.detach().clone(), right.detach().clone(), iters, test_mode))
        flow = self.offset.expand(left.shape[0], 1, *left.shape[-2:]).clone()
        return flow, flow


class FixturePadder:
    def __init__(self, shape, divis_by):
        self.shape = shape

    def pad(self, *values):
        return values

    def unpad(self, value):
        return value


def upstream_modules():
    core, raft, utils, padder = (ModuleType(name) for name in
        ("core", "core.raft_stereo", "core.utils", "core.utils.utils"))
    core.__path__, utils.__path__ = [], []
    raft.RAFTStereo, padder.InputPadder = FixtureRAFT, FixturePadder
    return {"core": core, "core.raft_stereo": raft, "core.utils": utils, "core.utils.utils": padder}


class EmbeddedRaftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ipde-embedded-raft-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "upstream"
        (self.root / "core").mkdir(parents=True)
        (self.root / "core/raft_stereo.py").write_text("# fixture source identity\n")
        (self.root / "models").mkdir()
        # A valid, deliberately different default exists so an accidental fallback
        # would run successfully but fail the selected-weight assertions below.
        torch.save({"offset": torch.tensor(-.5)}, self.root / "models/raftstereo-middlebury.pth")
        self.configuration = {
            "hidden_dims": [64, 96, 128], "corr_implementation": "reg", "shared_backbone": False,
            "corr_levels": 3, "corr_radius": 2, "n_downsample": 2, "context_norm": "group",
            "slow_fast_gru": False, "n_gru_layers": 3, "mixed_precision": False,
        }
        self.payload = {"schema": "ipde-display-depth-v1",
            "architecture": {"architecture": "stereo-display-query-v2",
                             "units": "relative_depth", "raft_configuration": self.configuration},
            "state_dict": {"raft.module.offset": torch.tensor(-2.25),
                           "decoder.bias": torch.tensor(99.0), "left.weight": torch.ones(3, 3)}}
        self.left = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        self.right = self.left[:, ::-1].copy()
        self.spatial = {"raft_stereo_ready": True, "left_camera": {"width": 8, "height": 4},
                        "principal_point_delta_x_pixels": 0.0,
                        "focal_length_pixels_for_depth": 10.0, "baseline_meters": .1}
        FixtureRAFT.calls.clear(); FixtureRAFT.configurations.clear()
        modules = patch.dict(sys.modules, upstream_modules())
        modules.start(); self.addCleanup(modules.stop)
        registration = patch("ipde.spatial.register_stereo_rows", side_effect=lambda left, right:
                             (right, np.ones(right.shape[:2], dtype=bool), {"applied": False}))
        registration.start(); self.addCleanup(registration.stop)

    def save(self, payload, name="renamed-realtime-iraftstereo_rvc.pth"):
        checkpoint = self.directory / name
        torch.save(payload, checkpoint)
        return checkpoint

    def run_checkpoint(self, checkpoint, *, member=None, allow=True):
        return run_raft_stereo(self.left, self.right, self.spatial, RaftStereoOptions(
            root=self.root, model=checkpoint, model_member=member, device="cpu",
            iterations=4, allow_display_checkpoint=allow))

    def test_actual_runner_uses_selected_embedded_weights_and_configuration(self):
        checkpoint = self.save(self.payload)
        originals = [array.copy() for array in (self.left, self.right)]
        result = self.run_checkpoint(checkpoint)
        self.assertEqual(FixtureRAFT.configurations, [self.configuration])
        self.assertEqual(len(FixtureRAFT.calls), 2)
        self.assertEqual([(call[2], call[3]) for call in FixtureRAFT.calls], [(4, True), (4, True)])
        np.testing.assert_array_equal(FixtureRAFT.calls[0][0][0].permute(1, 2, 0).numpy(), self.left.astype(np.float32))
        np.testing.assert_array_equal(FixtureRAFT.calls[0][1][0].permute(1, 2, 0).numpy(), self.right.astype(np.float32))
        np.testing.assert_array_equal(result.signed_flow_pixels, np.full((4, 8), -2.25, dtype=np.float32))
        np.testing.assert_array_equal(result.height_disparity_pixels, np.full((4, 8), 2.25, dtype=np.float32))
        np.testing.assert_array_equal(result.depth_meters, np.full((4, 8), np.float32(1) / np.float32(2.25), dtype=np.float32))
        self.assertFalse(result.support_mask[:, :3].any())
        self.assertTrue(result.support_mask[:, 3:].all())
        self.assertEqual(result.details["checkpoint_component"], "embedded_raft_correspondence")
        self.assertEqual(result.details["checkpoint_path"], str(checkpoint.resolve()))
        self.assertEqual(result.details["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(result.details["correlation_implementation"], "reg")
        self.assertEqual(result.details["iterations"], 4)
        for array, original in zip((self.left, self.right), originals):
            np.testing.assert_array_equal(array, original)

    def test_zip_member_uses_selected_component_instead_of_default_member(self):
        archive = self.directory / "selected-models.zip"
        member = "trained/chosen.pth"
        selected, stock = BytesIO(), BytesIO()
        torch.save(self.payload, selected)
        torch.save({"offset": torch.tensor(-.5)}, stock)
        with zipfile.ZipFile(archive, "w") as models:
            models.writestr(member, selected.getvalue())
            models.writestr("raftstereo-middlebury.pth", stock.getvalue())
        original = archive.read_bytes()
        result = self.run_checkpoint(archive, member=member)
        np.testing.assert_array_equal(result.signed_flow_pixels, -2.25)
        self.assertEqual(FixtureRAFT.configurations, [self.configuration])
        self.assertEqual(result.details["checkpoint_member"], member)
        self.assertEqual(result.details["checkpoint_sha256"], hashlib.sha256(selected.getvalue()).hexdigest())
        self.assertEqual(archive.read_bytes(), original)

    def test_missing_or_unknown_configuration_and_weights_never_fall_back(self):
        cases = {}
        missing_configuration = deepcopy(self.payload)
        missing_configuration["architecture"].pop("raft_configuration")
        cases["missing-configuration"] = missing_configuration
        unknown_configuration = deepcopy(self.payload)
        unknown_configuration["architecture"]["raft_configuration"]["unknown"] = True
        cases["unknown-configuration"] = unknown_configuration
        incomplete_configuration = deepcopy(self.payload)
        incomplete_configuration["architecture"]["raft_configuration"].pop("hidden_dims")
        cases["incomplete-configuration"] = incomplete_configuration
        no_raft_weights = deepcopy(self.payload)
        no_raft_weights["state_dict"].pop("raft.module.offset")
        cases["missing-weights"] = no_raft_weights
        unknown_raft_weights = deepcopy(self.payload)
        unknown_raft_weights["state_dict"]["raft.module.unknown"] = torch.tensor(10.)
        cases["unknown-weights"] = unknown_raft_weights
        for name, payload in cases.items():
            with self.subTest(case=name):
                checkpoint = self.save(payload, name + ".pth")
                FixtureRAFT.calls.clear()
                with self.assertRaises(RaftStereoError):
                    self.run_checkpoint(checkpoint)
                self.assertEqual(FixtureRAFT.calls, [])

    def test_display_checkpoint_requires_explicit_native_component_dispatch(self):
        checkpoint = self.save(self.payload)
        with self.assertRaisesRegex(RaftStereoError, "display-depth decoder"):
            self.run_checkpoint(checkpoint, allow=False)
        self.assertEqual(FixtureRAFT.calls, [])
        self.assertEqual(FixtureRAFT.configurations, [])


if __name__ == "__main__":
    unittest.main()
