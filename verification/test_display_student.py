"""Exercise the real display decoder with a tiny differentiable stereo fixture."""
from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from ipde import display_student as student


class FixtureRAFT(torch.nn.Module):
    """Use both cameras while keeping the real student decoder inexpensive."""
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __init__(self, configuration):
        super().__init__()
        self.update_block = torch.nn.Conv2d(6, 1, 1)
        with torch.no_grad():
            self.update_block.weight[:, :3].fill_(0.2)
            self.update_block.weight[:, 3:].fill_(-0.1)
            self.update_block.bias.fill_(0.05)

    def freeze_bn(self):
        pass

    def forward(self, left, right, iters=1, test_mode=False):
        self.calls.append((left.detach().clone(), right.detach().clone()))
        result = self.update_block(torch.cat((left, right), dim=1) / 255)
        return (result, result) if test_mode else [result] * iters


class FixturePadder:
    def __init__(self, shape, divis_by=32):
        height, width = shape[-2:]
        extra_h, extra_w = (-height) % divis_by, (-width) % divis_by
        self.padding = (extra_w // 2, extra_w - extra_w // 2,
                        extra_h // 2, extra_h - extra_h // 2)

    def pad(self, *values):
        return [F.pad(value, self.padding, mode="replicate") for value in values]

    def unpad(self, value):
        left, right, top, bottom = self.padding
        return value[..., top:value.shape[-2] - bottom, left:value.shape[-1] - right]


class DisplayStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="ipde-display-model-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "core").mkdir()
        (self.root / "core/raft_stereo.py").write_text("# source-tree identity fixture\n")
        self.original = self.root / "original.pth"
        torch.save(FixtureRAFT(None).state_dict(), self.original)
        self.upstream = patch.object(student, "_upstream", return_value=(FixtureRAFT, FixturePadder))
        self.upstream.start()
        self.addCleanup(self.upstream.stop)
        FixtureRAFT.calls.clear()
        generator = np.random.default_rng(901)
        self.left = generator.integers(0, 256, (35, 47, 3), dtype=np.uint8)
        self.right = generator.integers(0, 256, self.left.shape, dtype=np.uint8)

    def model(self, scope="update"):
        return student.create_student(raft_root=self.root, raft_model=self.original,
            device="cpu", train_scope=scope, iterations=2, quality=0, seed=123)

    def payload(self, model, configuration):
        return {"schema": student.SCHEMA, "architecture": deepcopy(configuration),
                "ipde_configuration": deepcopy(configuration), "state_dict": model.state_dict()}

    def context(self, model):
        return model.encode(student.rgb_tensor(self.left, {}, torch, "cpu"),
                            student.rgb_tensor(self.right, {}, torch, "cpu"))

    def test_native_rgb_values_and_storage_are_preserved(self):
        original = self.left.tobytes()
        tensor = student.rgb_tensor(self.left, {}, torch, "cpu")
        self.assertEqual(tuple(tensor.shape), (1, 3, 35, 47))
        np.testing.assert_array_equal(tensor[0].permute(1, 2, 0).numpy(), self.left.astype(np.float32))
        tensor[0, 0, 0, 0] = 999
        self.assertEqual(self.left.tobytes(), original)
        high_bits = np.array([[[0, 511, 1023], [201, 701, 1]]], np.uint16)
        high_original = high_bits.tobytes()
        converted = student.rgb_tensor(high_bits, {"source_bit_depth": 10}, torch, "cpu")
        np.testing.assert_array_equal(converted[0].permute(1, 2, 0).numpy(),
                                      high_bits.astype(np.float32) * np.float32(255 / 1023))
        self.assertEqual(high_bits.tobytes(), high_original)
        linear = np.array([[[0.0, 0.25, 1.0]]], np.float32)
        np.testing.assert_array_equal(student.rgb_tensor(linear, {}, torch, "cpu")[0].permute(1, 2, 0).numpy(), linear * 255)

    def test_encoder_receives_native_images_and_both_cameras_have_gradients(self):
        model, _, _ = self.model("full")
        left = student.rgb_tensor(self.left, {}, torch, "cpu").requires_grad_()
        right = student.rgb_tensor(self.right, {}, torch, "cpu").requires_grad_()
        observed = []
        hook = model.encoder.register_forward_pre_hook(lambda _module, args: observed.append(args[0].detach().clone()))
        try:
            context = model.encode(left, right)
        finally:
            hook.remove()
        self.assertEqual(context["input_shape"], (35, 47))
        self.assertEqual([tuple(value.shape) for value in observed], [(1, 3, 35, 47)] * 2)
        torch.testing.assert_close(observed[0], left.detach() / 127.5 - 1, rtol=0, atol=0)
        torch.testing.assert_close(observed[1], right.detach() / 127.5 - 1, rtol=0, atol=0)
        padder = FixturePadder(left.shape)
        torch.testing.assert_close(padder.unpad(FixtureRAFT.calls[-1][0]), left.detach(), rtol=0, atol=0)
        torch.testing.assert_close(padder.unpad(FixtureRAFT.calls[-1][1]), right.detach(), rtol=0, atol=0)
        prediction = model.render(context, (83, 109), (0, 83, 0, 109))
        prediction.square().mean().backward()
        for gradient in (left.grad, right.grad, model.raft.update_block.weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_tiled_queries_match_full_render_with_learned_camera_offsets(self):
        model, _, _ = self.model()
        model.eval()
        with torch.no_grad():
            model.offset[-1].bias.copy_(torch.tensor([0.07, -0.04, -0.05, 0.03]))
            context = self.context(model)
            shape = (83, 109)
            full = model.render(context, shape, (0, shape[0], 0, shape[1]))
            tiled = torch.empty_like(full)
            visited = torch.zeros(shape, dtype=torch.int8)
            for bounds in student.tile_bounds(shape, 32):
                top, bottom, left, right = bounds
                tiled[..., top:bottom, left:right] = model.render(context, shape, bounds)
                visited[top:bottom, left:right] += 1
        self.assertTrue(bool((visited == 1).all()))
        torch.testing.assert_close(tiled, full, rtol=1e-6, atol=1e-6)
        self.assertTrue(bool(torch.isfinite(full).all()))
        self.assertTrue(bool((full > 0).all()))

    def test_prediction_can_evaluate_an_arbitrary_larger_grid_without_input_mutation(self):
        model, configuration, _ = self.model()
        checkpoint = self.root / "student.pth"
        torch.save(self.payload(model, configuration), checkpoint)
        original_left, original_right = self.left.tobytes(), self.right.tobytes()
        result, metadata = student.predict_display_depth(self.left, self.right, (125, 193), checkpoint,
            raft_root=self.root, device="cpu", tile_size=32)
        self.assertEqual(result.shape, (125, 193))
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result > 0).all())
        self.assertEqual(metadata["native_stereo_shape"], [35, 47])
        self.assertEqual(metadata["output_shape"], [125, 193])
        self.assertEqual(metadata["reference_image"], "display")
        self.assertEqual(metadata["teacher_transport"], "none")
        self.assertEqual((self.left.tobytes(), self.right.tobytes()), (original_left, original_right))

    def test_checkpoint_reload_preserves_weights_configuration_and_predictions(self):
        model, configuration, _ = self.model("full")
        checkpoint = self.root / "student.pth"
        torch.save(self.payload(model, configuration), checkpoint)
        restored, saved_configuration, device = student.load_student_checkpoint(checkpoint,
            raft_root=self.root, device="cpu", train_scope="full")
        self.assertEqual(saved_configuration, configuration)
        self.assertEqual(device, "cpu")
        self.assertEqual(saved_configuration["channels"], 16)
        self.assertEqual(saved_configuration["decoder_channels"], 32)
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)
        with torch.no_grad():
            bounds = (0, 49, 0, 71)
            torch.testing.assert_close(restored.render(self.context(restored), (49, 71), bounds),
                                       model.render(self.context(model), (49, 71), bounds), rtol=0, atol=0)

    def test_archived_checkpoint_loads_the_selected_member_exactly(self):
        model, configuration, _ = self.model("full")
        selected = self.payload(model, configuration)
        other = deepcopy(selected)
        other["state_dict"]["depth.4.bias"].add_(5)
        archive = self.root / "trained-models.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for name, payload in (("other.pth", other), ("exports/chosen.pth", selected)):
                buffer = io.BytesIO(); torch.save(payload, buffer)
                zipped.writestr(name, buffer.getvalue())
        restored, saved_configuration, device = student.load_student_checkpoint(archive,
            checkpoint_member="exports/chosen.pth", raft_root=self.root, device="cpu", train_scope="full")
        self.assertEqual(saved_configuration, configuration)
        self.assertEqual(device, "cpu")
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)

    def test_checkpoint_schema_aliases_and_weights_are_strict(self):
        model, configuration, _ = self.model()
        good = self.payload(model, configuration)
        cases = []
        bad = deepcopy(good); bad["schema"] = "ipde-depth-checkpoint-v1"; cases.append(("stock schema", bad))
        bad = deepcopy(good); bad["ipde_configuration"]["units"] = "relative_depth"; cases.append(("mismatched metadata", bad))
        bad = deepcopy(good); del bad["state_dict"]["depth.4.bias"]; cases.append(("missing weight", bad))
        bad = deepcopy(good); bad["state_dict"]["unused.weight"] = torch.zeros(1); cases.append(("unknown weight", bad))
        bad = deepcopy(good); bad["state_dict"]["depth.4.bias"] = torch.zeros(2); cases.append(("wrong shape", bad))
        bad = deepcopy(good); bad["state_dict"]["depth.4.bias"][0] = float("nan"); cases.append(("nonfinite weight", bad))
        bad = deepcopy(good); bad["state_dict"]["depth.4.bias"] = torch.tensor([1e300], dtype=torch.float64); cases.append(("conversion overflow", bad))
        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises((student.DisplayStudentError, RuntimeError)):
                student.load_student_checkpoint(payload, raft_root=self.root, device="cpu")
        with self.assertRaisesRegex(student.DisplayStudentError, "not a stereo-to-display"):
            student.load_student_checkpoint(self.original, raft_root=self.root, device="cpu")
        checkpoint = self.root / "student.pth"
        torch.save(good, checkpoint)
        with self.assertRaisesRegex(student.DisplayStudentError, "Use --resume"):
            student.create_student(raft_root=self.root, raft_model=checkpoint, device="cpu", quality=0)

    def test_checkpoint_rejects_changed_coordinate_contract_or_unknown_architecture_fields(self):
        model, configuration, _ = self.model()
        for field, value in (("architecture", "stock-raft"), ("input_reference", "resized_stereo"),
                             ("output_reference", "left"), ("teacher_transport", "homography"),
                             ("channels", 8), ("iterations", True), ("unknown_field", "silently ignored")):
            with self.subTest(field=field):
                payload = self.payload(model, configuration)
                payload["architecture"][field] = value
                payload["ipde_configuration"][field] = value
                with self.assertRaises(student.DisplayStudentError):
                    student.load_student_checkpoint(payload, raft_root=self.root, device="cpu")

    def test_scope_keeps_decoder_trainable_and_freezes_only_raft_when_requested(self):
        model, _, _ = self.model("update")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.raft.parameters()))
        self.assertTrue(all(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("raft.")))
        model.render(self.context(model), (39, 51), (0, 39, 0, 51)).mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.raft.parameters()))
        self.assertGreater(float(model.depth[-1].weight.grad.abs().sum()), 0)
        model.set_scope("full")
        self.assertTrue(all(parameter.requires_grad for parameter in model.raft.parameters()))


if __name__ == "__main__":
    unittest.main()
