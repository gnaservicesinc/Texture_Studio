"""PBRnxt adapter contracts, using tiny stand-ins rather than GPU model runs."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_pbrnxt as adapter


class Encoder(nn.Module):
    def forward(self, value):
        return value, value, value


class Decoder(nn.Module):
    def forward(self, body, stage1, stage2, stage3):
        return body + stage1 + stage2 + stage3, stage2, stage3


class TinyGenerator(nn.Module):
    """Same public layer interface, much smaller; no actual PBRnxt execution."""
    def __init__(self, *unused):
        super().__init__()
        self.dim = 2
        self.out_nc = [3, 3, 1, 1]
        self.m_head = nn.Conv2d(3, self.dim, 1)
        self.m_enc = Encoder()
        self.m_body = nn.Identity()
        self.m_fuse = nn.Conv2d(self.dim * 4, self.dim * 4, 1)
        for index, channels in enumerate(self.out_nc):
            setattr(self, f"m_dec_{index}", Decoder())
            tail = nn.Conv2d(self.dim, channels, 1)
            with torch.no_grad():
                tail.weight.zero_()
                tail.bias.fill_((-.25, .5, .2, 1.3)[index])
            setattr(self, f"m_tail_{index}", tail)

    def forward(self, *args, **kwargs):
        raise AssertionError("The upstream border-padding forward must be bypassed")


class TinyRRDB(nn.Module):
    def __init__(self, in_nc, out_nc, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(in_nc, 2, 1), nn.Upsample(scale_factor=2, mode="nearest"),
                                   nn.Conv2d(2, 2, 1), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(2, out_nc, 1))

    def forward(self, *args, **kwargs):
        raise AssertionError("Original RRDB enlargement forward must be bypassed")


def tiny_complete():
    return adapter.NativePBRnxtComplete(TinyGenerator(),
                                       nn.ModuleList([TinyRRDB(11, count) for count in adapter.MAP_CHANNELS]), {})


class PBRnxtAdapterTests(unittest.TestCase):
    def test_native_forward_bypasses_border_upscaler_and_preserves_raw_maps(self):
        model = adapter.NativePBRnxt(TinyGenerator(), {})
        source = torch.linspace(0, 1, 3 * 64 * 128).reshape(1, 3, 64, 128)
        original = source.clone()
        with patch.object(adapter.F, "pad", side_effect=AssertionError("No image padding")), \
             patch.object(adapter.F, "interpolate", side_effect=AssertionError("No image resizing")):
            maps = model.maps(source)
        self.assertEqual(list(maps), ["diffuse", "normal", "roughness", "height"])
        self.assertEqual([value.shape[1] for value in maps.values()], [3, 3, 1, 1])
        self.assertTrue(all(value.shape[-2:] == (64, 128) for value in maps.values()))
        self.assertLess(float(maps["diffuse"].detach().min()), 0)
        self.assertGreater(float(maps["height"].detach().min()), 1)
        torch.testing.assert_close(source, original, atol=0, rtol=0)

    def test_invalid_native_grid_rejected_before_any_model_layer(self):
        model = adapter.NativePBRnxt(TinyGenerator(), {})
        with patch.object(model.gen.m_head, "forward", side_effect=AssertionError("No model execution")):
            for shape in ((1, 3, 63, 64), (1, 3, 64, 65), (1, 3, 32, 32)):
                with self.assertRaisesRegex(ValueError, "padding and resizing are forbidden"):
                    model(torch.zeros(shape))
            with self.assertRaisesRegex(ValueError, "floating-point"):
                model(torch.zeros((1, 3, 64, 64), dtype=torch.uint16))

    def test_native_head_and_feature_weights_receive_real_gradients(self):
        gen = TinyGenerator()
        for index in range(4):
            getattr(gen, f"m_tail_{index}").weight.data.fill_(.1)
        model = adapter.NativePBRnxt(gen, {})
        value = torch.rand(1, 3, 64, 64)
        model(value).square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_device_noise_changes_no_state_keys_and_eval_is_deterministic(self):
        noise = adapter.DeviceGaussianNoise()
        self.assertEqual(dict(noise.state_dict()), {})
        self.assertEqual(list(noise.buffers()), [])
        self.assertEqual(list(noise.parameters()), [])
        value = torch.ones(1, 3, 8, 12)
        noise.eval()
        state = torch.random.get_rng_state().clone()
        self.assertIs(noise(value), value)
        torch.testing.assert_close(torch.random.get_rng_state(), state, atol=0, rtol=0)
        noise.train()
        training = noise(value.requires_grad_())
        self.assertEqual(training.device.type, "cpu")
        self.assertEqual(training.shape, value.shape)
        self.assertFalse(torch.equal(training, value))
        training.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_changed_executable_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = "archs/scunetv2_arch.py"
            path = directory / name
            path.parent.mkdir()
            path.write_bytes(b"changed")
            with patch.object(adapter, "SOURCE_FILES", {name: (7, "0" * 64)}), \
                 patch.object(adapter.importlib.util, "spec_from_file_location", side_effect=AssertionError("No unverified import")):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    adapter._architecture(directory)

    def test_pinned_download_verified_and_existing_files_never_replaced(self):
        payload = b"pinned source"
        identity = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "architecture.py"
            with self.assertRaisesRegex(FileNotFoundError, "enable download"):
                adapter._obtain_file(path, len(payload), identity, "https://example.test/pin", False)
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                adapter._obtain_file(path, len(payload), identity, "https://example.test/pin", True)
            self.assertEqual(path.read_bytes(), payload)
            original_inode = path.stat().st_ino
            with patch.object(adapter.urllib.request, "urlopen", side_effect=AssertionError("No re-download")):
                adapter._obtain_file(path, len(payload), identity, "https://example.test/pin", True)
            self.assertEqual(path.stat().st_ino, original_inode)
            path.write_bytes(b"bad existing source")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                adapter._obtain_file(path, len(payload), identity, "https://example.test/pin", True)
            self.assertEqual(path.read_bytes(), b"bad existing source")

    def test_wrong_download_not_published_and_partial_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "weights.pth"
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    adapter._obtain_file(path, 5, "0" * 64, "https://example.test/pin", True)
            self.assertEqual(list(directory.iterdir()), [])

    def test_strict_generator_load_retains_pretrained_height_and_ignores_only_upscaler(self):
        gen = TinyGenerator()
        state = {"gen." + key: value.clone() for key, value in gen.state_dict().items()}
        state["ups.0.upscaler.weight"] = torch.tensor([27.])
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "weights.pth"
            torch.save(state, weights)
            with patch.object(adapter, "WEIGHTS_BYTES", weights.stat().st_size), \
                 patch.object(adapter, "WEIGHTS_SHA256", adapter.sha256(weights)), \
                 patch.object(adapter, "_architecture", return_value=SimpleNamespace(SCUNet=TinyGenerator)), \
                 patch.object(adapter, "source_provenance", return_value=[]):
                model = adapter.load_pretrained(Path(temporary), weights)
            self.assertFalse(model.training)
            for name, value in gen.state_dict().items():
                torch.testing.assert_close(value, model.gen.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(model.provenance["map_channels"]["height"], 1)
            self.assertFalse(model.provenance["image_padding"])
            self.assertFalse(model.provenance["upscaler_used"])
            self.assertEqual(model.provenance["generator_parameters"], sum(parameter.numel() for parameter in gen.parameters()))

    def test_missing_pretrained_decoder_key_fails_without_random_fallback(self):
        state = {"gen." + key: value for key, value in TinyGenerator().state_dict().items()}
        del state["gen.m_tail_3.weight"]
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "weights.pth"
            torch.save(state, weights)
            with patch.object(adapter, "WEIGHTS_BYTES", weights.stat().st_size), \
                 patch.object(adapter, "WEIGHTS_SHA256", adapter.sha256(weights)), \
                 patch.object(adapter, "_architecture", return_value=SimpleNamespace(SCUNet=TinyGenerator)):
                with self.assertRaisesRegex(RuntimeError, "m_tail_3.weight"):
                    adapter.load_pretrained(Path(temporary), weights)

    def test_complete_mapping_retains_every_convolution_on_original_grid(self):
        model = tiny_complete().eval()
        counts = [0] * 4
        handles = []
        for number, branch in enumerate(model.ups):
            for module in branch.model:
                if isinstance(module, nn.Conv2d):
                    def count(module, args, output, number=number):
                        counts[number] += 1
                    handles.append(module.register_forward_hook(count))
        source = torch.rand(1, 3, 64, 128)
        original = source.clone()
        with patch.object(adapter.F, "pad", side_effect=AssertionError("No image padding")), \
             patch.object(adapter.F, "interpolate", side_effect=AssertionError("No enlargement or resizing")):
            output = model(source)
        self.assertEqual(output.shape, (1, 8, 64, 128))
        self.assertEqual(counts, [3, 3, 3, 3])
        torch.testing.assert_close(source, original, atol=0, rtol=0)
        for handle in handles:
            handle.remove()

    def test_optimized_height_equals_full_output_without_other_final_branches(self):
        model = tiny_complete().eval()
        source = torch.rand(1, 3, 64, 64)
        expected = model(source)[:, 7:8].detach()
        with patch.object(model.ups[0].model[0], "forward", side_effect=AssertionError("No diffuse final branch")), \
             patch.object(model.ups[1].model[0], "forward", side_effect=AssertionError("No normal final branch")), \
             patch.object(model.ups[2].model[0], "forward", side_effect=AssertionError("No roughness final branch")):
            height = model.height(source)
        torch.testing.assert_close(height, expected, atol=0, rtol=0)

    def test_final_height_refinement_backpropagates_only_selected_pretrained_branch(self):
        model = tiny_complete().eval()
        model.requires_grad_(False)
        model.ups[3].requires_grad_(True)
        source = torch.rand(1, 3, 64, 64)
        target = torch.full((1, 1, 64, 64), .7)
        torch.nn.functional.l1_loss(model.height(source), target).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.gen.parameters()))
        self.assertTrue(all(parameter.grad is None for branch in model.ups[:3] for parameter in branch.parameters()))
        final = model.ups[3].model[-1]
        self.assertGreater(float(final.weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.ups[3].parameters()))

    def test_complete_strict_load_retains_all_pretrained_branches_and_provenance(self):
        original = tiny_complete()
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "complete.pth"
            torch.save(original.state_dict(), weights)
            with patch.object(adapter, "WEIGHTS_BYTES", weights.stat().st_size), \
                 patch.object(adapter, "WEIGHTS_SHA256", adapter.sha256(weights)), \
                 patch.object(adapter, "_architecture", return_value=SimpleNamespace(SCUNet=TinyGenerator)), \
                 patch.object(adapter, "_rrdb_architecture", return_value=SimpleNamespace(RRDBNet=TinyRRDB)), \
                 patch.object(adapter, "source_provenance", return_value=[]):
                restored = adapter.load_complete_pretrained(Path(temporary), weights)
            self.assertFalse(restored.training)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(restored.provenance["published_output_scale"], 4)
            self.assertEqual(restored.provenance["adapted_output_scale"], 1)
            self.assertTrue(restored.provenance["all_pretrained_parameter_keys_loaded"])
            self.assertTrue(restored.provenance["trained_final_rrdb_mapping_used"])
            self.assertFalse(restored.provenance["production_eligible"])
            self.assertFalse(restored.provenance["image_padding"])
            self.assertFalse(restored.provenance["image_resizing"])

    def test_missing_final_height_weights_rejected_without_fallback(self):
        state = tiny_complete().state_dict()
        del state["ups.3.model.4.weight"]
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "incomplete.pth"
            torch.save(state, weights)
            with patch.object(adapter, "WEIGHTS_BYTES", weights.stat().st_size), \
                 patch.object(adapter, "WEIGHTS_SHA256", adapter.sha256(weights)), \
                 patch.object(adapter, "_architecture", return_value=SimpleNamespace(SCUNet=TinyGenerator)), \
                 patch.object(adapter, "_rrdb_architecture", return_value=SimpleNamespace(RRDBNet=TinyRRDB)):
                with self.assertRaisesRegex(RuntimeError, "ups.3.model.4.weight"):
                    adapter.load_complete_pretrained(Path(temporary), weights)

    def test_unexpected_upscaler_operator_is_not_silently_adapted(self):
        branch = TinyRRDB(11, 1)
        branch.model[1] = nn.Upsample(scale_factor=3, mode="nearest")
        with self.assertRaisesRegex(ValueError, "unexpected enlargement"):
            adapter.native_branch_forward(branch, torch.rand(1, 11, 64, 64))


if __name__ == "__main__":
    unittest.main()
