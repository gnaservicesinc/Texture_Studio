"""CPU tiny-module LoRA differentiation, strict head and read-only contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_material_adapters as probe
from frozen_dino_height import ARCHITECTURE, CODE_REVISION, DIAGNOSTIC_SCHEMA, MODEL_SHA256, ConditionedHeightNet
from test_material_fit_diagnostics import dataset_fixture
from train_material_height import digest, find_samples


class TinyAttention(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, value):
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        return self.proj((q + k + v) / 3)


class TinyBlock(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.attn = TinyAttention(hidden)

    def forward(self, value):
        return value + self.attn(value)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 14, stride=14)
        self.blocks = nn.ModuleList((TinyBlock(), TinyBlock()))
        self.norm = nn.LayerNorm(8)
        self.requires_grad_(False)

    def forward_features(self, rgb):
        value = self.patch_embed(rgb).flatten(2).transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return {"x_norm_patchtokens": self.norm(value)}


def test_zero_initialized_lora_identity_first_B_and_second_A_gradients():
    torch.manual_seed(2307)
    base = nn.Linear(8, 8)
    original = {name: value.detach().clone() for name, value in base.state_dict().items()}
    module = probe.LoRALinear(base, 8, 8)
    value = torch.randn((2, 3, 8))
    torch.testing.assert_close(module(value), base(value), rtol=0, atol=0)
    optimizer = torch.optim.AdamW((module.lora_A, module.lora_B), lr=0.001, weight_decay=0)
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        a, b = module.lora_A.detach().clone(), module.lora_B.detach().clone()
        module(value).square().mean().backward()
        assert base.weight.grad is None and not base.weight.requires_grad
        assert torch.isfinite(module.lora_A.grad).all() and torch.isfinite(module.lora_B.grad).all()
        assert torch.count_nonzero(module.lora_B.grad)
        if step == 1:
            assert torch.count_nonzero(module.lora_A.grad) == 0
        else:
            assert torch.count_nonzero(module.lora_A.grad)
        optimizer.step()
        assert torch.any(b != module.lora_B)
        assert bool(torch.any(a != module.lora_A)) is (step == 2)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_exact_attention_targets_parameter_count_and_base_fingerprint():
    encoder = TinyEncoder()
    original_patch = encoder.patch_embed
    before = probe.base_fingerprint(encoder)
    adapters = probe.install_adapters(encoder, 8, 8, hidden=8, blocks=2)
    assert set(adapters) == {f"blocks.{index}.attn.{name}" for index in range(2) for name in ("qkv", "proj")}
    assert encoder.patch_embed is original_patch
    assert sum(module.lora_A.numel() + module.lora_B.numel() for module in adapters.values()) == 2 * 8 * (8 + 24 + 8 + 8)
    assert probe.base_fingerprint(encoder) == before
    with torch.no_grad():
        encoder.norm.weight[0] += 0.1
    assert probe.base_fingerprint(encoder) != before


def test_wrong_attention_shape_refused_before_any_injection():
    encoder = TinyEncoder()
    encoder.blocks[1].attn.proj = nn.Linear(8, 4)
    with pytest.raises(ValueError, match="layout"):
        probe.install_adapters(encoder, 8, 8, hidden=8, blocks=2)
    assert isinstance(encoder.blocks[0].attn.qkv, nn.Linear)


def test_differentiable_feature_path_rejects_no_grad_output():
    encoder = TinyEncoder()
    probe.install_adapters(encoder, 8, 8, hidden=8, blocks=2)
    source = torch.rand((1, 3, 32, 32))
    features = probe.differentiable_features(encoder, source, 28, hidden=8)
    assert features.shape == (1, 8, 2, 2) and features.requires_grad
    features.square().mean().backward()
    assert encoder.blocks[0].attn.qkv.lora_B.grad is not None
    with torch.no_grad(), pytest.raises(ValueError, match="differentiable"):
        probe.differentiable_features(encoder, source, 28, hidden=8)


def test_differentiable_head_exact_forward_and_feature_gradients():
    original = ConditionedHeightNet(4, 8, 4)
    with torch.no_grad():
        original.head.weight.fill_(0.03)
    differentiable = probe.DifferentiableHeightNet(4, 8, 4)
    differentiable.load_state_dict(original.state_dict(), strict=True)
    source, feature = torch.rand((1, 3, 32, 32)), torch.rand((1, 8, 2, 2), requires_grad=True)
    expected = original(source, feature.detach())
    actual = differentiable(source, feature)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all() and torch.count_nonzero(feature.grad)


def checked_head_file(path):
    head = ConditionedHeightNet()
    with torch.no_grad():
        head.head.weight.fill_(0.01)
    checkpoint = {"schema": DIAGNOSTIC_SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 12, "feature_channels": 768, "projection_channels": 12}, "model_state": head.state_dict(), "variant": "frozen_features", "step": 300, "diagnostic_only_not_production": True, "encoder_size": 518, "encoder": {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION}, "sample_map_sha256": {}}
    torch.save(checkpoint, path)
    return checkpoint


def test_head_exact_hash_schema_encoder_pin_and_learned_weights(tmp_path):
    path = tmp_path / "head.pt"
    checkpoint = checked_head_file(path)
    head, metadata = probe.load_checked_head(path, torch.device("cpu"), digest(path))
    assert torch.count_nonzero(head.head.weight)
    assert metadata["source_step"] == 300
    with pytest.raises(ValueError, match="SHA256"):
        probe.load_checked_head(path, torch.device("cpu"), "0" * 64)
    checkpoint["schema"] = "wrong"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="checked 300"):
        probe.load_checked_head(path, torch.device("cpu"), digest(path))
    checkpoint["schema"] = DIAGNOSTIC_SCHEMA
    checkpoint["encoder"]["checkpoint_sha256"] = "0" * 64
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="provenance"):
        probe.load_checked_head(path, torch.device("cpu"), digest(path))


def arguments(tmp_path):
    dataset = tmp_path / "dataset"
    dataset_fixture(dataset)
    head = tmp_path / "tiny-head.fixture"
    head.write_bytes(b"read-only head fixture")
    return SimpleNamespace(device="cpu", dataset=dataset, sample_id="material_auto_001", allow_unreviewed=True, mask_transparent_input=False, expected_size=32, output=tmp_path / "probe", head_checkpoint=head, model_directory=tmp_path, code_directory=tmp_path, rank=8, alpha=8, seed=2307, encoder_size=28, learning_rate=0.0001, max_seconds=120, max_driver_bytes=probe.MAX_DRIVER_BYTES)


def tiny_encoder_loader(_model, _code, device):
    torch.manual_seed(2307)
    return TinyEncoder().to(device), {"test_fixture": True, "all_encoder_parameters_frozen": True}


def tiny_head_loader(args):
    sample = find_samples(args.dataset, True)[0]
    maps = {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")}
    def load(_path, device):
        head = probe.DifferentiableHeightNet(4, 8, 4)
        with torch.no_grad():
            head.head.weight.fill_(0.04)
        return head.to(device), {"model_config": {"base_channels": 4, "feature_channels": 8, "projection_channels": 4}, "sample_map_sha256": {args.sample_id: maps}, "test_fixture": True}
    return load


def test_two_real_cpu_steps_update_adapters_keep_base_sources_and_target(tmp_path):
    args = arguments(tmp_path)
    originals = {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    head_original = args.head_checkpoint.read_bytes()
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["status"] == "complete" and report["completed_updates"] == 2
    assert report["original_pretrained_tensors_unchanged"] and report["target_float32_batch_unchanged"]
    assert report["original_pretrained_fingerprint_before"] == report["original_pretrained_fingerprint_after"]
    assert originals == {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    assert args.head_checkpoint.read_bytes() == head_original
    first, second = report["steps"]
    assert first["features_requires_grad"] and second["features_requires_grad"]
    assert all(step["head_gradient_norm_after_clipping"] > 0 and step["head_output_weights_changed"] for step in (first, second))
    assert first["any_B_parameters_changed"] and not first["any_A_parameters_changed"]
    assert second["any_B_parameters_changed"] and second["any_A_parameters_changed"]
    assert report["optimizer_tensor_state_bytes"] > 0
    assert not report["quality_training"] and not report["full_encoder_finetuning_memory_tested"]
    checkpoint = torch.load(args.output / "adapter-head.feasibility.pt", weights_only=True)
    assert checkpoint["schema"] == probe.SCHEMA and checkpoint["feasibility_only_not_quality_model"]
    assert "base_state" not in checkpoint
    assert all(set(values) == {"lora_A", "lora_B"} for values in checkpoint["adapter_state"].values())
    assert report["saved_checkpoint_bytes"] < 1024 * 1024
    json.dumps(report, allow_nan=False)


def test_forward_memory_soft_guard_stops_before_backprop(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    sequence = iter(({"mps_driver_bytes": 1}, {"mps_driver_bytes": 1}, {"mps_driver_bytes": 31 * 1024**3}))
    monkeypatch.setattr(probe, "memory", lambda _device: next(sequence))
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["status"] == "stopped_memory_soft_guard"
    assert report["completed_updates"] == 0
    assert not report["stopped_after_forward"]["backward_performed"]
    assert report["original_pretrained_tensors_unchanged"]


def test_optimizer_memory_soft_guard_retains_one_update_and_stops_next_forward(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    sequence = iter(({"mps_driver_bytes": 1}, {"mps_driver_bytes": 1}, {"mps_driver_bytes": 1}, {"mps_driver_bytes": 31 * 1024**3}))
    monkeypatch.setattr(probe, "memory", lambda _device: next(sequence))
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["status"] == "stopped_memory_soft_guard"
    assert report["completed_updates"] == 1
    assert report["stopped_after_optimizer"]["completed_update_retained"]
    assert report["steps"][0]["any_B_parameters_changed"]


def test_existing_driver_pressure_stops_before_forward(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(probe, "memory", lambda _device: {"mps_driver_bytes": 31 * 1024**3})
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["status"] == "stopped_memory_soft_guard"
    assert report["completed_updates"] == 0
    assert not report["stopped_before_forward"]["forward_performed"]


def test_second_update_exceeding_guard_is_not_nominal_complete(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    sequence = iter([{"mps_driver_bytes": 1}] * 6 + [{"mps_driver_bytes": 31 * 1024**3}])
    monkeypatch.setattr(probe, "memory", lambda _device: next(sequence))
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["completed_updates"] == 2
    assert report["status"] == "stopped_memory_soft_guard"
    assert report["stopped_after_optimizer"]["step"] == 2


def test_post_update_time_guard_stops_before_next_forward(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    clock, memory_calls = [0.0], [0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    def memory(_device):
        memory_calls[0] += 1
        if memory_calls[0] == 4:
            clock[0] = 121.0
        return {"mps_driver_bytes": 1}
    monkeypatch.setattr(probe, "memory", memory)
    report = probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert report["completed_updates"] == 1
    assert report["status"] == "stopped_time_soft_guard"
    assert memory_calls[0] == 4


def test_mps_cpu_fallback_refused_before_model_load(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.device = "mps"
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="refuses CPU fallback"):
        probe.run(args)


def test_changed_map_after_checked_decode_is_not_adopted_by_snapshot(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    original = probe.load_pair
    def decode_then_change(sample, device, **kwargs):
        result = original(sample, device, **kwargs)
        sample["input_path"].write_bytes(b"concurrent changed map")
        return result
    monkeypatch.setattr(probe, "load_pair", decode_then_change)
    with pytest.raises(ValueError, match="changed during probe preflight"):
        probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert not args.output.exists()


def test_manifest_snapshot_must_match_selected_metadata(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    original = probe.find_samples
    def select_then_change(dataset, allow):
        samples = original(dataset, allow)
        sample = samples[0]
        changed = dict(sample["metadata"], review_status="changed after selection")
        sample["metadata_path"].write_text(json.dumps(changed))
        return samples
    monkeypatch.setattr(probe, "find_samples", select_then_change)
    with pytest.raises(ValueError, match="Selected sample manifest changed"):
        probe.run(args, encoder_loader=tiny_encoder_loader, head_loader=tiny_head_loader(args), encoder_contract=(8, 2))
    assert not args.output.exists()
