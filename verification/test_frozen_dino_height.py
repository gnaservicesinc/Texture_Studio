"""Frozen-encoder loading, matched controls and native signal regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_frozen_dino_height import run
from frozen_dino_height import DIAGNOSTIC_SCHEMA, ConditionedHeightNet, convert_hf_state, encoder_input, extract_features, state_sha256
from test_material_fit_diagnostics import dataset_fixture
from train_material_height import digest, load_pair, find_samples


class TinyFrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward_features(self, rgb):
        side = rgb.shape[-1] // 14
        value = torch.nn.functional.avg_pool2d(rgb.mean(1, keepdim=True), 14)
        return {"x_norm_patchtokens": value.flatten(2).transpose(1, 2).repeat(1, 1, 768) * self.weight}


def tiny_loader(_model, _code, device):
    return TinyFrozenEncoder().to(device), {"frozen_parameters": 1, "all_encoder_parameters_frozen": True, "test_double": True}


def hf_fixture():
    state, expected = {}, {}
    for target, source, shape in (
        ("cls_token", "embeddings.cls_token", (1, 1, 4)), ("mask_token", "embeddings.mask_token", (1, 4)),
        ("pos_embed", "embeddings.position_embeddings", (1, 5, 4)),
        ("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight", (4, 3, 2, 2)),
        ("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias", (4,)),
        ("norm.weight", "layernorm.weight", (4,)), ("norm.bias", "layernorm.bias", (4,)),
    ):
        state[source] = torch.randn(shape)
        expected[target] = torch.zeros(shape)
    for suffix in ("weight", "bias"):
        shape = (4, 4) if suffix == "weight" else (4,)
        for index, name in enumerate(("query", "key", "value")):
            state[f"encoder.layer.0.attention.attention.{name}.{suffix}"] = torch.full(shape, float(index + 1))
        expected[f"blocks.0.attn.qkv.{suffix}"] = torch.zeros((12, 4) if suffix == "weight" else (12,))
        state[f"encoder.layer.0.attention.output.dense.{suffix}"] = torch.randn(shape)
        expected[f"blocks.0.attn.proj.{suffix}"] = torch.zeros(shape)
        for name in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
            state[f"encoder.layer.0.{name}.{suffix}"] = torch.randn(shape)
            expected[f"blocks.0.{name}.{suffix}"] = torch.zeros(shape)
    for index in (1, 2):
        state[f"encoder.layer.0.layer_scale{index}.lambda1"] = torch.randn(4)
        expected[f"blocks.0.ls{index}.gamma"] = torch.zeros(4)
    return state, expected


def test_strict_all_tensor_mapping_qkv_order_and_mask_token():
    source, expected = hf_fixture()
    result = convert_hf_state(source, expected, depth=1)
    assert set(result) == set(expected)
    for suffix in ("weight", "bias"):
        for index in range(3):
            assert torch.all(result[f"blocks.0.attn.qkv.{suffix}"][index * 4:(index+1)*4] == index + 1)
    torch.testing.assert_close(result["mask_token"], source["embeddings.mask_token"], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ("missing", "extra", "shape", "nonfinite", "half"))
def test_encoder_mapping_refuses_incomplete_or_wrong_precision(damage):
    source, expected = hf_fixture()
    if damage == "missing":
        source.pop("embeddings.cls_token")
    elif damage == "extra":
        source["unexpected"] = torch.ones(1)
    elif damage == "shape":
        source["embeddings.cls_token"] = torch.ones(2)
    elif damage == "nonfinite":
        source["embeddings.cls_token"].fill_(float("nan"))
    else:
        source["embeddings.cls_token"] = source["embeddings.cls_token"].half()
    with pytest.raises(ValueError):
        convert_hf_state(source, expected, depth=1)


def test_encoder_only_srgb_conversion_preserves_native_rgb_and_whole_crop():
    source = torch.full((1, 3, 32, 32), 0.21404114)
    original = source.clone()
    encoded = encoder_input(source, 28)
    expected = (torch.tensor(0.5) - torch.tensor((0.485, 0.456, 0.406))) / torch.tensor((0.229, 0.224, 0.225))
    torch.testing.assert_close(encoded[0, :, 14, 14], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    corners = torch.zeros((1, 3, 32, 32)); corners[..., 0:8, 0:8] = 1
    assert encoder_input(corners, 28)[0, 0, 0, 0] > 1
    with pytest.raises(ValueError, match="patch-aligned"):
        encoder_input(source, 512)


def test_frozen_cached_features_no_graph_and_native_head_real_gradients():
    encoder = TinyFrozenEncoder()
    source = torch.rand((1, 3, 32, 32))
    features = extract_features(encoder, source, 28)
    assert features.shape == (1, 768, 2, 2) and not features.requires_grad
    model = ConditionedHeightNet(4, 768, 4)
    # Zero head means the rest acquires gradients after the first real update.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(source, features)
        assert output.shape == (1, 1, 32, 32)
        (output - source[:, :1]).square().mean().backward()
        assert torch.isfinite(model.head.weight.grad).all()
        optimizer.step()
    assert torch.count_nonzero(model.encoder[0].layers[0].weight.grad)
    assert torch.count_nonzero(model.feature_projection.weight.grad)
    assert encoder.weight.grad is None and not encoder.training
    with pytest.raises(ValueError, match="detached"):
        model(source, features.requires_grad_())


def test_projection_precedes_upsampling_and_native_1024_output():
    model = ConditionedHeightNet(4, 768, 4).eval()
    shapes = []
    hook = model.feature_projection.register_forward_hook(lambda _module, incoming, outgoing: shapes.append((incoming[0].shape, outgoing.shape)))
    with torch.no_grad():
        result = model(torch.rand((1, 3, 1024, 1024)), torch.rand((1, 768, 37, 37)))
    hook.remove()
    assert result.shape == (1, 1, 1024, 1024)
    assert shapes == [(torch.Size((1, 768, 37, 37)), torch.Size((1, 4, 37, 37)))]


def test_matched_cpu_controls_source_integrity_and_bounded_diagnostic(tmp_path):
    dataset = tmp_path / "dataset"
    dataset_fixture(dataset)
    original_files = {str(path): digest(path) for path in dataset.rglob("*") if path.is_file()}
    args = SimpleNamespace(device="cpu", encoder_device="cpu", dataset=dataset, material="material", sample_id=None, validation_sample_id=None, allow_unreviewed=True, mask_transparent_input=False, expected_size=32, output=tmp_path / "output", model_directory=tmp_path, code_directory=tmp_path, encoder_size=28, seed=2307, learning_rate=0.001, steps=3, base_channels=4, projection_channels=4, variants=["zero_features", "frozen_features"], max_minutes=0, log_every=3)
    sample = find_samples(dataset, True)[0]
    _, target_before, _ = load_pair(sample, torch.device("cpu"), return_mask=True)
    report = run(args, encoder_loader=tiny_loader)
    _, target_after, _ = load_pair(sample, torch.device("cpu"), return_mask=True)
    torch.testing.assert_close(target_before, target_after, rtol=0, atol=0)
    assert report["not_lora"] and report["not_fresh_material_generalization"]
    assert report["matched_completed_steps"]
    assert report["artifact_bytes"] < 200 * 1024 * 1024
    assert original_files == {str(path): digest(path) for path in dataset.rglob("*") if path.is_file()}
    assert len(list(args.output.glob("*/checkpoint.final.pt"))) == 2
    initial_hashes = {value["initial_head_state_sha256"] for value in report["variants"].values()}
    assert initial_hashes == {report["initial_head_state_sha256"]}
    for variant in args.variants:
        result = report["variants"][variant]
        assert result["initial_prediction_identically_0_5"]
        assert result["gradient_proof"]["step_1"]["head_parameters_changed"]
        assert result["gradient_proof"]["step_2"]["native_rgb_encoder_parameters_changed"]
        assert result["gradient_proof"]["step_2"]["native_rgb_encoder_gradient_nonzero"]
        assert not result["gradient_proof"]["step_2"]["cached_encoder_features_require_grad"]
        if variant == "frozen_features":
            assert result["gradient_proof"]["step_2"]["feature_projection_gradient_norm"] > 0
        else:
            assert result["gradient_proof"]["step_2"]["feature_projection_gradient_norm"] == 0
        saved = torch.load(args.output / variant / "checkpoint.final.pt", weights_only=True)
        assert saved["schema"] == DIAGNOSTIC_SCHEMA and saved["diagnostic_only_not_production"]
        reloaded = ConditionedHeightNet(**saved["model_config"])
        reloaded.load_state_dict(saved["model_state"], strict=True)
        assert state_sha256(reloaded.state_dict()) == state_sha256(saved["model_state"])
        export = args.output / variant / "final-training-fit"
        metadata = json.loads((export / "final-training-fit.prediction.json").read_text())
        assert metadata["blender_color_space"] == "Non-Color"
        assert metadata["schema"] == DIAGNOSTIC_SCHEMA
        values = np.load(export / "final-training-fit.height.float32.npy", allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (32, 32)
    json.dumps(report, allow_nan=False)
