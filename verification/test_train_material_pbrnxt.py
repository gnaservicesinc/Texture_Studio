"""Refinement changes pretrained height parameters and preserves native data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_material_pbrnxt as trainer
from material_pbrnxt_data import digest, numeric_height


class TinyPretrained(nn.Module):
    """Small CPU architecture fixture; tests the actual optimizer/state path."""
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Dropout(.9))
        self.ups = nn.ModuleList(nn.Conv2d(4, 1, 1) for _ in range(4))
        self.provenance = {"fixture": "pretrained structure", "native_grid": 64}
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                parameter.fill_(.01 + index * .002)

    def height(self, rgb):
        return self.ups[3](self.gen(rgb))


def settings(**overrides):
    values = dict(size=256, seed=17, learning_rate=.03, weight_decay=0., memory_gib=48., cache_gib=1.)
    values.update(overrides)
    return SimpleNamespace(**values)


def checkpoint_fixture(tmp_path):
    model = TinyPretrained()
    parameters = trainer.configure_refinement(model)
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0)
    rgb, target = torch.full((1, 3, 256, 256), .4), torch.full((1, 1, 256, 256), .6)
    initial = trainer.predict(model, rgb).detach().clone()
    loss = (trainer.predict(model, rgb) - target).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, model, optimizer, 1, {"dataset": {"index_sha256": "test-index"}}, settings())
    return model, optimizer, path, rgb, initial


def assert_same_weights(first, second):
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_only_existing_pretrained_final_height_parameters_receive_optimizer_updates(tmp_path):
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = trainer.configure_refinement(model)
    expected_names = {name for name, _ in model.named_parameters() if name.startswith("ups.3.")}
    actual_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert actual_names == expected_names and len(parameters) == 2
    assert not model.training and not model.gen[1].training
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0)
    rgb = torch.full((1, 3, 256, 256), .4)
    prediction = trainer.predict(model, rgb)
    assert prediction.shape == (1, 1, 256, 256)
    (prediction - .6).square().mean().backward()
    optimizer.step()
    after = model.state_dict()
    assert all(not torch.equal(before[name], after[name]) for name in expected_names)
    assert all(torch.equal(before[name], after[name]) for name in before if name not in expected_names)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in expected_names)
    with pytest.raises(ValueError, match="Pretrained height decoder"):
        trainer.configure_refinement(nn.Linear(3, 1))


def test_delta_checkpoint_roundtrip_reproduces_refined_numeric_height_without_changing_base(tmp_path):
    refined, _, path, rgb, initial = checkpoint_fixture(tmp_path)
    saved = torch.load(path, weights_only=True)
    assert saved["schema"] == trainer.SCHEMA
    assert saved["base_sha256"] == trainer.WEIGHTS_SHA256 and saved["base_revision"] == trainer.REVISION
    assert set(saved["height_decoder_state"]) == {"ups.3.weight", "ups.3.bias"}
    assert saved["training_crop_size"] == 256 and saved["configuration"]["size"] == 256
    assert saved["image_padding"] is saved["image_resizing"] is False
    assert saved["target_transfer"] == "UInt16 linear numeric codes /65535"
    assert saved["experimental"] and not saved["production_eligible"]
    restored = TinyPretrained()
    trainer.configure_refinement(restored)
    frozen = {name: tensor.clone() for name, tensor in restored.state_dict().items() if not name.startswith("ups.3.")}
    result = trainer.restore_refinement(restored, path)
    assert result["step"] == 1
    assert_same_weights(frozen, {name: tensor for name, tensor in restored.state_dict().items() if name in frozen})
    torch.testing.assert_close(trainer.predict(restored, rgb), trainer.predict(refined, rgb), rtol=0, atol=0)
    assert not torch.equal(trainer.predict(restored, rgb), initial)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("field,value", [
    ("schema", "other-model-v1"), ("base_sha256", "0" * 64), ("base_revision", "unknown"),
    ("target", "roughness"), ("image_padding", True), ("image_resizing", True),
    ("training_crop_size", 255), ("training_crop_size", 257), ("training_crop_size", True),
    ("training_crop_size", 256.), ("step", -1), ("step", True), ("step", 1.5),
    ("configuration", None), ("configuration", {"size": 512}),
    ("height_decoder_state", None),
])
def test_checkpoint_rejects_wrong_identity_grid_and_metadata_without_partial_changes(tmp_path, field, value):
    _, _, path, _, _ = checkpoint_fixture(tmp_path)
    saved = torch.load(path, weights_only=True)
    saved[field] = value
    torch.save(saved, path)
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        trainer.restore_refinement(model, path)
    assert_same_weights(before, model.state_dict())


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "nan", "inf", "integer", "float64", "non-tensor", "non-dict"])
def test_checkpoint_rejects_incompatible_delta_tensors_before_any_weights_change(tmp_path, problem):
    _, _, path, _, _ = checkpoint_fixture(tmp_path)
    saved = torch.load(path, weights_only=True)
    state = saved["height_decoder_state"]
    if problem == "missing":
        state.pop("ups.3.bias")
    elif problem == "extra":
        state["gen.0.weight"] = torch.ones(4, 3, 1, 1)
    elif problem == "shape":
        state["ups.3.bias"] = torch.ones(2)
    elif problem == "nan":
        state["ups.3.bias"][0] = float("nan")
    elif problem == "inf":
        state["ups.3.bias"][0] = float("inf")
    elif problem == "integer":
        state["ups.3.bias"] = state["ups.3.bias"].to(torch.int32)
    elif problem == "float64":
        state["ups.3.bias"] = state["ups.3.bias"].to(torch.float64)
    elif problem == "non-tensor":
        state["ups.3.bias"] = [1.]
    elif problem == "non-dict":
        saved = torch.ones(1)
    torch.save(saved, path)
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        trainer.restore_refinement(model, path)
    assert_same_weights(before, model.state_dict())


def test_resource_guard_rejects_too_large_native_grid_before_dataset_or_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer.os, "sysconf", lambda name: 16 * 1024 * 1024 if name == "SC_PHYS_PAGES" else 4096)
    for name in ("select_pairs", "load_complete_pretrained", "obtain_pretrained"):
        monkeypatch.setattr(trainer, name, lambda *_args, **_kwargs: pytest.fail("Oversized grid must stop before dataset/model allocation"))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="smaller real training crop"):
        trainer.main(["train", "--dataset", str(tmp_path / "missing"), "--output", str(output),
                      "--size", "2048", "--memory-gib", "56", "--cache-gib", "4", "--device", "cpu"])
    assert not output.exists()
    plan = trainer.resource_plan(1024, 56, 4)
    assert plan["combined_estimate_gib"] < 56 and plan["cpu_cache_gib"] == 4
    with pytest.raises(ValueError, match="padding or resizing"):
        trainer.resource_plan(1000, 56, 4)
    with pytest.raises(ValueError, match="4GiB for macOS"):
        trainer.resource_plan(1024, 61, 4)


@pytest.mark.parametrize("problem", ["small-grid", "extra-channel", "nan"])
def test_predict_rejects_changed_output_grid_channels_or_invalid_values(problem):
    class BrokenModel:
        def height(self, rgb):
            if problem == "small-grid":
                return torch.zeros(1, 1, 32, 32)
            if problem == "extra-channel":
                return torch.zeros(1, 3, 256, 256)
            return torch.full((1, 1, 256, 256), float("nan"))

    with pytest.raises(ValueError, match="invalid native height"):
        trainer.predict(BrokenModel(), torch.zeros(1, 3, 256, 256))


class ArrayCache:
    def __init__(self, rgb, height):
        self.rgb, self.height = rgb, height

    def load(self, _pair):
        return self.rgb, self.height


def comparison_fixture():
    yy, xx = np.indices((320, 320))
    codes = (10000 + ((xx * 7 + yy * 13) % 20000)).astype(np.uint16)
    height = torch.from_numpy(numeric_height(codes)[None, None])
    rgb = torch.cat((height, height / 2, height / 3), dim=1)
    pair = {"metadata": {"sample_id": "surface_001"}, "sha256": {"input": "photo-identity", "height": "height-identity"}}
    model = TinyPretrained().eval()
    return model, pair, ArrayCache(rgb, height)


def export(model, pair, cache, folder, label, seed=17):
    return trainer.export_comparison(model, [pair], cache, folder, label, torch.device("cpu"), 256, seed)


def test_comparison_shared_originals_reused_across_labels_with_exact_native_source_identity(tmp_path):
    pytest.importorskip("OpenEXR")
    model, pair, cache = comparison_fixture()
    first = export(model, pair, cache, tmp_path, "pretrained-base")
    directory = tmp_path / "surface_001"
    shared_names = ("reference-height.exr", "reference-height.png", "reference-relief-8x.png", "source.png")
    originals = {name: (digest(directory / name), (directory / name).stat().st_mtime_ns) for name in shared_names}
    with torch.no_grad():
        model.ups[3].bias.add_(.1)
    second = export(model, pair, cache, tmp_path, "refined-checkpoint")
    assert {name: (digest(directory / name), (directory / name).stat().st_mtime_ns) for name in shared_names} == originals
    assert first[0]["model_label"] == "pretrained-base" and second[0]["model_label"] == "refined-checkpoint"
    assert first[0]["source_rectangle"] == second[0]["source_rectangle"]
    assert first[0]["native_size"] == second[0]["native_size"] == 256
    assert second[0]["height_min"] > first[0]["height_min"]
    assert digest(Path(first[0]["height_exr"])) != digest(Path(second[0]["height_exr"]))
    metadata = json.loads((directory / "source-identity.json").read_text())
    assert metadata["identity"]["source_sha256"] == pair["sha256"]
    assert metadata["identity"]["native_size"] == 256
    assert len(metadata["identity"]["height_float32_sha256"]) == 64
    assert metadata["files"] == {name: originals[name][0] for name in shared_names}


@pytest.mark.parametrize("problem", ["new-crop", "changed-source", "changed-reference", "missing-sidecar", "malformed-sidecar", "non-dict-sidecar"])
def test_comparison_rejects_stale_shared_sources_before_writing_candidate(tmp_path, problem):
    pytest.importorskip("OpenEXR")
    model, pair, cache = comparison_fixture()
    export(model, pair, cache, tmp_path, "base")
    directory = tmp_path / "surface_001"
    seed = 17
    if problem == "new-crop":
        seed = 23
    elif problem == "changed-source":
        pair = copy.deepcopy(pair)
        pair["sha256"]["input"] = "new-source-identity"
    elif problem == "changed-reference":
        (directory / "reference-height.png").write_bytes(b"changed data")
    elif problem == "missing-sidecar":
        (directory / "source-identity.json").unlink()
    elif problem == "malformed-sidecar":
        (directory / "source-identity.json").write_text("not JSON")
    elif problem == "non-dict-sidecar":
        (directory / "source-identity.json").write_text("[]")
    with pytest.raises(ValueError, match="source|reference|identity"):
        export(model, pair, cache, tmp_path, "candidate", seed)
    assert not (directory / "candidate-height.exr").exists()
    assert not (directory / "candidate-height.png").exists()
