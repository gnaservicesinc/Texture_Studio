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


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_head = nn.Conv2d(3, 4, 1)
        self.noise = nn.Dropout(.9)
        self.m_dec_3 = nn.Conv2d(4, 4, 1)
        self.m_dec_3.register_buffer("relative_position_index", torch.arange(16).reshape(4, 4))
        self.m_dec_3.register_buffer("relative_coords_table", torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        self.m_tail_3 = nn.Conv2d(4, 4, 1)
        self.m_dec_0 = nn.Conv2d(4, 4, 1)

    def forward(self, rgb):
        shared = self.noise(self.m_head(rgb))
        return self.m_tail_3(self.m_dec_3(shared)) + self.m_dec_0(shared)


class TinyPretrained(nn.Module):
    """Small CPU architecture fixture; tests the actual optimizer/state path."""
    def __init__(self):
        super().__init__()
        self.gen = TinyGenerator()
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


def checkpoint_fixture(tmp_path, scope="final-height"):
    model = TinyPretrained()
    parameters = trainer.configure_refinement(model, scope)
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0)
    rgb, target = torch.full((1, 3, 256, 256), .4), torch.full((1, 1, 256, 256), .6)
    initial = trainer.predict(model, rgb).detach().clone()
    loss = (trainer.predict(model, rgb) - target).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, model, optimizer, 1, {"dataset": {"index_sha256": "test-index"}}, settings(scope=scope))
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
    assert not model.training and not model.gen.noise.training
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
    assert "_snapshot_sha256" not in saved
    restored = TinyPretrained()
    trainer.configure_refinement(restored)
    frozen = {name: tensor.clone() for name, tensor in restored.state_dict().items() if not name.startswith("ups.3.")}
    result = trainer.restore_refinement(restored, path)
    assert result["step"] == 1
    assert_same_weights(frozen, {name: tensor for name, tensor in restored.state_dict().items() if name in frozen})
    torch.testing.assert_close(trainer.predict(restored, rgb), trainer.predict(refined, rgb), rtol=0, atol=0)
    assert not torch.equal(trainer.predict(restored, rgb), initial)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("inspection_size", [256, 512])
def test_evaluation_labels_loaded_checkpoint_snapshot_when_path_is_replaced(tmp_path, monkeypatch, inspection_size):
    refined, _, checkpoint, rgb, _ = checkpoint_fixture(tmp_path)
    original_checksum = digest(checkpoint)
    replacement_state = torch.load(checkpoint, weights_only=True)
    replacement_state["height_decoder_state"]["ups.3.bias"].add_(.2)
    replacement_state["step"] = 2
    pair = {"dimensions": [inspection_size, inspection_size], "metadata": {"sample_id": "surface_001"}}
    monkeypatch.setattr(trainer, "select_pairs", lambda *_args: ([pair], [], {"fixture": "dataset"}))
    monkeypatch.setattr(trainer, "load_complete_pretrained", lambda *_args, **_kwargs: TinyPretrained())
    monkeypatch.setattr(trainer.signal, "signal", lambda *_args: None)

    def export_snapshot(model, _pairs, _cache, folder, label, _device, size, _seed):
        if label == "pretrained-base":
            replacement = tmp_path / "replacement.pt"
            torch.save(replacement_state, replacement)
            replacement.replace(checkpoint)
        else:
            torch.testing.assert_close(trainer.predict(model, rgb), trainer.predict(refined, rgb), rtol=0, atol=0)
        return [{"sample": "surface_001", "model_label": label, "native_size": size,
                 "source_rectangle": [0, 0, size, size], "source_png": str(folder / "source.png"),
                 "reference_exr": str(folder / "reference.exr"), "height_exr": str(folder / f"{label}.exr")}]

    monkeypatch.setattr(trainer, "export_comparison", export_snapshot)
    output = tmp_path / "evaluation"
    assert trainer.main(["evaluate", "--dataset", str(tmp_path / "dataset"), "--output", str(output),
                         "--checkpoint", str(checkpoint), "--source-dir", str(tmp_path / "source"),
                         "--weights", str(tmp_path / "weights"), "--size", str(inspection_size), "--memory-gib", "48",
                         "--cache-gib", "0", "--device", "cpu"]) == 0
    assert digest(checkpoint) != original_checksum
    report = json.loads((output / "run.json").read_text())
    assert report["operation"] == "evaluate"
    assert report["training_crop_size"] == 256 and report["inspection_crop_size"] == inspection_size
    assert [record["model_label"] for record in report["comparison"]] == ["pretrained-base", "refined-checkpoint"]
    assert report["parent"]["sha256"] == report["checkpoint_sha256"] == original_checksum
    assert report["saved_step"] == report["parent"]["step"] == 1
    variants = json.loads((output / "review-manifest.json").read_text())["materials"][0]["variants"]
    assert all(variant["checkpoint_sha256"] == original_checksum for variant in variants if variant.get("checkpoint"))
    assert len(variants) == 3  # Original target, exact base, and one checkpoint.
    assert "_snapshot_sha256" not in report


def test_height_decoder_scope_updates_only_existing_pretrained_height_path():
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = trainer.configure_refinement(model, "height-decoder")
    prefixes = ("gen.m_dec_3.", "gen.m_tail_3.", "ups.3.")
    expected = {name for name, _parameter in model.named_parameters() if name.startswith(prefixes)}
    assert len(parameters) == 6
    assert {name for name, value in model.named_parameters() if value.requires_grad} == expected
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0)
    trainer.predict(model, torch.full((1, 3, 256, 256), .4)).sub(.6).square().mean().backward()
    optimizer.step()
    assert all(not torch.equal(before[name], model.state_dict()[name]) for name in expected)
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before if name not in expected)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in expected)
    assert not model.gen.noise.training


def test_height_decoder_delta_roundtrip_preserves_generator_state_for_evaluation(tmp_path):
    refined, _, path, rgb, _ = checkpoint_fixture(tmp_path, "height-decoder")
    saved = trainer.read_refinement_checkpoint(path)
    assert saved["refinement_scope"] == saved["configuration"]["scope"] == "height-decoder"
    assert saved["trainable_prefixes"] == ["gen.m_dec_3.", "gen.m_tail_3.", "ups.3."]
    assert set(saved["height_decoder_state"]) == {name for name in refined.state_dict()
                                                if name.startswith(tuple(saved["trainable_prefixes"]))}
    assert trainer.resolve_scope("evaluate", None, saved) == "height-decoder"
    assert trainer.resolve_scope("refine", None, saved) == "height-decoder"
    assert saved["height_decoder_state"]["gen.m_dec_3.relative_position_index"].dtype == torch.int64
    restored = TinyPretrained()
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    trainer.restore_refinement(restored, path)
    torch.testing.assert_close(trainer.predict(restored.eval(), rgb), trainer.predict(refined, rgb), rtol=0, atol=0)
    assert all(torch.equal(before[name], restored.state_dict()[name]) for name in before
               if name not in saved["height_decoder_state"])
    with pytest.raises(ValueError, match="cannot be narrowed"):
        trainer.resolve_scope("refine", "final-height", saved)


def test_legacy_final_height_checkpoint_can_widen_without_losing_parent_weights(tmp_path):
    refined, _, path, rgb, _ = checkpoint_fixture(tmp_path)
    saved = torch.load(path, weights_only=True)
    saved.pop("refinement_scope")
    saved.pop("trainable_prefixes")
    saved["configuration"].pop("scope")
    torch.save(saved, path)
    assert trainer.checkpoint_scope(saved) == "final-height"
    assert trainer.resolve_scope("train", None, None) == "final-height"
    assert trainer.resolve_scope("refine", None, saved) == "final-height"
    scope = trainer.resolve_scope("refine", "height-decoder", saved)
    widened = TinyPretrained()
    generator = {name: value.clone() for name, value in widened.state_dict().items() if name.startswith("gen.")}
    trainer.configure_refinement(widened, scope)
    trainer.restore_refinement(widened, path)
    assert all(torch.equal(value, widened.state_dict()[name]) for name, value in generator.items())
    torch.testing.assert_close(trainer.predict(widened, rgb), trainer.predict(refined, rgb), rtol=0, atol=0)
    optimizer = torch.optim.AdamW([value for value in widened.parameters() if value.requires_grad], lr=.01)
    widened_path = tmp_path / "widened.pt"
    trainer.save_checkpoint(widened_path, widened, optimizer, 1, {"dataset": {}}, settings(scope=scope))
    widened_saved = trainer.read_refinement_checkpoint(widened_path)
    assert widened_saved["refinement_scope"] == "height-decoder"
    assert "gen.m_dec_3.weight" in widened_saved["height_decoder_state"]
    assert torch.equal(widened_saved["height_decoder_state"]["ups.3.weight"], saved["height_decoder_state"]["ups.3.weight"])
    with pytest.raises(ValueError, match="Evaluate uses"):
        trainer.resolve_scope("evaluate", "height-decoder", saved)


@pytest.mark.parametrize("mutation", ["unknown-scope", "not-string", "wrong-prefix", "missing-prefixes", "missing-scope", "configuration-mismatch", "missing-configuration-scope"])
def test_invalid_scope_metadata_rejected_before_model_mutation(tmp_path, mutation):
    _, _, path, _, _ = checkpoint_fixture(tmp_path, "height-decoder")
    saved = torch.load(path, weights_only=True)
    if mutation == "unknown-scope":
        saved["refinement_scope"] = "all-network"
    elif mutation == "not-string":
        saved["refinement_scope"] = []
    elif mutation == "wrong-prefix":
        saved["trainable_prefixes"].append("gen.m_head.")
    elif mutation == "missing-prefixes":
        saved.pop("trainable_prefixes")
    elif mutation == "missing-scope":
        saved.pop("refinement_scope")
    elif mutation == "configuration-mismatch":
        saved["configuration"]["scope"] = "final-height"
    elif mutation == "missing-configuration-scope":
        saved["configuration"].pop("scope")
    torch.save(saved, path)
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="scope"):
        trainer.restore_refinement(model, path)
    assert_same_weights(before, model.state_dict())


def test_height_decoder_requires_every_pretrained_branch_before_changing_trainability():
    model = TinyPretrained()
    del model.gen.m_tail_3
    before = [value.requires_grad for value in model.parameters()]
    with pytest.raises(ValueError, match="Pretrained height decoder"):
        trainer.configure_refinement(model, "height-decoder")
    assert [value.requires_grad for value in model.parameters()] == before
    with pytest.raises(ValueError, match="Refinement scope"):
        trainer.configure_refinement(model, "all-network")


@pytest.mark.parametrize("buffer", ["relative_position_index", "relative_coords_table"])
def test_height_decoder_checkpoint_rejects_changed_pretrained_layout_buffer(tmp_path, buffer):
    _, _, path, _, _ = checkpoint_fixture(tmp_path, "height-decoder")
    saved = torch.load(path, weights_only=True)
    saved["height_decoder_state"][f"gen.m_dec_3.{buffer}"][0, 0] += 1
    torch.save(saved, path)
    model = TinyPretrained()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="layout buffers"):
        trainer.restore_refinement(model, path)
    assert_same_weights(before, model.state_dict())


def test_review_manifest_declares_actual_model_input_diffuse_transfer(tmp_path):
    record = {"sample": "surface_001", "source_png": "/review/source.png", "reference_exr": "/review/reference.exr",
              "model_label": "pretrained-base", "height_exr": "/review/base.exr", "native_size": 1024,
              "source_rectangle": [0, 0, 1024, 1024]}
    path = tmp_path / "review.json"
    trainer.write_review_manifest({"comparison": [record]}, path)
    group = json.loads(path.read_text())["materials"][0]
    assert group["diffuse_encoding"] == "sRGB"
    assert group["target_original_bits"] == 16
    assert group["variants"][1]["checkpoint_sha256"] == trainer.WEIGHTS_SHA256


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
    wider = trainer.resource_plan(1024, 56, 4, "height-decoder")
    assert wider["driver_estimate_gib"] > plan["driver_estimate_gib"]
    assert wider["combined_estimate_gib"] == 41 and wider["refinement_scope"] == "height-decoder"
    assert "Measured1024" in wider["estimate_basis"]
    with pytest.raises(ValueError, match="smaller real training crop"):
        trainer.main(["train", "--dataset", str(tmp_path / "missing"), "--output", str(output),
                      "--size", "2048", "--memory-gib", "56", "--cache-gib", "4", "--device", "cpu", "--scope", "height-decoder"])


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
