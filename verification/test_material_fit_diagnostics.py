"""Capacity-diagnostic losses, metrics and source-preservation regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_material_fit import fit_metrics, run, squared_objective
from material_dataset import write_png
from train_material_height import digest, load_checkpoint


def structured_target(size=32):
    x, y = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="xy")
    return (0.5 + 0.1 * torch.sin(x * 20) + 0.1 * torch.cos(y * 17))[None, None]


def test_squared_objective_preserves_constant_origin_and_target():
    target = structured_target()
    prediction = target * 0.8
    original = target.clone()
    first, _ = squared_objective(prediction, target)
    second, _ = squared_objective(prediction + 0.1, target - 0.2)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(target, original, rtol=0, atol=0)


def test_squared_objective_distinguishes_lost_amplitude():
    target = structured_target()
    perfect, _ = squared_objective(target, target)
    small, _ = squared_objective(0.5 + (target - 0.5) * 0.2, target)
    flat, _ = squared_objective(torch.full_like(target, 0.5), target)
    assert perfect == 0
    assert flat > small > perfect
    torch.testing.assert_close(flat, torch.tensor(3.0), rtol=1e-5, atol=1e-5)


def test_squared_masked_values_and_gradients_are_excluded():
    target = structured_target(64)
    prediction = torch.full_like(target, 0.5, requires_grad=True)
    mask = torch.ones_like(target)
    mask[..., 20:44, 20:44] = 0
    first, _ = squared_objective(prediction, target, mask)
    changed = target.clone()
    changed[..., 26:38, 26:38] = 100
    second, _ = squared_objective(prediction, changed, mask)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    second.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[..., 20:44, 20:44]) == 0


def test_flat_targets_have_finite_loss_and_undefined_correlation():
    flat = torch.full((1, 1, 32, 32), 0.5)
    loss, components = squared_objective(flat, flat)
    assert loss == 0
    assert all(np.isfinite(value) for value in components.values())
    metrics = fit_metrics(flat, flat)
    assert metrics["centered_height_correlation"] is None
    assert metrics["gradient_vector_cosine"] is None
    json.dumps(metrics, allow_nan=False)


def test_metric_correlation_sign_gain_and_amplitude():
    target = structured_target()
    perfect = fit_metrics(target, target)
    inverted = fit_metrics(1 - target, target)
    scaled = fit_metrics(0.5 + (target - 0.5) * 0.25, target)
    assert perfect["centered_height_correlation"] == pytest.approx(1, abs=1e-6)
    assert perfect["native_gradient_rms_amplitude_ratio"] == pytest.approx(1, abs=1e-6)
    assert inverted["centered_height_correlation"] == pytest.approx(-1, abs=1e-6)
    assert inverted["gradient_vector_cosine"] == pytest.approx(-1, abs=1e-6)
    assert scaled["centered_height_signed_gain"] == pytest.approx(0.25, abs=1e-6)
    assert scaled["height_amplitude_std_ratio"] == pytest.approx(0.25, abs=1e-6)
    assert scaled["native_gradient_rms_amplitude_ratio"] == pytest.approx(0.25, abs=1e-6)


def dataset_fixture(root: Path):
    items = []
    for number, split in enumerate(("train", "validation")):
        identity = f"material_auto_{number + 1:03d}"
        path = root / "samples" / identity
        path.mkdir(parents=True)
        target = structured_target()[0, 0].numpy()
        codes = np.rint(target * 65535).astype(np.uint16)
        rgb = np.repeat(codes[..., None], 3, axis=-1)
        write_png(path / "diffuse.png", rgb)
        write_png(path / "displacement.png", codes[..., None])
        metadata = {"schema_version": 2, "sample_id": identity, "material_id": "material", "split": split, "status": "prepared", "source_precision_verified": True, "crop_values_verified": True, "sample_pixel_dimensions": [32, 32], "crop_rectangle_top_left_xywh": [number * 32, 0, 32, 32], "split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "source_region_role": split, "maps": {"input": "diffuse.png", "height": "displacement.png"}, "map_metadata": {"input": {"encoding": "linear", "sample_sha256": digest(path / "diffuse.png"), "source": {"file_sha256": "a" * 64, "width": 64, "height": 32}}, "height": {"encoding": "linear_data", "sample_sha256": digest(path / "displacement.png"), "source": {"file_sha256": "b" * 64, "width": 64, "height": 32}}}}
        (path / "sample.json").write_text(json.dumps(metadata))
        items.append({key: metadata[key] for key in ("sample_id", "material_id", "split", "status")})
        items[-1]["path"] = "samples/" + identity
    (root / "dataset.json").write_text(json.dumps({"schema_version": 2, "split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "samples": items}))


def test_real_cpu_diagnostic_keeps_sources_fixed_and_updates_encoder(tmp_path):
    dataset = tmp_path / "dataset"
    dataset_fixture(dataset)
    hashes = {str(path): digest(path) for path in dataset.rglob("*") if path.is_file()}
    args = SimpleNamespace(device="cpu", dataset=dataset, material="material", sample_id=None, validation_sample_id=None, skip_validation=False, allow_unreviewed=True, mask_transparent_input=False, expected_size=32, output=tmp_path / "output", seed=2307, learning_rate=0.001, steps=3, base_channels=4, variants=["current_l1", "relative_squared"], max_minutes=0, log_every=3)
    report = run(args)
    assert report["not_generalization_validation"]
    assert report["artifact_bytes"] < 200 * 1024 * 1024
    assert len(list(args.output.glob("*/checkpoint.final.pt"))) == 2
    for variant in args.variants:
        result = report["variants"][variant]
        assert result["completed_steps"] == 3
        assert result["first_real_step"]["head_parameters_changed"]
        assert result["second_step_upstream_gradients"]["encoder_gradient_nonzero"]
        assert result["second_step_upstream_gradients"]["encoder_parameters_changed"]
        model, checkpoint = load_checkpoint(args.output / variant / "checkpoint.final.pt", torch.device("cpu"))
        assert checkpoint["diagnostic_only_not_production"]
        assert checkpoint["step"] == 3
        assert model(torch.rand(1, 3, 32, 32)).shape == (1, 1, 32, 32)
    assert hashes == {str(path): digest(path) for path in dataset.rglob("*") if path.is_file()}
