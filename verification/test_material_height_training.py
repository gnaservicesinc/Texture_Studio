"""Precision/split/checkpoint regression tests for the material pilot."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from material_dataset import read_png, write_png
from material_height_model import ARCHITECTURE, SCHEMA, MaterialHeightNet, height_loss
import train_material_height as training_cli
from train_material_height import digest, evaluate, find_samples, linear_rgb, load_checkpoint, load_pair, opengl_normal_from_height, prediction_metrics, probe_native_sizes, save_prediction, train, write_float_exr


def fixture_sample(directory: Path, material: str, split: str, rgba: bool = False, size: int = 32) -> dict:
    directory.mkdir(parents=True)
    height = np.resize(np.array([32767, 32768, 32769], dtype=np.uint16), (size, size, 1))
    if rgba:
        height = np.concatenate((np.repeat(height, 3, axis=-1), np.full((size, size, 1), 65535, dtype=np.uint16)), axis=-1)
    diffuse = np.full((size, size, 3), 32768, dtype=np.uint16)
    write_png(directory / "diffuse.png", diffuse)
    write_png(directory / "displacement.png", height)
    metadata = {"schema_version": 2, "sample_id": directory.name, "material_id": material, "split": split, "status": "prepared", "maps": {"input": "diffuse.png", "height": "displacement.png"}, "source_precision_verified": True, "crop_values_verified": True, "sample_pixel_dimensions": [size, size], "map_metadata": {"input": {"encoding": "linear"}, "height": {"encoding": "linear_data"}}}
    metadata["map_metadata"]["input"]["sample_sha256"] = digest(directory / "diffuse.png")
    metadata["map_metadata"]["height"]["sample_sha256"] = digest(directory / "displacement.png")
    (directory / "sample.json").write_text(json.dumps(metadata))
    return {"metadata": metadata, "metadata_path": directory / "sample.json", "input_path": directory / "diffuse.png", "height_path": directory / "displacement.png"}


@pytest.mark.parametrize("rgba", [False, True])
def test_adjacent_height_codes_survive_float32_without_gamma(tmp_path, rgba):
    sample = fixture_sample(tmp_path / "sample", "one", "train", rgba)
    source, target = load_pair(sample, torch.device("cpu"))
    assert source.dtype == target.dtype == torch.float32
    assert source[0, 0, 0, 0] == np.float32(32768) / np.float32(65535)
    expected = np.array([32767, 32768, 32769], dtype=np.float32) / np.float32(65535)
    np.testing.assert_array_equal(target.numpy()[0, 0, 0, :3], expected)
    assert torch.unique(target).numel() == 3


def test_nonidentical_rgb_height_is_rejected(tmp_path):
    sample = fixture_sample(tmp_path / "sample", "one", "train", True)
    height = np.full((32, 32, 4), 65535, dtype=np.uint16)
    height[..., 0] = 32768
    sample["height_path"].unlink()
    write_png(sample["height_path"], height)
    sample["metadata"]["map_metadata"]["height"]["sample_sha256"] = digest(sample["height_path"])
    with pytest.raises(ValueError, match="channels differ"):
        load_pair(sample, torch.device("cpu"))


def test_nonopaque_height_is_rejected(tmp_path):
    sample = fixture_sample(tmp_path / "sample", "one", "train", True)
    height = np.full((32, 32, 4), 32768, dtype=np.uint16)
    sample["height_path"].unlink()
    with pytest.raises(ValueError, match="alpha"):
        write_png(sample["height_path"], height)
        sample["metadata"]["map_metadata"]["height"]["sample_sha256"] = digest(sample["height_path"])
        load_pair(sample, torch.device("cpu"))


def test_dataset_index_ignores_orphan_and_rejects_material_leakage(tmp_path):
    first = fixture_sample(tmp_path / "samples" / "one", "same", "train")
    second = fixture_sample(tmp_path / "samples" / "two", "same", "validation")
    fixture_sample(tmp_path / "samples" / "orphan", "ignored", "train")
    items = [{**{key: sample["metadata"][key] for key in ("sample_id", "material_id", "status", "split")}, "path": "samples/" + sample["metadata"]["sample_id"]} for sample in (first, second)]
    (tmp_path / "dataset.json").write_text(json.dumps({"schema_version": 2, "samples": items}))
    with pytest.raises(ValueError, match="cross splits"):
        find_samples(tmp_path, True)
    second["metadata"]["material_id"] = "other"
    (second["metadata_path"]).write_text(json.dumps(second["metadata"]))
    items[1]["material_id"] = "other"
    (tmp_path / "dataset.json").write_text(json.dumps({"schema_version": 2, "samples": items}))
    assert len(find_samples(tmp_path, True)) == 2


def test_native_output_gradient_step_and_safe_checkpoint(tmp_path):
    torch.manual_seed(1)
    model = MaterialHeightNet(4)
    source = torch.rand(1, 3, 33, 35)
    target = torch.rand(1, 1, 33, 35)
    prediction = model(source)
    assert prediction.shape == target.shape
    loss, _ = height_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert torch.count_nonzero(model.head.weight.grad)
    torch.optim.AdamW(model.parameters()).step()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 4}, "model_state": model.state_dict()}, checkpoint)
    restored, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    torch.testing.assert_close(restored(source), model(source), rtol=0, atol=0)


@pytest.mark.parametrize("channels", [1, 3])
def test_float32_zip_exr_preserves_adjacent_float_samples(tmp_path, channels):
    shape = (8, 9) if channels == 1 else (8, 9, 3)
    values = np.resize(np.array([0.5, np.nextafter(np.float32(0.5), np.float32(1)), 1 / 65535], dtype=np.float32), shape)
    write_float_exr(tmp_path / "numeric.exr", values)


def test_opengl_normal_axes_and_upwards_y():
    xramp = np.broadcast_to(np.linspace(0, 1, 32, dtype=np.float32)[None, :], (32, 32))
    yramp = xramp.T
    xnormal, ynormal = opengl_normal_from_height(xramp), opengl_normal_from_height(yramp)
    assert np.all(xnormal[..., 0] < 0.5)
    np.testing.assert_array_equal(xnormal[..., 1], 0.5)
    np.testing.assert_array_equal(ynormal[..., 0], 0.5)
    assert np.all(ynormal[..., 1] > 0.5)


def test_offset_invariant_loss_changes_no_source_amplitudes():
    prediction = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    original = target.clone()
    first, _ = height_loss(prediction, target, offset_invariant=True, highpass_weight=4)
    second, _ = height_loss(prediction + 0.25, target - 0.1, offset_invariant=True, highpass_weight=4)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(target, original, rtol=0, atol=0)


def test_rectangular_normals_use_same_pixel_spacing_on_both_axes():
    x, y = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(32, dtype=np.float32))
    xnormal = opengl_normal_from_height(x / 100)
    ynormal = opengl_normal_from_height(y / 100)
    np.testing.assert_allclose(0.5 - xnormal[..., 0], ynormal[..., 1] - 0.5, rtol=0, atol=1e-7)


def test_prepared_map_edit_is_rejected_before_training(tmp_path):
    sample = fixture_sample(tmp_path / "sample", "one", "train")
    sample["height_path"].unlink()
    write_png(sample["height_path"], np.full((32, 32, 1), 12345, dtype=np.uint16))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_pair(sample, torch.device("cpu"))


def test_inference_export_has_no_synthetic_target(tmp_path):
    save_prediction(tmp_path, torch.full((1, 1, 16, 16), 0.5), torch.full((1, 3, 16, 16), 0.5), None, "prediction")
    assert (tmp_path / "prediction.height.float32.exr").is_file()
    assert (tmp_path / "prediction.normal_opengl.float32.exr").is_file()
    assert not list(tmp_path.glob("target*"))


@pytest.mark.parametrize("height_channels", (1, 2))
def test_size_probe_uses_native_parent_crops_and_updates_real_weights(tmp_path, height_channels):
    sample = fixture_sample(tmp_path / "parent", "one", "train")
    if height_channels == 2:
        codes, _ = read_png(sample["height_path"])
        sample["height_path"].unlink()
        write_png(sample["height_path"], np.concatenate((codes, np.full_like(codes, 65531)), axis=-1))
    checkpoint = tmp_path / "checkpoint.pt"
    model = MaterialHeightNet(4)
    torch.save({"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 4}, "model_state": model.state_dict(), "loss": {"gradient_weight": 8.0, "offset_invariant_height_loss": True, "detail_highpass_weight": 4.0}}, checkpoint)
    original_input, original_height = digest(sample["input_path"]), digest(sample["height_path"])
    args = SimpleNamespace(device="cpu", source_input=sample["input_path"], source_height=sample["height_path"], input_encoding="linear", checkpoint=checkpoint, output=tmp_path / "probe", warmup_steps=0, steps=1, sizes=[16, 32])
    probe_native_sizes(args)
    result = json.loads((args.output / "probe.json").read_text())
    assert all(row["successful"] and row["head_parameters_changed"] for row in result["results"])
    assert [row["crop_rectangle_top_left_xywh"] for row in result["results"]] == [[0, 0, 16, 16], [0, 0, 32, 32]]
    assert digest(sample["input_path"]) == original_input
    assert digest(sample["height_path"]) == original_height
    if height_channels == 2:
        assert result["scalar_alpha_policy"]["alpha_used_to_scale_scalar"] is False


def test_size_probe_rejects_meaningful_height_alpha_before_training(tmp_path):
    sample = fixture_sample(tmp_path / "parent", "one", "train")
    codes, _ = read_png(sample["height_path"])
    sample["height_path"].unlink()
    write_png(sample["height_path"], np.concatenate((codes, np.full_like(codes, 65526)), axis=-1))
    before = digest(sample["height_path"])
    args = SimpleNamespace(device="cpu", source_input=sample["input_path"], source_height=sample["height_path"])
    with pytest.raises(ValueError, match="alpha requiring review"):
        probe_native_sizes(args)
    assert digest(sample["height_path"]) == before


def region_fixture(tmp_path: Path, second_rectangle: list[int], size: int = 32) -> None:
    samples = [fixture_sample(tmp_path / "samples" / name, "same", split, size=size) for name, split in (("train", "train"), ("validation", "validation"))]
    items = []
    for sample, rectangle in zip(samples, ([0, 0, size, size], second_rectangle)):
        metadata = sample["metadata"]
        metadata.update(split_strategy="heldout-region-v1", validation_scope="unseen_regions_of_known_materials", source_region_role=metadata["split"], crop_rectangle_top_left_xywh=rectangle)
        for role, fingerprint in (("input", "a" * 64), ("height", "b" * 64)):
            metadata["map_metadata"][role]["source"] = {"file_sha256": fingerprint, "width": size * 2, "height": size * 2}
        sample["metadata_path"].write_text(json.dumps(metadata))
        items.append({**{key: metadata[key] for key in ("sample_id", "material_id", "split", "status")}, "path": "samples/" + metadata["sample_id"]})
    (tmp_path / "dataset.json").write_text(json.dumps({"schema_version": 2, "split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "samples": items}))


def test_explicit_region_policy_accepts_touching_nonoverlapping_native_crops(tmp_path):
    region_fixture(tmp_path, [1024, 0, 1024, 1024], size=1024)
    samples = find_samples(tmp_path, True)
    assert len(samples) == 2
    assert samples[0]["validation_scope"] == "unseen regions of known materials"


def test_explicit_region_policy_rejects_overlapping_source_pixels(tmp_path):
    region_fixture(tmp_path, [31, 0, 32, 32])
    with pytest.raises(ValueError, match="source regions overlap"):
        find_samples(tmp_path, True)


def test_region_policy_cannot_be_inferred_from_sample_names(tmp_path):
    region_fixture(tmp_path, [32, 0, 32, 32])
    index = json.loads((tmp_path / "dataset.json").read_text())
    del index["split_strategy"]
    (tmp_path / "dataset.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Material identities cross splits"):
        find_samples(tmp_path, True)


def test_region_policy_requires_every_validation_material_in_training(tmp_path):
    region_fixture(tmp_path, [32, 0, 32, 32])
    index = json.loads((tmp_path / "dataset.json").read_text())
    index["samples"][1]["material_id"] = "another"
    metadata_path = tmp_path / "samples" / "validation" / "sample.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["material_id"] = "another"
    metadata_path.write_text(json.dumps(metadata))
    (tmp_path / "dataset.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="every material"):
        find_samples(tmp_path, True)


def test_training_metrics_labeled_and_artifact_limit_preserves_all_metrics(tmp_path):
    samples = [fixture_sample(tmp_path / "samples" / name, name, "train") for name in ("one", "two")]
    output = tmp_path / "metrics"
    output.mkdir()
    result = evaluate(MaterialHeightNet(4), samples, torch.device("cpu"), output, "training-fit", export_limit=1)
    assert result["sample_count"] == 2
    assert result["validation_scope"] == "training regions already used for optimization"
    assert result["prediction_exports"] == 1
    assert len(list((output / "predictions").glob("*/*.height.float32.exr"))) == 1


def test_nearopaque_alpha_tolerance_is_eight_uint16_codes_only():
    rgba = np.full((16, 16, 4), 65535, dtype=np.uint16)
    rgba[..., :3] = 32768
    rgba[..., 3] = 65527
    rgb = linear_rgb(rgba, "linear")
    np.testing.assert_array_equal(rgb, np.float32(32768) / np.float32(65535))
    rgba[0, 0, 3] = 65526
    with pytest.raises(ValueError, match="Transparent"):
        linear_rgb(rgba, "linear")


def test_alpha_mask_preserves_rgb_and_targets_with_eight_pixel_margin(tmp_path):
    sample = fixture_sample(tmp_path / "sample", "one", "train", size=128)
    rgba = np.full((128, 128, 4), 65535, dtype=np.uint16)
    rgba[..., :3] = 32768
    rgba[64, 64, 3] = 52966
    sample["input_path"].unlink()
    write_png(sample["input_path"], rgba)
    sample["metadata"]["map_metadata"]["input"]["sample_sha256"] = digest(sample["input_path"])
    with pytest.raises(ValueError, match="diffuse.png"):
        load_pair(sample, torch.device("cpu"))
    source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=True, return_mask=True)
    assert mask[0, 0, 56:73, 56:73].sum() == 0
    assert mask[0, 0, 55, 64] == 1
    assert int(mask.sum()) == 128 * 128 - 17 * 17
    np.testing.assert_array_equal(source.numpy(), np.float32(32768) / np.float32(65535))
    assert torch.unique(target).numel() == 3
    assert sample["source_alpha_excluded_pixel_fraction"] == 1 / (128 * 128)


def test_masked_loss_and_metrics_ignore_invalid_targets_and_gradients():
    prediction = torch.rand(1, 1, 64, 64, requires_grad=True)
    target = torch.rand(1, 1, 64, 64)
    mask = torch.ones_like(target)
    mask[..., 20:44, 20:44] = 0
    first_loss, _ = height_loss(prediction, target, offset_invariant=True, highpass_weight=4, mask=mask)
    first_metrics = prediction_metrics(prediction, target, mask)
    target[..., 26:38, 26:38] = 100
    second_loss, _ = height_loss(prediction, target, offset_invariant=True, highpass_weight=4, mask=mask)
    second_metrics = prediction_metrics(prediction, target, mask)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_metrics == second_metrics
    second_loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[..., 20:44, 20:44]) == 0


def test_large_alpha_exclusion_is_rejected(tmp_path):
    sample = fixture_sample(tmp_path / "sample", "one", "train", size=128)
    rgba = np.full((128, 128, 4), 32768, dtype=np.uint16)
    sample["input_path"].unlink()
    write_png(sample["input_path"], rgba)
    sample["metadata"]["map_metadata"]["input"]["sample_sha256"] = digest(sample["input_path"])
    with pytest.raises(ValueError, match="More than 10%"):
        load_pair(sample, torch.device("cpu"), mask_input_alpha=True, return_mask=True)


def test_warmstart_squared_training_and_sparse_validation_keep_best_validated(tmp_path, monkeypatch):
    region_fixture(tmp_path / "dataset", [32, 0, 32, 32])
    initial = tmp_path / "initial.pt"
    model = MaterialHeightNet(4)
    with torch.no_grad():
        model.head.weight.fill_(0.01)
    torch.save({"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 4}, "model_state": model.state_dict(), "step": 100}, initial)
    output = tmp_path / "run"
    args = SimpleNamespace(device="cpu", seed=2307, dataset=tmp_path / "dataset", allow_unreviewed=True, mask_input_alpha=False, output=output, base_channels=4, initial_checkpoint=initial, learning_rate=0.001, gradient_weight=16, offset_invariant_loss=True, highpass_weight=4, objective="relative-squared", validation_every_epochs=5, export_limit=0, baseline_directory=None, epochs=10, max_steps=3, max_minutes=0, evaluate_training=True)
    labels = []
    original_evaluate = training_cli.evaluate
    def recorded(*arguments, **kwargs):
        labels.append(arguments[4])
        return original_evaluate(*arguments, **kwargs)
    monkeypatch.setattr(training_cli, "evaluate", recorded)
    train(args)
    summary = json.loads((output / "summary.json").read_text())
    latest = json.loads((output / "evaluation.latest.json").read_text())
    checkpoint = torch.load(output / "checkpoint.best.pt", weights_only=True)
    assert summary["initial_checkpoint"]["sha256"] == digest(initial)
    assert summary["initial_checkpoint"]["trained_steps"] == 100
    for source in summary["implementation_sources"].values():
        assert digest(output / source["snapshot_relative_path"]) == source["sha256"]
    assert summary["optimizer_state_resumed"] is False
    assert summary["completed_steps"] == 3
    assert summary["best_checkpoint_validated_at_same_step"]
    assert labels == ["initial", "latest", "best", "training-fit"]
    assert checkpoint["validation_score"] <= latest["mean_objective_loss"]
    assert checkpoint["validation_step"] == checkpoint["step"]
    assert checkpoint["step"] in (0, 3)
    assert summary["best_validation_objective_score"] == checkpoint["validation_score"]
    assert not list(output.glob("predictions/*/*.exr"))


def test_warmstart_rejects_architecture_mismatch_before_creating_output(tmp_path):
    region_fixture(tmp_path / "dataset", [32, 0, 32, 32])
    initial = tmp_path / "initial.pt"
    model = MaterialHeightNet(4)
    torch.save({"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 4}, "model_state": model.state_dict(), "step": 100}, initial)
    args = SimpleNamespace(device="cpu", seed=2307, dataset=tmp_path / "dataset", allow_unreviewed=True, mask_input_alpha=False, output=tmp_path / "run", base_channels=8, initial_checkpoint=initial)
    with pytest.raises(ValueError, match="base_channels differs"):
        train(args)
    assert not args.output.exists()


def test_validated_warmstart_remains_best_when_new_steps_worsen(tmp_path, monkeypatch):
    region_fixture(tmp_path / "dataset", [32, 0, 32, 32])
    initial = tmp_path / "initial.pt"
    model = MaterialHeightNet(4)
    torch.save({"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": 4}, "model_state": model.state_dict(), "step": 100}, initial)
    args = SimpleNamespace(device="cpu", seed=2307, dataset=tmp_path / "dataset", allow_unreviewed=True, mask_input_alpha=False, output=tmp_path / "run", base_channels=4, initial_checkpoint=initial, learning_rate=0.001, gradient_weight=16, offset_invariant_loss=True, highpass_weight=4, objective="relative-squared", validation_every_epochs=5, export_limit=0, baseline_directory=None, epochs=2, max_steps=2, max_minutes=0, evaluate_training=False)
    original_evaluate = training_cli.evaluate
    def worsened(*arguments, **kwargs):
        result = original_evaluate(*arguments, **kwargs)
        result["mean_objective_loss"] = 10.0 if arguments[4] == "latest" else 1.0
        return result
    monkeypatch.setattr(training_cli, "evaluate", worsened)
    train(args)
    checkpoint = torch.load(args.output / "checkpoint.best.pt", weights_only=True)
    summary = json.loads((args.output / "summary.json").read_text())
    assert checkpoint["step"] == checkpoint["validation_step"] == 0
    assert checkpoint["validation_score"] == 1.0
    assert summary["completed_steps"] == 2
    assert summary["best_improved_over_initial_objective"] is False
