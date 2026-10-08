"""Real tiny CPU optimizer, resumability and untouched numeric map contracts."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import material_training_cycle as cycle
from frozen_dino_height import MODEL_SHA256, CODE_REVISION, state_sha256
from material_dataset import read_png, write_png
from test_material_curriculum import make_dataset
from train_material_height import digest, find_samples


@pytest.mark.parametrize("command,required", (
    ("train", ["--dataset", "missing-dataset"]),
    ("resume", ["--resume-checkpoint", "missing-checkpoint.pt"]),
    ("probe", ["--dataset-1024", "missing-1k", "--dataset-2048", "missing-2k",
               "--warm-start", "missing-checkpoint.pt"]),
))
def test_archival_cli_rejects_without_opt_in_before_loading_or_setup(
        command, required, tmp_path, monkeypatch, capsys):
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["material_training_cycle.py", command,
        *required, "--output", str(output)])
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Retired CLI reached checkpoint loading or training setup")
    monkeypatch.setattr(cycle.torch, "load", forbidden)
    monkeypatch.setattr(cycle, "run_train", forbidden)
    monkeypatch.setattr(cycle, "run_probe", forbidden)
    monkeypatch.setattr(cycle, "configure_training_resources", forbidden)
    with pytest.raises(SystemExit) as stopped:
        cycle.main()
    assert stopped.value.code == 2
    assert "--allow-retired-experiment" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("command,required", (
    ("train", ["--dataset", "historical-dataset"]),
    ("resume", ["--resume-checkpoint", "historical-checkpoint.pt"]),
    ("probe", ["--dataset-1024", "historical-1k", "--dataset-2048", "historical-2k",
               "--warm-start", "historical-checkpoint.pt"]),
))
def test_archival_cli_explicit_opt_in_warns_and_preserves_command_dispatch(
        command, required, tmp_path, monkeypatch, capsys):
    calls = []
    def archived_run(settings):
        calls.append(settings)
        return {"status": "complete", "selected_step": 4}
    def checkpoint_load(path, **kwargs):
        assert command == "resume" and path == Path("historical-checkpoint.pt")
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"run_settings": {"learning_rate": 0.002,
                                  "allow_retired_experiment": False}}
    monkeypatch.setattr(sys, "argv", ["material_training_cycle.py", command,
        "--allow-retired-experiment", *required, "--output", str(tmp_path / "archive")])
    monkeypatch.setattr(cycle.torch, "load", checkpoint_load)
    monkeypatch.setattr(cycle, "run_train", archived_run)
    monkeypatch.setattr(cycle, "run_probe", archived_run)
    cycle.main()
    assert len(calls) == 1 and calls[0].command == command
    assert calls[0].allow_retired_experiment
    if command == "resume":
        assert calls[0].learning_rate == 0.002
    captured = capsys.readouterr()
    assert "WARNING:" in captured.err and "DINOv2 material training is retired" in captured.err
    assert json.loads(captured.out)["event"] == "complete"
    assert not (tmp_path / "archive").exists()


def dataset_with_two_corners(root):
    make_dataset(root)
    index = json.loads((root / "dataset.json").read_text())
    items = []
    for item in index["samples"]:
        path = root / item["path"] / "sample.json"
        metadata = json.loads(path.read_text())
        for role in ("input", "height"):
            metadata["map_metadata"][role]["source"]["height"] = 64
        size = 32
        normal = np.full((size, size, 3), 32768, dtype=np.uint16)
        normal[..., 2] = 65535
        rough = np.arange(size * size, dtype=np.uint16).reshape(size, size, 1) * 40
        for role, codes in (("normal", normal), ("roughness", rough)):
            name = role + ".png"
            write_png(path.parent / name, codes)
            metadata["maps"][role] = name
            metadata["map_metadata"][role] = {"encoding": "linear_data", "sample_sha256": digest(path.parent / name), "normal_convention": "OpenGL +Y", "source": {"suffix": "nor_gl", "file_sha256": metadata["map_metadata"]["height"]["source"]["file_sha256"], "width": 64, "height": 64}}
        path.write_text(json.dumps(metadata))
        items.append(item)
        if metadata["split"] == "train":
            duplicate = copy.deepcopy(metadata)
            duplicate["sample_id"] = metadata["sample_id"].replace("001", "002")
            duplicate["crop_rectangle_top_left_xywh"] = [0, 32, 32, 32]
            destination = root / "samples" / duplicate["sample_id"]
            destination.mkdir()
            for filename in metadata["maps"].values():
                (destination / filename).write_bytes((path.parent / filename).read_bytes())
            (destination / "sample.json").write_text(json.dumps(duplicate))
            entry = {key: duplicate[key] for key in ("sample_id", "material_id", "split", "status")}
            entry["path"] = "samples/" + duplicate["sample_id"]
            items.append(entry)
    index["samples"] = items
    (root / "dataset.json").write_text(json.dumps(index))


def arguments(tmp_path, **overrides):
    root = tmp_path / "dataset"
    if not root.exists():
        dataset_with_two_corners(root)
    args = SimpleNamespace(command="train", dataset=root, output=tmp_path / "cycle", device="cpu",
        materials=["white_stucco_02"], target="height", expected_size=32, updates_per_crop=2,
        seed=2307, encoder_size=28, model_directory=tmp_path, code_directory=tmp_path,
        warm_start=None, warm_start_sha256=None, resume_checkpoint=None, base_channels=4,
        projection_channels=4, learning_rate=0.001, weight_decay=1e-4, checkpoint_every=1,
        evaluate_every=4, max_minutes=15, max_driver_bytes=cycle.MAX_DRIVER_BYTES,
        selection="final", prediction_limit=2, allow_unreviewed=True, mask_transparent_input=False,
        export_split="paired", write_npy=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def tiny_encoder_loader(_model, _code, device):
    encoder = torch.nn.Linear(1, 1).requires_grad_(False).eval().to(device)
    return encoder, {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION, "test_fixture": True}


@torch.no_grad()
def tiny_features(_encoder, source, _size):
    return torch.nn.functional.avg_pool2d(source.mean(dim=1, keepdim=True), 16).repeat(1, 768, 1, 1).detach()


def run(args):
    return cycle.run_train(args, encoder_loader=tiny_encoder_loader, feature_extractor=tiny_features)


def test_balanced_rounds_preserve_both_corners_and_resumable_prefix():
    first, extended = cycle.balanced_schedule(8, 100, 2307), cycle.balanced_schedule(8, 200, 2307)
    assert Counter(first) == {index: 100 for index in range(8)}
    assert extended[:len(first)] == first
    assert all(set(first[start:start + 8]) == set(range(8)) for start in range(0, len(first), 8))
    with pytest.raises(ValueError):
        cycle.balanced_schedule(0, 2, 1)


def test_actual_cpu_train_native_export_and_no_source_changes(tmp_path):
    args = arguments(tmp_path)
    hashes = {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    report = run(args)
    assert report["complete_schedule"] and report["completed_steps"] == 4
    assert report["per_crop_update_counts"] == {"white_stucco_02_auto_001": 2, "white_stucco_02_auto_002": 2}
    assert report["selection_policy"] == "final" and report["selected_step"] == 4
    assert report["exported_sample_ids"] == ["white_stucco_02_auto_001", "white_stucco_02_auto_003"]
    assert report["target_precision_bits"] == {"white_stucco_02_auto_001": 16, "white_stucco_02_auto_002": 16, "white_stucco_02_auto_003": 16}
    assert report["source_files_verified_unchanged"] and not report["target_resized"]
    assert hashes == {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    checkpoint = torch.load(args.output / "checkpoint.latest.pt", weights_only=True)
    assert checkpoint["optimizer_state"]["state"] and checkpoint["complete_schedule"]
    assert state_sha256(checkpoint["head_state"]) != report["initial_head_state_sha256"]
    assert [entry["step"] for entry in report["history"]] == [0, 4]
    for identity in report["exported_sample_ids"]:
        import OpenEXR
        values = OpenEXR.File(str(args.output / "predictions" / identity / "selected.height.float32.exr"), separate_channels=True).channels()["Y"].pixels
        assert values.shape == (32, 32) and values.dtype == np.float32
        assert (args.output / "predictions" / identity / "selected.normal_opengl.float32.exr").is_file()
    assert not list(args.output.rglob("*.npy"))
    json.dumps(report, allow_nan=False)


def test_resume_optimizer_is_identical_to_uninterrupted_updates(tmp_path, monkeypatch):
    args = arguments(tmp_path, prediction_limit=0)
    full = run(args)
    expected = torch.load(args.output / "checkpoint.latest.pt", weights_only=True)
    initial_guard = cycle.Guard.check
    updates = {"count": 0}
    def stop_after_two(guard, stage):
        initial_guard(guard, stage)
        if stage == "after optimizer":
            updates["count"] += 1
            if updates["count"] == 2:
                raise cycle.CycleLimit("intentional CPU interruption")
    monkeypatch.setattr(cycle.Guard, "check", stop_after_two)
    partial = arguments(tmp_path, output=tmp_path / "partial", prediction_limit=0)
    stopped = run(partial)
    assert stopped["status"] == "stopped" and stopped["completed_steps"] == 2
    monkeypatch.setattr(cycle.Guard, "check", initial_guard)
    resumed = arguments(tmp_path, output=tmp_path / "resumed", prediction_limit=0,
        resume_checkpoint=partial.output / "checkpoint.latest.pt")
    actual_report = run(resumed)
    actual = torch.load(resumed.output / "checkpoint.latest.pt", weights_only=True)
    assert actual_report["started_from_step"] == 2 and not actual_report["warm_start"]["optimizer_reset"]
    assert actual_report["per_crop_update_counts"] == full["per_crop_update_counts"]
    assert state_sha256(actual["head_state"]) == state_sha256(expected["head_state"])
    for index in expected["optimizer_state"]["state"]:
        for key, value in expected["optimizer_state"]["state"][index].items():
            torch.testing.assert_close(actual["optimizer_state"]["state"][index][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("change", ("target", "expected_size", "learning_rate", "seed", "selected_files_sha256"))
def test_resume_rejects_identity_changes(change):
    identity = {"target": "height", "expected_size": 1024, "learning_rate": 0.001, "seed": 7,
        "selected_files_sha256": {"file": "a"}, "training_sample_ids": ["one", "two"]}
    config = {"target": "height"}
    checkpoint = {"schema": cycle.SCHEMA, "identity": identity, "head_config": config,
        "schedule": [0, 1], "step": 1, "per_crop_update_counts": [1, 0], "optimizer_state": {}}
    altered = copy.deepcopy(identity)
    altered[change] = "changed"
    with pytest.raises(ValueError, match="Resume identity"):
        cycle.validate_resume(checkpoint, altered, [0, 1], config)


def test_selected_final_not_replaced_by_worse_or_better_metric(tmp_path, monkeypatch):
    args = arguments(tmp_path, prediction_limit=0)
    original = cycle.map_metrics
    def initial_always_best(prediction, target, mask, role):
        result = original(prediction, target, mask, role)
        result["objective"] = 0 if torch.equal(prediction, torch.full_like(prediction, 0.5)) else 100
        return result
    monkeypatch.setattr(cycle, "map_metrics", initial_always_best)
    report = run(args)
    assert report["best_validation_step"] == 0 and report["selected_step"] == 4
    second = arguments(tmp_path, output=tmp_path / "best", prediction_limit=0, selection="best-validation")
    chosen = run(second)
    assert chosen["best_validation_step"] == chosen["selected_step"] == 0
    checkpoint = torch.load(second.output / "checkpoint.selected.pt", weights_only=True)
    assert checkpoint["step"] == 0 and checkpoint["per_crop_update_counts"] == [0, 0]


def test_raw_roughness_and_direct_normals_are_independent_targets(tmp_path):
    args = arguments(tmp_path)
    sample = find_samples(args.dataset, True)[0]
    _, height, _ = cycle.load_target_pair(sample, torch.device("cpu"), "height")
    _, rough, _ = cycle.load_target_pair(sample, torch.device("cpu"), "roughness")
    _, normal, _ = cycle.load_target_pair(sample, torch.device("cpu"), "normal")
    assert not torch.equal(height, rough) and normal.shape == (1, 3, 32, 32)
    codes, _ = read_png(sample["metadata_path"].parent / "roughness.png")
    np.testing.assert_array_equal(rough.numpy()[0, 0], codes[..., 0].astype(np.float32) / np.float32(65535))
    perfect, _ = cycle.map_objective(rough, rough, None, "roughness")
    shifted, _ = cycle.map_objective(rough + 0.1, rough, None, "roughness")
    assert float(perfect) == 0 and float(shifted) > 0.009
    perfect, _ = cycle.map_objective(normal, normal, None, "normal")
    inverted = normal.clone()
    inverted[:, 2] = 1 - inverted[:, 2]
    wrong, _ = cycle.map_objective(inverted, normal, None, "normal")
    assert float(perfect) < 1e-6 and float(wrong) > 1


@pytest.mark.parametrize("target", ("roughness", "normal"))
def test_independent_map_heads_actual_cpu_learning_exports(tmp_path, target):
    args = arguments(tmp_path, target=target, write_npy=True)
    report = run(args)
    assert report["head_config"]["target"] == target and report["complete_schedule"]
    for identity in report["exported_sample_ids"]:
        directory = args.output / "predictions" / identity
        value = np.load(directory / f"selected.{target}.float32.npy")
        assert value.shape == ((32, 32, 3) if target == "normal" else (32, 32))
        metadata = json.loads((directory / "cycle-export.json").read_text())
        assert metadata["target_source_bits"] == 16 and metadata["blender_color_space"] == "Non-Color"
        if target == "normal":
            np.testing.assert_allclose(np.linalg.norm(value * 2 - 1, axis=-1), 1, atol=2e-6)
            assert "Directly supervised" in metadata["normal_origin"]


def test_checkpoint_strict_pins_shapes_and_weight_only_warm_start(tmp_path):
    args = arguments(tmp_path, prediction_limit=0)
    run(args)
    path = args.output / "checkpoint.final.pt"
    model, info = cycle.load_cycle_head(path, torch.device("cpu"), digest(path))
    assert info["optimizer_reset"] and model.target == "height"
    normal, transfer = cycle.load_cycle_head(path, torch.device("cpu"), transfer_target="normal")
    assert transfer["output_layer_reset_for_new_target"] and normal.head.out_channels == 3
    for key in model.state_dict():
        if not key.startswith("head."):
            torch.testing.assert_close(normal.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="checksum"):
        cycle.load_cycle_head(path, torch.device("cpu"), "0" * 64)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["encoder"]["checkpoint_sha256"] = "0" * 64
    corrupt = tmp_path / "corrupt.pt"
    torch.save(checkpoint, corrupt)
    with pytest.raises(ValueError, match="pinned"):
        cycle.load_cycle_head(corrupt, torch.device("cpu"))


def test_rejects_native_resize_and_changed_map_before_model_load(tmp_path):
    args = arguments(tmp_path, expected_size=64)
    with pytest.raises(ValueError, match="never resized"):
        run(args)
    assert not args.output.exists()
    args.expected_size = 32
    sample = find_samples(args.dataset, True)[0]
    sample["input_path"].write_bytes(b"mutated")
    with pytest.raises((ValueError, OSError)):
        run(args)
    assert not args.output.exists()


def test_post_update_memory_stop_keeps_real_optimizer_and_report(tmp_path, monkeypatch):
    args = arguments(tmp_path, prediction_limit=0)
    original = cycle.Guard.check
    def stop(guard, stage):
        original(guard, stage)
        if stage == "after optimizer":
            raise cycle.CycleLimit("memory soft guard after optimizer")
    monkeypatch.setattr(cycle.Guard, "check", stop)
    report = run(args)
    checkpoint = torch.load(args.output / "checkpoint.latest.pt", weights_only=True)
    assert report["status"] == "stopped" and report["completed_steps"] == 1
    assert checkpoint["step"] == 1 and checkpoint["optimizer_state"]["state"]
    assert sum(checkpoint["per_crop_update_counts"]) == 1


def test_export_time_guard_preserves_completed_metrics_and_checkpoints(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    original = cycle.Guard.check
    def stop(guard, stage):
        original(guard, stage)
        if stage == "export after forward":
            raise cycle.CycleLimit("time soft guard during export")
    monkeypatch.setattr(cycle.Guard, "check", stop)
    report = run(args)
    assert report["complete_schedule"] and report["selected_step"] == 4
    assert "during export" in report["export_stop_reason"] and not report["exported_sample_ids"]
    assert (args.output / "checkpoint.selected.pt").is_file()


def test_malformed_resume_without_optimizer_or_invalid_prefix_rejected():
    identity = {"training_sample_ids": ["one", "two"]}
    checkpoint = {"schema": cycle.SCHEMA, "identity": identity, "head_config": {},
        "schedule": [0, 1], "step": 1, "per_crop_update_counts": [1, 0], "optimizer_state": None}
    with pytest.raises(ValueError, match="optimizer state"):
        cycle.validate_resume(checkpoint, identity, [0, 1], {})
    checkpoint["optimizer_state"] = {}
    with pytest.raises(ValueError, match="original balanced"):
        cycle.validate_resume(checkpoint, identity, [1, 0], {})


def test_true_probe_updates_both_native_sizes_without_source_changes(tmp_path):
    args = arguments(tmp_path, prediction_limit=0)
    run(args)
    second_root = tmp_path / "dataset64"
    dataset_with_two_corners(second_root)
    # Independent synthetic native64 fixture; create actual64 PNGs from code
    # formulas rather than resizing32 targets. This is a CPU contract test.
    for path in second_root.glob("samples/*/sample.json"):
        metadata = json.loads(path.read_text())
        row, column = np.indices((64, 64), dtype=np.uint16)
        scalar = (row * 301 + column * 173 + 11000).astype(np.uint16)
        for role in ("input", "height"):
            name = metadata["maps"][role]
            values = np.repeat(scalar[..., None], 3, axis=-1) if role == "input" else scalar[..., None]
            (path.parent / name).unlink()
            write_png(path.parent / name, values)
            metadata["map_metadata"][role]["sample_sha256"] = digest(path.parent / name)
        metadata["sample_pixel_dimensions"] = [64, 64]
        metadata["crop_rectangle_top_left_xywh"] = [value * 2 for value in metadata["crop_rectangle_top_left_xywh"]]
        for role in ("input", "height"):
            metadata["map_metadata"][role]["source"].update(width=128, height=128)
        path.write_text(json.dumps(metadata))
    original = {str(path): digest(path) for root in (args.dataset, second_root) for path in root.rglob("*") if path.is_file()}
    probe = SimpleNamespace(device="cpu", steps=2, max_minutes=2, max_driver_bytes=cycle.MAX_DRIVER_BYTES,
        encoder_size=28, learning_rate=0.001, model_directory=tmp_path, code_directory=tmp_path,
        dataset_1024=args.dataset, dataset_2048=second_root, material="white_stucco_02",
        warm_start=args.output / "checkpoint.final.pt", warm_start_sha256=None,
        allow_unreviewed=True, mask_transparent_input=False, output=tmp_path / "probe")
    report = cycle.run_probe(probe, encoder_loader=tiny_encoder_loader,
        feature_extractor=tiny_features, native_sizes=(32, 64))
    assert set(report["sizes"]) == {"32", "64"}
    for size, result in report["sizes"].items():
        assert result["status"] == "complete" and len(result["steps"]) == 3
        assert result["steps"][0]["warmup"] and not result["steps"][1]["warmup"]
        assert result["mean_measured_seconds"] > 0
        for step in result["steps"]:
            assert step["finite_gradient_norm"] > 0 and step["head_parameters_changed"]
            assert step["native_output_dimensions"] == [int(size), int(size)]
    assert report["source_files_verified_unchanged"]
    assert original == {str(path): digest(path) for root in (args.dataset, second_root) for path in root.rglob("*") if path.is_file()}
    assert not list(probe.output.rglob("*.pt"))


def test_roughness_eight_bit_source_is_honestly_retained(tmp_path):
    args = arguments(tmp_path)
    sample = find_samples(args.dataset, True)[0]
    codes = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32, 1).astype(np.uint8)
    path = sample["metadata_path"].parent / "roughness.png"
    path.unlink()
    write_png(path, codes)
    sample["metadata"]["map_metadata"]["roughness"]["sample_sha256"] = digest(path)
    _, target, _ = cycle.load_target_pair(sample, torch.device("cpu"), "roughness")
    assert sample["target_sample_bits"] == 8
    np.testing.assert_array_equal(target.numpy()[0, 0], codes[..., 0].astype(np.float32) / np.float32(255))


def test_actual_export_oom_retains_completed_training_summary(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    def allocation_failure(*_args):
        raise RuntimeError("out of memory during export")
    monkeypatch.setattr(cycle, "export_map", allocation_failure)
    report = run(args)
    assert report["complete_schedule"] and report["status"] == "complete"
    assert "actual export allocation failure" in report["export_stop_reason"]
    assert (args.output / "summary.json").is_file()
    assert (args.output / "checkpoint.latest.pt").is_file()


def test_partial_feature_cache_oom_skips_uncached_validation_and_preserves_summary(tmp_path):
    args = arguments(tmp_path, export_split="validation")
    count = {"calls": 0}
    def allocation_failure(encoder, source, size):
        count["calls"] += 1
        if count["calls"] == 3:
            raise RuntimeError("out of memory during feature cache")
        return tiny_features(encoder, source, size)
    report = cycle.run_train(args, encoder_loader=tiny_encoder_loader,
        feature_extractor=allocation_failure)
    assert report["status"] == "stopped" and report["completed_steps"] == 0
    assert not report["exported_sample_ids"] and (args.output / "summary.json").is_file()
    checkpoint = torch.load(args.output / "checkpoint.latest.pt", weights_only=True)
    assert checkpoint["step"] == 0 and checkpoint["encoder"]["checkpoint_sha256"] == MODEL_SHA256


def test_probe_setup_memory_stop_produces_summary_without_training(tmp_path, monkeypatch):
    args = arguments(tmp_path, prediction_limit=0)
    run(args)
    probe = SimpleNamespace(device="cpu", steps=1, max_minutes=2, max_driver_bytes=cycle.MAX_DRIVER_BYTES,
        encoder_size=28, learning_rate=0.001, model_directory=tmp_path, code_directory=tmp_path,
        dataset_1024=args.dataset, dataset_2048=args.dataset, material="white_stucco_02",
        warm_start=args.output / "checkpoint.final.pt", warm_start_sha256=None,
        allow_unreviewed=True, mask_transparent_input=False, output=tmp_path / "probe")
    monkeypatch.setattr(cycle, "memory", lambda _device: {"mps_driver_bytes": cycle.MAX_DRIVER_BYTES + 1})
    report = cycle.run_probe(probe, encoder_loader=tiny_encoder_loader,
        feature_extractor=tiny_features, native_sizes=(32, 32))
    assert report["status"] == "stopped during setup" and not report["sizes"]
    assert report["source_files_verified_unchanged"] and (probe.output / "summary.json").is_file()


@pytest.mark.parametrize("channels", (2, 4))
def test_independent_roughness_alpha_precision_boundary(tmp_path, channels):
    args = arguments(tmp_path)
    sample = find_samples(args.dataset, True)[0]
    gray = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32, 1) + 33000
    scalar = gray if channels == 2 else np.repeat(gray, 3, axis=-1)
    codes = np.concatenate((scalar, np.full_like(gray, 65527)), axis=-1)
    path = sample["metadata_path"].parent / "roughness.png"
    details = sample["metadata"]["map_metadata"]["roughness"]
    def publish(values):
        path.unlink()
        write_png(path, values)
        details["sample_sha256"] = digest(path)
    publish(codes)
    if channels == 2:
        with pytest.raises(ValueError, match="policy"):
            cycle.load_target_pair(sample, torch.device("cpu"), "roughness")
    details["scalar_alpha_policy"] = {"scalar_component": "grayscale", "alpha_preserved_in_png": True,
        "alpha_used_to_scale_scalar": False, "uint16_near_opaque_tolerance_codes": 8}
    _, target, _ = cycle.load_target_pair(sample, torch.device("cpu"), "roughness")
    np.testing.assert_array_equal(target.numpy()[0, 0], gray[..., 0].astype(np.float32) / np.float32(65535))
    codes[0, 0, -1] = 65526
    publish(codes)
    with pytest.raises(ValueError, match="alpha"):
        cycle.load_target_pair(sample, torch.device("cpu"), "roughness")


@pytest.mark.parametrize("dtype,accepted,rejected", ((np.uint16, 65527, 65526), (np.uint8, 255, 254)))
def test_direct_normal_auxiliary_alpha_boundary_preserves_rgb(tmp_path, dtype, accepted, rejected):
    args = arguments(tmp_path)
    sample = find_samples(args.dataset, True)[0]
    maximum = np.iinfo(dtype).max
    row, column = np.indices((32, 32))
    color = np.stack((row * 3 + maximum // 3, column * 2 + maximum // 3,
        np.full_like(row, maximum - 9)), axis=-1).astype(dtype)
    codes = np.concatenate((color, np.full((32, 32, 1), accepted, dtype=dtype)), axis=-1)
    path = sample["metadata_path"].parent / "normal.png"
    details = sample["metadata"]["map_metadata"]["normal"]
    def publish(values):
        path.unlink()
        write_png(path, values)
        details["sample_sha256"] = digest(path)
    publish(codes)
    original = path.read_bytes()
    _, target, _ = cycle.load_target_pair(sample, torch.device("cpu"), "normal")
    np.testing.assert_array_equal(target.numpy()[0].transpose(1, 2, 0),
        color.astype(np.float32) / np.float32(maximum))
    assert path.read_bytes() == original
    policy = sample["target_auxiliary_alpha_policy"]
    assert policy["accepted_alpha_deficit_codes"] == (8 if dtype == np.uint16 else 0)
    assert policy["minimum_observed_alpha_code"] == accepted
    assert not policy["alpha_used_to_scale_numeric_components"]
    cycle.export_map(tmp_path / "export", target, torch.zeros(1, 3, 32, 32), target,
        "normal", sample, 0)
    exported = json.loads((tmp_path / "export/cycle-export.json").read_text())
    assert exported["target_auxiliary_alpha_policy"] == policy
    codes[0, 0, -1] = rejected
    publish(codes)
    with pytest.raises(ValueError, match="alpha"):
        cycle.load_target_pair(sample, torch.device("cpu"), "normal")


def test_valid_short_normal_vectors_are_not_rejected_or_modified():
    target = torch.zeros(1, 3, 4, 4) + 0.5
    target[:, 2] = 0.6  # A valid +Z direction shortened by image interpolation.
    original = target.clone()
    prediction = target.clone().requires_grad_()
    loss, components = cycle.map_objective(prediction, target, None, "normal")
    assert float(loss.detach()) < 1e-6
    assert components["target_vector_minimum_length"] == pytest.approx(0.2, abs=1e-6)
    assert components["target_vector_fraction_shorter_than_0_5"] == 1
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(target, original, atol=0, rtol=0)
    with pytest.raises(ValueError, match="near-zero"):
        cycle.map_objective(prediction, torch.full_like(target, 0.5), None, "normal")
