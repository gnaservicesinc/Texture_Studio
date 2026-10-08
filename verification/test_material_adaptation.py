"""Matched quality-experiment contracts exercised with tiny CPU-only models."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_material_adaptation as diagnostic
from frozen_dino_height import state_sha256
from probe_material_adapters import DifferentiableHeightNet
from test_material_adapter_probe import tiny_encoder_loader
from test_material_curriculum import make_dataset
from train_material_height import digest, find_samples


def arguments(tmp_path):
    dataset = tmp_path / "dataset"
    make_dataset(dataset)
    checkpoint = tmp_path / "checked-head.fixture"
    checkpoint.write_bytes(b"Test-only injected head loader; production uses the fixed checked SHA256")
    return SimpleNamespace(dataset=dataset, output=tmp_path / "experiment", head_checkpoint=checkpoint,
        model_directory=tmp_path, code_directory=tmp_path, device="cpu", allow_unreviewed=True,
        mask_transparent_input=False, expected_size=32, encoder_size=28, updates_per_crop=2,
        seed=2307, evaluate_every=4, checkpoint_every=3, max_minutes=12,
        max_driver_bytes=diagnostic.MAX_DRIVER_BYTES)


def head_loader(args):
    selected = find_samples(args.dataset, True)
    bindings = {sample["metadata"]["sample_id"]: {
        role: sample["metadata"]["map_metadata"][role]["sample_sha256"]
        for role in ("input", "height")}
        for sample in selected if sample["metadata"]["material_id"] == diagnostic.MATERIALS[0]}
    torch.manual_seed(709)
    template = DifferentiableHeightNet(4, 8, 4)
    with torch.no_grad():
        template.head.weight.fill_(0.04)
    initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    def load(_path, device):
        head = DifferentiableHeightNet(4, 8, 4)
        head.load_state_dict(initial_state, strict=True)
        return head.to(device), {"model_config": {"base_channels": 4, "feature_channels": 8,
            "projection_channels": 4}, "sample_map_sha256": bindings, "source_step": 300,
            "test_fixture": True}
    return load, initial_state


def run_fixture(args):
    loader, state = head_loader(args)
    return diagnostic.run(args, encoder_loader=tiny_encoder_loader, head_loader=loader,
        encoder_contract=(8, 2)), state


def test_balanced_schedule_exact_counts_and_equal_rounds():
    schedule = diagnostic.balanced_schedule(300, 2307)
    assert len(schedule) == 1200 and Counter(schedule) == {index: 300 for index in range(4)}
    assert all(set(schedule[offset:offset + 4]) == {0, 1, 2, 3}
        for offset in range(0, len(schedule), 4))
    assert schedule == diagnostic.balanced_schedule(300, 2307)
    assert schedule != diagnostic.balanced_schedule(300, 2308)
    for invalid in (0, 1, 1001):
        with pytest.raises(ValueError, match="updates per crop"):
            diagnostic.balanced_schedule(invalid, 2307)


@pytest.mark.parametrize("name,value,match", (
    ("encoder_size", 532, "working size"), ("encoder_size", 513, "working size"),
    ("max_minutes", 0, "positive time"), ("max_minutes", 13, "positive time"),
    ("expected_size", 2048, "native"),
))
def test_bounds_rejected_before_encoder_or_output_creation(tmp_path, name, value, match):
    args = arguments(tmp_path)
    setattr(args, name, value)
    def forbidden(*_args):
        raise AssertionError("Bounds must be rejected before loading models")
    with pytest.raises(ValueError, match=match):
        diagnostic.run(args, encoder_loader=forbidden)
    assert not args.output.exists()


def test_cpu_pair_true_lora_gradients_no_grad_evaluation_and_raw_source_invariants(tmp_path):
    args = arguments(tmp_path)
    original_files = {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    original_head = digest(args.head_checkpoint)
    report, initial_state = run_fixture(args)
    assert report["schema"] == diagnostic.SCHEMA
    assert report["matched_complete_schedules"] and not report["production_promotion"]
    assert report["source_files_and_native_targets_unchanged"]
    assert report["original_pretrained_base_fingerprint_after"] == report["original_pretrained_base_fingerprint"]
    assert report["frozen_feature_cache_detached"] and report["frozen_feature_cache_device"] == "cpu"
    assert len(report["selected_files_sha256"]) == 25
    assert report["native_dimensions"] == [32, 32] and report["encoder_dimensions"] == [28, 28]
    assert original_files == {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    assert digest(args.head_checkpoint) == original_head
    assert report["artifact_bytes"] < diagnostic.ARTIFACT_BUDGET
    assert report["checkpoint_bytes"] < diagnostic.CHECKPOINT_BUDGET
    assert set(report["variants"]) == {"frozen", "lora"}
    hashes = {value["initial_head_state_sha256"] for value in report["variants"].values()}
    assert hashes == {state_sha256(initial_state)}
    for name, result in report["variants"].items():
        assert result["complete"] and result["completed_steps"] == 8
        assert result["per_crop_update_counts"] == [2, 2, 2, 2]
        assert [record["step"] for record in result["phase_history"]] == [0, 4, 8]
        assert result["final_training_fit"]["sample_count"] == 4
        assert result["final_known_material_disjoint_regions"]["sample_count"] == 4
        assert result["optimizer_tensor_state_bytes"] > 0
        assert len(result["exported_sample_ids"]) == 2
        final = torch.load(args.output / name / "checkpoint.final.pt", weights_only=True)
        assert final["step"] == 8 and final["complete_schedule"]
        assert "base_state" not in final and "model_state" not in final
        assert final["optimizer_state"]["state"]
        assert final["schedule_sha256"] == report["schedule_sha256"]
        assert not (args.output / name / "checkpoint.latest.pt").exists()
        assert len(list((args.output / name).rglob("*.height.float32.exr"))) == 2
        if name == "frozen":
            assert not final["adapter_state"] and result["adapter_parameters"] == 0
        else:
            assert final["adapter_state"] and result["zero_adapter_initial_features_and_predictions_identical"]
            first, second = result["gradient_proof"]["step_1"], result["gradient_proof"]["step_2"]
            for proof in first["adapters"].values():
                assert not proof["lora_A"]["gradient_nonzero"] and not proof["lora_A"]["parameters_changed"]
                assert proof["lora_B"]["gradient_nonzero"] and proof["lora_B"]["parameters_changed"]
            assert all(proof["lora_A"]["gradient_nonzero"] and proof["lora_A"]["parameters_changed"]
                for proof in second["adapters"].values())
    json.dumps(report, allow_nan=False)


def test_initial_best_retained_when_every_updated_objective_is_worse(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(diagnostic, "selection_score", lambda evaluation: 0.0 if evaluation["step"] == 0 else 1.0)
    report, initial = run_fixture(args)
    for name, variant in report["variants"].items():
        assert variant["selected_step"] == 0 and not variant["selection_improved_initial"]
        selected = torch.load(args.output / name / "checkpoint.selected.pt", weights_only=True)
        assert selected["step"] == 0 and "optimizer_state" not in selected
        assert state_sha256(selected["head_state"]) == state_sha256(initial)
        assert all(torch.count_nonzero(values["lora_B"]) == 0
            for values in selected["adapter_state"].values())


def test_post_optimizer_memory_limit_retains_one_real_update_and_durable_state(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    original_step = torch.optim.AdamW.step
    state = {"updated": False}
    def observed_step(optimizer, *values, **keywords):
        result = original_step(optimizer, *values, **keywords)
        state["updated"] = True
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", observed_step)
    monkeypatch.setattr(diagnostic, "memory", lambda _device: {
        "mps_driver_bytes": args.max_driver_bytes + 1 if state["updated"] else 0})
    report, _ = run_fixture(args)
    result = report["variants"]["frozen"]
    assert not report["matched_complete_schedules"] and set(report["variants"]) == {"frozen"}
    assert result["status"] == "stopped_memory_soft_guard" and result["completed_steps"] == 1
    assert report["stopped_stage"]["stage"] == "training_after_optimizer"
    checkpoint = torch.load(args.output / "frozen/checkpoint.final.pt", weights_only=True)
    assert checkpoint["step"] == 1 and checkpoint["optimizer_state"]["state"]
    assert not checkpoint["complete_schedule"]


def test_late_export_guard_preserves_complete_training_and_final_metrics(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    actual_save = diagnostic.save_prediction
    state = {"exported": False}
    def export(*values, **keywords):
        result = actual_save(*values, **keywords)
        state["exported"] = True
        return result
    monkeypatch.setattr(diagnostic, "save_prediction", export)
    monkeypatch.setattr(diagnostic, "memory", lambda _device: {
        "mps_driver_bytes": args.max_driver_bytes + 1 if state["exported"] else 0})
    report, _ = run_fixture(args)
    result = report["variants"]["frozen"]
    assert not report["matched_complete_schedules"] and result["status"] == "stopped_memory_soft_guard"
    assert result["training_and_final_evaluation_complete"] and not result["complete"]
    assert result["completed_steps"] == 8 and result["final_complete_evaluation"]["step"] == 8
    assert len(result["exported_sample_ids"]) == 1
    assert report["stopped_stage"]["stage"] == "export_after_write"
    checkpoint = torch.load(args.output / "frozen/checkpoint.final.pt", weights_only=True)
    assert checkpoint["step"] == 8 and checkpoint["complete_schedule"]
    assert (args.output / "summary.json").is_file()


def test_time_guard_returns_partial_report_without_starting_an_update(tmp_path):
    args = arguments(tmp_path)
    args.max_minutes = 1e-12
    report, _ = run_fixture(args)
    result = report["variants"]["frozen"]
    assert result["status"] == "stopped_time_soft_guard" and result["completed_steps"] == 0
    assert not report["matched_complete_schedules"]
    assert result["initial_evaluation"] is None and result["selected_checkpoint_sha256"] is None
    assert (args.output / "frozen/checkpoint.final.pt").is_file()


def test_selected_digest_guard_refuses_modified_source(tmp_path):
    path = tmp_path / "source.png"
    path.write_bytes(b"original")
    expected = {str(path): digest(path)}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during adaptation"):
        diagnostic.verify_selected_files(expected)
