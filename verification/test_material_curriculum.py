"""Balanced replay schedules, durable checkpoints and native source contracts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_material_curriculum as diagnostic
from material_dataset import write_png
from material_height_model import MaterialHeightNet
from train_material_height import digest, find_samples, load_pair


def test_complete_mixed_and_curriculum_have_exact_300_updates_each():
    schedules, boundaries, phases = diagnostic.schedules(300, 2307)
    assert boundaries == [75, 225, 450, 1200]
    assert phases == [[75, 0, 0, 0], [75, 75, 0, 0], [75, 75, 75, 0], [75, 150, 225, 300]]
    for schedule in schedules.values():
        assert len(schedule) == 1200
        assert Counter(schedule) == {0: 300, 1: 300, 2: 300, 3: 300}
    assert Counter(schedules["curriculum"][:75]) == {0: 75}
    assert Counter(schedules["curriculum"][:225]) == {0: 150, 1: 75}
    assert Counter(schedules["curriculum"][:450]) == {0: 225, 1: 150, 2: 75}
    assert all(set(schedules["mixed"][offset:offset+4]) == {0, 1, 2, 3} for offset in range(0, 1200, 4))
    assert diagnostic.schedules(300, 2307)[0] == schedules
    assert diagnostic.schedules(300, 2308)[0] != schedules


@pytest.mark.parametrize("updates", (0, 1, 3, 5, 299))
def test_invalid_update_counts_refused(updates):
    with pytest.raises(ValueError, match="multiple of four"):
        diagnostic.schedules(updates, 2307)


def test_atomic_checkpoint_failure_keeps_previous_complete_payload(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.latest.pt"
    diagnostic.save_checkpoint_atomic(path, {"step": 100, "tensor": torch.tensor([1.0])})
    original = path.read_bytes()
    def fail(_data, stream):
        stream.write(b"incomplete")
        raise OSError("disk error")
    monkeypatch.setattr(diagnostic.torch, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        diagnostic.save_checkpoint_atomic(path, {"step": 200})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
    assert torch.load(path, weights_only=True)["step"] == 100


def make_dataset(root: Path):
    items = []
    for material_index, material in enumerate(diagnostic.MATERIALS):
        for number, split in ((1, "train"), (3, "validation")):
            identity = material + f"_auto_{number:03d}"
            directory = root / "samples" / identity
            directory.mkdir(parents=True)
            x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
            height = np.clip(0.5 + 0.1 * np.sin(x * (12 + material_index)) + 0.08 * np.cos(y * 15 + number / 5), 0, 1)
            codes = np.rint(height * 65535).astype(np.uint16)
            rgb = np.repeat(codes[..., None], 3, axis=-1)
            write_png(directory / "diffuse.png", rgb)
            write_png(directory / "displacement.png", codes[..., None])
            metadata = {"schema_version": 2, "sample_id": identity, "material_id": material, "split": split, "status": "prepared", "source_precision_verified": True, "crop_values_verified": True, "sample_pixel_dimensions": [32, 32], "crop_rectangle_top_left_xywh": [0 if number == 1 else 32, 0, 32, 32], "split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "source_region_role": split, "maps": {"input": "diffuse.png", "height": "displacement.png"}, "map_metadata": {role: {"encoding": "linear" if role == "input" else "linear_data", "sample_sha256": digest(directory / filename), "source": {"file_sha256": hashlib.sha256((material+role).encode()).hexdigest(), "width": 64, "height": 32}} for role, filename in (("input", "diffuse.png"), ("height", "displacement.png"))}}
            (directory / "sample.json").write_text(json.dumps(metadata))
            items.append({key: metadata[key] for key in ("sample_id", "material_id", "split", "status")})
            items[-1]["path"] = "samples/" + identity
    (root / "dataset.json").write_text(json.dumps({"schema_version": 2, "split_strategy": "heldout-region-v1", "validation_scope": "unseen_regions_of_known_materials", "samples": items}))


def arguments(tmp_path, **overrides):
    dataset = tmp_path / "dataset"
    make_dataset(dataset)
    return SimpleNamespace(dataset=dataset, materials=list(diagnostic.MATERIALS), expected_size=32, updates_per_crop=4, seed=2307, device="cpu", allow_unreviewed=True, mask_transparent_input=False, output=tmp_path / "output", learning_rate=0.001, checkpoint_every=5, max_minutes=0, wider_if_detail_below=0.7, skip_wider=True, **overrides)


def test_real_cpu_paired_capacity_counts_provenance_and_gradient_proof(tmp_path):
    args = arguments(tmp_path)
    originals = {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    first = find_samples(args.dataset, True)[0]
    _, height_before, _ = load_pair(first, torch.device("cpu"), return_mask=True)
    report = diagnostic.run(args)
    _, height_after, _ = load_pair(first, torch.device("cpu"), return_mask=True)
    torch.testing.assert_close(height_before, height_after, atol=0, rtol=0)
    assert originals == {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    assert report["matched_complete_12_channel_schedules"]
    assert not report["dataset_index_changed_during_run"]
    assert set(report["variants"]) == {"mixed12", "curriculum12"}
    assert report["checkpoint_bytes"] < diagnostic.CHECKPOINT_BUDGET
    assert report["artifact_bytes"] < diagnostic.ARTIFACT_BUDGET
    assert len(list((args.output / "dataset-snapshot").glob("*.json"))) == 9
    hashes = {value["initial_head_state_sha256"] for value in report["variants"].values()}
    assert len(hashes) == 1
    for name, variant in report["variants"].items():
        assert variant["initial_prediction_identically_0_5"]
        assert variant["completed_steps"] == 16
        assert variant["per_crop_update_counts"] == [4, 4, 4, 4]
        assert variant["gradient_proof"]["step_1"]["head_parameters_changed"]
        assert not variant["gradient_proof"]["step_1"]["native_rgb_encoder_gradient_nonzero"]
        assert variant["gradient_proof"]["step_2"]["native_rgb_encoder_gradient_nonzero"]
        summary = json.loads((args.output / name / "summary.json").read_text())
        assert [record["step"] for record in summary["phase_history"]] == [1, 3, 6, 16]
        assert summary["final_training_fit"]["sample_count"] == 4
        assert summary["final_known_material_disjoint_regions"]["sample_count"] == 4
        checkpoint = torch.load(args.output / name / "checkpoint.final.pt", weights_only=True)
        assert checkpoint["schema"] == diagnostic.SCHEMA
        assert checkpoint["diagnostic_only_not_production"]
        assert checkpoint["per_crop_update_counts"] == [4, 4, 4, 4]
        assert checkpoint["optimizer_state"]["state"]
        loaded = MaterialHeightNet(**checkpoint["model_config"])
        loaded.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        assert not (args.output / name / "checkpoint.latest.pt").exists()
        assert len(list((args.output / name).rglob("*.height.float32.exr"))) == 2
    json.dumps(report, allow_nan=False)


def test_time_limit_marks_partial_counts_and_keeps_recoverable_checkpoint(tmp_path):
    args = arguments(tmp_path)
    args.max_minutes = 1e-12
    report = diagnostic.run(args)
    assert not report["matched_complete_12_channel_schedules"]
    assert set(report["variants"]) == {"mixed12"}
    result = report["variants"]["mixed12"]
    assert result["time_limited"] and not result["complete"]
    assert result["completed_steps"] == 0 and result["per_crop_update_counts"] == [0] * 4
    checkpoint = torch.load(args.output / "mixed12/checkpoint.final.pt", weights_only=True)
    assert not checkpoint["complete"]


def test_conditional_wider_head_preserves_counts_and_checkpoint_budget(tmp_path):
    args = arguments(tmp_path)
    args.skip_wider = False
    args.wider_if_detail_below = 1.0
    report = diagnostic.run(args)
    assert report["wider_condition"]["triggered"]
    assert set(report["variants"]) == {"mixed12", "curriculum12", "mixed24"}
    wider = report["variants"]["mixed24"]
    assert wider["base_channels"] == 24 and wider["trainable_parameters"] == 869281
    assert wider["complete"] and wider["per_crop_update_counts"] == [4] * 4
    assert wider["initial_head_state_sha256"] != report["variants"]["mixed12"]["initial_head_state_sha256"]
    assert report["checkpoint_bytes"] < diagnostic.CHECKPOINT_BUDGET
    assert len(list(args.output.rglob("checkpoint.final.pt"))) == 3
    assert len(list(args.output.rglob("*.height.float32.exr"))) == 6


def test_selected_source_sha_guard_rejects_mutated_file(tmp_path):
    path = tmp_path / "source.png"
    path.write_bytes(b"original")
    expected = {str(path): digest(path)}
    diagnostic.verify_source_files(expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during"):
        diagnostic.verify_source_files(expected)
