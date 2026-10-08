#!/usr/bin/env python3
"""Bounded four-material capacity test: mixed updates versus gradual replay.

Every complete schedule gives each native training crop the same update count.
Validation uses disjoint regions of those known materials. This diagnostic does
not prepare, rewrite or relabel source data, and does not select an app model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from diagnose_material_fit import fit_metrics
from frozen_dino_height import state_sha256
from material_height_model import ARCHITECTURE, MaterialHeightNet, squared_objective
from train_material_height import choose_device, digest, find_samples, load_pair, memory, save_prediction, write_json

SCHEMA = "texture-studio-four-material-fit-diagnostic-v1"
MATERIALS = ("white_stucco_02", "farm_soil", "cotton_jersey", "broken_brick_wall")
CHECKPOINT_BUDGET = 20 * 1024 * 1024
ARTIFACT_BUDGET = 300 * 1024 * 1024


def round_schedule(counts: list[int], rng: random.Random) -> list[int]:
    """Shuffle balanced rounds without changing any requested crop count."""
    remaining, schedule = list(counts), []
    while any(remaining):
        active = [index for index, count in enumerate(remaining) if count]
        rng.shuffle(active)
        schedule.extend(active)
        for index in active:
            remaining[index] -= 1
    return schedule


def schedules(updates_per_crop: int, seed: int) -> tuple[dict[str, list[int]], list[int], list[list[int]]]:
    if updates_per_crop < 4 or updates_per_crop % 4:
        raise ValueError("updates_per_crop must be a multiple of four >= 4")
    quarter = updates_per_crop // 4
    phase_counts = [[quarter, 0, 0, 0], [quarter, quarter, 0, 0], [quarter, quarter, quarter, 0], [quarter, quarter * 2, quarter * 3, quarter * 4]]
    rng = random.Random(seed)
    curriculum, boundaries = [], []
    for counts in phase_counts:
        curriculum.extend(round_schedule(counts, rng))
        boundaries.append(len(curriculum))
    mixed = round_schedule([updates_per_crop] * 4, random.Random(seed))
    for schedule in (mixed, curriculum):
        if Counter(schedule) != Counter({index: updates_per_crop for index in range(4)}):
            raise ValueError("Schedule does not have exactly matched final per-crop updates")
    return {"mixed": mixed, "curriculum": curriculum}, boundaries, phase_counts


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint_atomic(path: Path, checkpoint: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = records[0]["metrics"]
    result = {}
    for key in keys:
        values = [record["metrics"][key] for record in records]
        if all(type(value) in (int, float) for value in values):
            result[key] = sum(values) / len(values)
        elif any(value is None for value in values):
            result[key] = None
    return result


@torch.no_grad()
def evaluate_samples(model: MaterialHeightNet, samples: list[dict[str, Any]], device: torch.device, mask_alpha: bool, cached: list[tuple] | None = None) -> dict[str, Any]:
    previous_mode = model.training
    model.eval()
    records = []
    try:
        for index, sample in enumerate(samples):
            source, target, mask = cached[index] if cached is not None else load_pair(sample, device, mask_input_alpha=mask_alpha, return_mask=True)
            prediction = model(source)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite native height: {sample['metadata_path']}")
            metrics = fit_metrics(prediction, target, mask)
            metrics["relative_squared_sum"] = sum(metrics[key] for key in ("relative_centered_height_mse", "relative_multiscale_gradient_mse", "relative_detail_highpass_mse"))
            records.append({"material_id": sample["metadata"]["material_id"], "sample_id": sample["metadata"]["sample_id"], "loss_valid_pixel_fraction": sample["loss_valid_pixel_fraction"], "metrics": metrics})
            del prediction, source, target, mask
    finally:
        model.train(previous_mode)
    return {"samples": records, "mean_metrics": mean_metrics(records), "sample_count": len(records)}


def update_peak(peak: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if isinstance(value, int):
            peak[key] = max(int(peak.get(key, 0)), value)


def verify_source_files(expected: dict[str, str]) -> None:
    for path, expected_digest in expected.items():
        if digest(Path(path)) != expected_digest:
            raise ValueError(f"Selected source sample changed during diagnostic: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.materials) != 4 or len(set(args.materials)) != 4:
        raise ValueError("Select exactly four distinct materials")
    if any(not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", material) for material in args.materials):
        raise ValueError("Material identities must be normalized safe source slugs")
    if not 16 <= args.expected_size <= 1024:
        raise ValueError("Bounded pilot requires native crops up to 1024, without resizing")
    if args.learning_rate <= 0 or args.max_minutes < 0 or args.checkpoint_every < 1 or not 0 <= args.wider_if_detail_below <= 1:
        raise ValueError("Invalid optimization/time/checkpoint/quality limit")
    schedule_variants, boundaries, phase_counts = schedules(args.updates_per_crop, args.seed)
    device = choose_device(args.device)
    index_path = args.dataset / "dataset.json"
    index_bytes = index_path.read_bytes()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    found = find_samples(args.dataset, args.allow_unreviewed)
    if digest(index_path) != index_hash:
        raise ValueError("Dataset index changed while selecting diagnostic samples")
    training, validation = [], []
    for material in args.materials:
        for split, suffix, selected in (("train", "001", training), ("validation", "003", validation)):
            matching = [sample for sample in found if sample["metadata"]["material_id"] == material and sample["metadata"]["split"] == split and sample["metadata"]["sample_id"] == material + "_auto_" + suffix]
            if len(matching) != 1:
                raise ValueError(f"Need exact prepared {material}_auto_{suffix} in {split}")
            selected.append(matching[0])
    source_files, manifest_bytes = {}, {}
    for sample in training + validation:
        payload = sample["metadata_path"].read_bytes()
        if json.loads(payload) != sample["metadata"]:
            raise ValueError(f"Sample manifest changed during selection: {sample['metadata_path']}")
        manifest_bytes[sample["metadata"]["sample_id"]] = payload
        source_files[str(sample["metadata_path"].resolve())] = hashlib.sha256(payload).hexdigest()
        for role, path_key in (("input", "input_path"), ("height", "height_path")):
            source_files[str(sample[path_key].resolve())] = sample["metadata"]["map_metadata"][role]["sample_sha256"]
        source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_transparent_input, return_mask=True)
        if tuple(source.shape[-2:]) != (args.expected_size, args.expected_size):
            raise ValueError(f"Expected exact native {args.expected_size}² source: {sample['metadata_path']}")
        del source, target, mask
    verify_source_files(source_files)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "dataset-snapshot"
    snapshot.mkdir()
    (snapshot / "dataset.json").write_bytes(index_bytes)
    for identity, payload in manifest_bytes.items():
        (snapshot / (identity + ".json")).write_bytes(payload)
    implementation = args.output / "implementation"
    implementation.mkdir()
    implementation_hashes = {}
    for name in ("diagnose_material_curriculum.py", "diagnose_material_fit.py", "material_height_model.py", "frozen_dino_height.py", "train_material_height.py", "material_dataset.py"):
        path = Path(__file__).parent / name
        (implementation / name).write_bytes(path.read_bytes())
        implementation_hashes[name] = digest(path)
    report: dict[str, Any] = {
        "schema": SCHEMA, "architecture": ARCHITECTURE,
        "diagnostic_scope": "Capacity fitting of four repeated native training crops; each separate validation crop is an unseen region of a known material",
        "validation_scope": "unseen regions of known materials", "not_fresh_material_generalization": True,
        "materials_in_introduction_order": list(args.materials), "native_dimensions": [args.expected_size, args.expected_size],
        "source_images_modified": False, "target_rescaled": False, "no_augmentation": True,
        "target_encoding": "Raw uint16 codes / 65535 into active Float32; no gamma or per-patch range normalization",
        "objective": "relative_squared", "learning_rate": args.learning_rate, "weight_decay": 1e-4, "seed": args.seed, "batch_size": 1,
        "requested_updates_per_crop": args.updates_per_crop, "requested_steps_per_variant": args.updates_per_crop * 4,
        "curriculum_phase_counts": phase_counts, "phase_end_steps": boundaries,
        "schedules": schedule_variants, "schedule_sha256": {key: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest() for key, value in schedule_variants.items()},
        "matched_final_crop_update_counts_required": True,
        "dataset_index_sha256_at_selection": index_hash, "selected_source_files_sha256": source_files,
        "selected_crop_rectangles_top_left_xywh": {sample["metadata"]["sample_id"]: sample["metadata"]["crop_rectangle_top_left_xywh"] for sample in training + validation},
        "alpha_masking_enabled": args.mask_transparent_input,
        "loss_valid_pixel_fractions": {sample["metadata"]["sample_id"]: sample["loss_valid_pixel_fraction"] for sample in training + validation},
        "implementation_sha256": implementation_hashes, "device": str(device), "torch_version": str(torch.__version__),
        "checkpoint_every_steps": args.checkpoint_every, "checkpoint_policy": "Atomic model + AdamW latest every 100 steps by default, and each phase; one final checkpoint per completed variant; automatic resume CLI not implemented",
        "max_minutes_per_variant": args.max_minutes, "wider_if_mean_training_detail_below": args.wider_if_detail_below,
        "wider_condition_uses_training_capacity_only": True,
        "initialization_policy": "Identical initial weights for paired 12-channel variants; wider 24-channel model uses the same seed but an independent architecture initialization",
        "export_policy": "Final first-material training and disjoint-region predictions only per variant; metrics include all four crops; up to six predictions total",
        "artifact_budget_bytes": ARTIFACT_BUDGET, "checkpoint_budget_bytes": CHECKPOINT_BUDGET,
        "production_promotion": False, "variants": {},
    }
    write_json(args.output / "run.json", report)
    # Cache only four active training Float32 pairs; validation loads one at a time.
    cached = [load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True) for sample in training]
    report["cached_training_tensor_bytes"] = sum(value.numel() * value.element_size() for pair in cached for value in pair if value is not None)
    cancelled = False
    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
    old_handlers = {name: signal.signal(name, stop) for name in (signal.SIGINT, signal.SIGTERM)}
    initial_12_hash = None
    try:
        for variant, width, schedule_name in (("mixed12", 12, "mixed"), ("curriculum12", 12, "curriculum"), ("mixed24", 24, "mixed")):
            if variant == "mixed24":
                previous = report["variants"].get("mixed12", {})
                detail = previous.get("final_training_fit", {}).get("mean_metrics", {}).get("detail_highpass_correlation_radius_4")
                trigger = previous.get("complete", False) and (detail is None or detail < args.wider_if_detail_below)
                report["wider_condition"] = {"evaluated_mixed12_mean_training_detail": detail, "threshold": args.wider_if_detail_below, "triggered": trigger and not args.skip_wider, "explicitly_skipped": args.skip_wider}
                if not trigger or args.skip_wider:
                    write_json(args.output / "summary.json", report)
                    break
            if cancelled:
                break
            torch.manual_seed(args.seed)
            model = MaterialHeightNet(width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            initial_hash = state_sha256(model.state_dict())
            if width == 12:
                if initial_12_hash is None:
                    initial_12_hash = initial_hash
                elif initial_hash != initial_12_hash:
                    raise ValueError("The 12-channel paired variants did not receive identical initial weights")
            for source, _, _ in cached:
                with torch.no_grad():
                    prediction = model(source)
                    if not torch.equal(prediction, torch.full_like(prediction, 0.5)):
                        raise ValueError("Initial native height must be exactly 0.5 for every crop")
                    del prediction
            directory = args.output / variant
            directory.mkdir()
            schedule = schedule_variants[schedule_name]
            initial = evaluate_samples(model, training, device, args.mask_transparent_input, cached)
            initial_validation = evaluate_samples(model, validation, device, args.mask_transparent_input)
            started, completed_steps, per_crop = time.monotonic(), 0, [0] * 4
            peak, history, proofs = memory(device), [], {}
            time_limited = False
            def checkpoint_payload(complete: bool, metrics: dict | None = None) -> dict[str, Any]:
                return {"schema": SCHEMA, "architecture": ARCHITECTURE, "variant": variant, "model_config": {"base_channels": width}, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "optimizer_initialized_fresh": True, "step": completed_steps, "per_crop_update_counts": list(per_crop), "schedule": schedule, "schedule_sha256": report["schedule_sha256"][schedule_name], "initial_state_sha256": initial_hash, "dataset_index_sha256_at_selection": index_hash, "selected_source_files_sha256": source_files, "target_encoding": report["target_encoding"], "objective": report["objective"], "seed": args.seed, "complete": complete, "diagnostic_only_not_production": True, "metrics": metrics, "torch_rng_state": torch.get_rng_state()}
            save_checkpoint_atomic(directory / "checkpoint.latest.pt", checkpoint_payload(False))
            print(json.dumps({"event": "initial", "variant": variant, "initial_head_state_sha256": initial_hash, "training_fit": initial, "known_material_disjoint_regions": initial_validation}), flush=True)
            model.train()
            for position, crop_index in enumerate(schedule):
                if cancelled:
                    break
                if args.max_minutes and time.monotonic() - started >= args.max_minutes * 60:
                    time_limited = True
                    break
                source, target, mask = cached[crop_index]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(source)
                loss, components = squared_objective(prediction, target, mask)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite relative-squared loss: {variant}, step {position+1}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if position < 2:
                    upstream = model.encoder[0].layers[0].weight
                    before_head, before_upstream = model.head.weight.detach().clone(), upstream.detach().clone()
                    proofs[f"step_{position+1}"] = {"gradient_norm_before_clipping": float(norm), "head_gradient_norm_after_clipping": float(model.head.weight.grad.norm()), "native_rgb_encoder_gradient_nonzero": bool(torch.count_nonzero(upstream.grad)), "native_rgb_encoder_gradient_norm_after_clipping": float(upstream.grad.norm()), "native_dimensions": list(prediction.shape[-2:])}
                optimizer.step()
                completed_steps += 1
                per_crop[crop_index] += 1
                if position < 2:
                    proofs[f"step_{position+1}"].update({"head_parameters_changed": bool(torch.any(before_head != model.head.weight)), "native_rgb_encoder_parameters_changed": bool(torch.any(before_upstream != upstream))})
                    if position == 0 and not proofs["step_1"]["head_parameters_changed"]:
                        raise ValueError("Diagnostic failed to update output-head weights")
                    if position == 1 and not proofs["step_2"]["native_rgb_encoder_gradient_nonzero"]:
                        raise ValueError("Diagnostic failed to obtain an actual native-encoder learning gradient")
                update_peak(peak, memory(device))
                if completed_steps % args.checkpoint_every == 0:
                    save_checkpoint_atomic(directory / "checkpoint.latest.pt", checkpoint_payload(False))
                    print(json.dumps({"event": "training", "variant": variant, "step": completed_steps, "per_crop_updates": per_crop, "elapsed_seconds": time.monotonic() - started, "loss": float(loss.detach()), "loss_components": components, "memory": memory(device)}), flush=True)
                if completed_steps in boundaries:
                    train_result = evaluate_samples(model, training, device, args.mask_transparent_input, cached)
                    val_result = evaluate_samples(model, validation, device, args.mask_transparent_input)
                    record = {"event": "phase_complete", "variant": variant, "step": completed_steps, "per_crop_update_counts": list(per_crop), "training_fit": train_result, "known_material_disjoint_regions": val_result, "elapsed_seconds": time.monotonic() - started}
                    history.append(record)
                    save_checkpoint_atomic(directory / "checkpoint.latest.pt", checkpoint_payload(False, record))
                    write_json(directory / "progress.json", {"variant": variant, "initial_training_fit": initial, "initial_known_material_disjoint_regions": initial_validation, "phase_history": history})
                    print(json.dumps(record), flush=True)
                del prediction, loss
            complete = completed_steps == len(schedule)
            if complete and per_crop != [args.updates_per_crop] * 4:
                raise ValueError("Complete diagnostic does not have matched per-crop updates")
            final_training = evaluate_samples(model, training, device, args.mask_transparent_input, cached)
            final_validation = evaluate_samples(model, validation, device, args.mask_transparent_input)
            verify_source_files(source_files)
            final_metrics = {"training_fit": final_training, "known_material_disjoint_regions": final_validation}
            save_checkpoint_atomic(directory / "checkpoint.latest.pt", checkpoint_payload(complete, final_metrics))
            os.replace(directory / "checkpoint.latest.pt", directory / "checkpoint.final.pt")
            model.eval()
            # Only the first material is exported; all four have complete metrics.
            for label, sample, pair in (("final-training-fit", training[0], cached[0]), ("final-known-material-disjoint-region", validation[0], None)):
                source, target, mask = pair if pair is not None else load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
                with torch.no_grad():
                    prediction = model(source)
                target_directory = directory / label
                save_prediction(target_directory, prediction, source, target, label)
                metadata_path = target_directory / (label + ".prediction.json")
                metadata = json.loads(metadata_path.read_text())
                metadata.update({"schema": SCHEMA, "variant": variant, "sample_id": sample["metadata"]["sample_id"], "diagnostic_only_not_production": True, "complete_schedule": complete})
                write_json(metadata_path, metadata)
                if mask is not None:
                    np.save(target_directory / "loss-valid.bool.npy", mask.detach().cpu().numpy()[0, 0].astype(bool), allow_pickle=False)
                    write_json(target_directory / "loss-valid.json", {"meaning": "Input-alpha loss/metric eligibility with 8px margin, not model-estimated confidence", "excluded_prediction_pixels_unassessed": True})
                del prediction, source, target, mask
            result = {"base_channels": width, "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()), "initial_head_state_sha256": initial_hash, "initial_prediction_identically_0_5": True, "initial_training_fit": initial, "initial_known_material_disjoint_regions": initial_validation, "final_training_fit": final_training, "final_known_material_disjoint_regions": final_validation, "completed_steps": completed_steps, "per_crop_update_counts": per_crop, "complete": complete, "cancelled": cancelled, "time_limited": time_limited, "elapsed_seconds": time.monotonic() - started, "gradient_proof": proofs, "phase_history": history, "sampled_peak_memory": peak, "checkpoint_sha256": digest(directory / "checkpoint.final.pt")}
            report["variants"][variant] = {key: value for key, value in result.items() if key != "phase_history"}
            write_json(directory / "summary.json", result)
            report["artifact_bytes"] = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
            report["checkpoint_bytes"] = sum(path.stat().st_size for path in args.output.rglob("checkpoint.*.pt"))
            if report["artifact_bytes"] > ARTIFACT_BUDGET or report["checkpoint_bytes"] > CHECKPOINT_BUDGET:
                raise ValueError("Bounded diagnostic artifact/checkpoint budget exceeded")
            write_json(args.output / "summary.json", report)
            print(json.dumps({"event": "variant_complete", "variant": variant, "complete": complete, "steps": completed_steps, "per_crop_updates": per_crop, "mean_training_fit": final_training["mean_metrics"], "mean_known_material_disjoint_regions": final_validation["mean_metrics"]}), flush=True)
            del model, optimizer
            if device.type == "mps":
                torch.mps.empty_cache()
            if cancelled or time_limited:
                break
    finally:
        for name, old_handler in old_handlers.items():
            signal.signal(name, old_handler)
    report["cancelled"] = cancelled
    report["matched_complete_12_channel_schedules"] = all(report["variants"].get(name, {}).get("complete", False) for name in ("mixed12", "curriculum12"))
    report["dataset_index_sha256_after"] = digest(index_path)
    report["dataset_index_changed_during_run"] = report["dataset_index_sha256_after"] != index_hash
    report["dataset_index_drift_policy"] = "Selected 8 manifests and 16 sample maps remain frozen and SHA verified; unrelated dataset additions are reported without relabeling this diagnostic"
    verify_source_files(source_files)
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--materials", nargs=4, default=list(MATERIALS))
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--updates-per-crop", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--expected-size", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=12, help="Per variant; 0 disables, partial comparisons marked incomplete")
    parser.add_argument("--wider-if-detail-below", type=float, default=0.7)
    parser.add_argument("--skip-wider", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--mask-transparent-input", action="store_true")
    args = parser.parse_args()
    if args.updates_per_crop > 1000:
        parser.error("Bounded diagnostic supports at most 1000 updates per crop")
    report = run(args)
    print(json.dumps({"event": "complete", "summary": str(args.output / "summary.json"), "matched_complete_12_channel_schedules": report["matched_complete_12_channel_schedules"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
