#!/usr/bin/env python3
"""Repeated-crop capacity diagnostic for native material-height learning.

The training crop is deliberately repeated without augmentation. It tests
whether this architecture/objective can fit a known pair, not generalization.
Original PNGs and the prepared dataset are read only. At most two variants and
one same-material disjoint validation crop are evaluated per invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from material_height_model import ARCHITECTURE, SCHEMA, MaterialHeightNet, detail_mask, height_loss, squared_objective, weighted_center, weighted_mean
from train_material_height import choose_device, digest, find_samples, load_pair, memory, prediction_metrics, save_prediction, write_json

VARIANTS = ("current_l1", "relative_squared")
DIAGNOSTIC_SCHEMA = "texture-studio-repeated-crop-fit-v1"




def correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor | None = None) -> float | None:
    first, second = weighted_center(first.float(), mask), weighted_center(second.float(), mask)
    denominator = (weighted_mean(first.square(), mask) * weighted_mean(second.square(), mask)).sqrt()
    if float(denominator) <= 1e-12:
        return None
    return float((weighted_mean(first * second, mask) / denominator).clamp(-1, 1))


@torch.no_grad()
def fit_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    result = prediction_metrics(prediction, target, mask)
    result["centered_height_correlation"] = correlation(prediction, target, mask)
    centered_prediction, centered_target = weighted_center(prediction, mask), weighted_center(target, mask)
    target_variance = weighted_mean(centered_target.square(), mask)
    result["centered_height_signed_gain"] = None if float(target_variance) <= 1e-12 else float(weighted_mean(centered_prediction * centered_target, mask) / target_variance)
    result["height_amplitude_std_ratio"] = result["prediction_std"] / max(result["target_std"], 1e-12)
    pg, tg, mg = [], [], []
    for axis in ("x", "y"):
        if axis == "x":
            pg.append(prediction[..., :, 1:] - prediction[..., :, :-1])
            tg.append(target[..., :, 1:] - target[..., :, :-1])
            mg.append(None if mask is None else mask[..., :, 1:] * mask[..., :, :-1])
        else:
            pg.append(prediction[..., 1:, :] - prediction[..., :-1, :])
            tg.append(target[..., 1:, :] - target[..., :-1, :])
            mg.append(None if mask is None else mask[..., 1:, :] * mask[..., :-1, :])
    correlations = [correlation(p, t, m) for p, t, m in zip(pg, tg, mg)]
    result["native_gradient_correlation"] = None if any(value is None for value in correlations) else sum(correlations) / 2
    penergy = sum(weighted_mean(p.square(), m) for p, m in zip(pg, mg)) / 2
    tenergy = sum(weighted_mean(t.square(), m) for t, m in zip(tg, mg)) / 2
    result["native_gradient_rms_amplitude_ratio"] = float((penergy / tenergy.clamp_min(1e-12)).sqrt())
    phigh = prediction - F.avg_pool2d(F.pad(prediction, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    thigh = target - F.avg_pool2d(F.pad(target, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    result["detail_highpass_correlation_radius_4"] = correlation(phigh, thigh, detail_mask(mask))
    px, py = pg[0][..., :-1, :], pg[1][..., :, :-1]
    tx, ty = tg[0][..., :-1, :], tg[1][..., :, :-1]
    cell_mask = None if mask is None else mask[..., :-1, :-1] * mask[..., :-1, 1:] * mask[..., 1:, :-1] * mask[..., 1:, 1:]
    vector_dot = weighted_mean(px * tx + py * ty, cell_mask)
    vector_denominator = (weighted_mean(px.square() + py.square(), cell_mask) * weighted_mean(tx.square() + ty.square(), cell_mask)).sqrt()
    result["gradient_vector_cosine"] = None if float(vector_denominator) <= 1e-12 else float((vector_dot / vector_denominator).clamp(-1, 1))
    result["sigmoid_saturation_fraction_below_0_01_above_0_99"] = float(weighted_mean(((prediction < 0.01) | (prediction > 0.99)).float(), mask))
    result["undefined_correlations_are_null"] = True
    _, squared_components = squared_objective(prediction, target, mask)
    result.update(squared_components)
    return result


def select_material_samples(dataset: Path, material: str, sample_id: str | None, validation_id: str | None, skip_validation: bool, allow_unreviewed: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    samples = find_samples(dataset, allow_unreviewed)
    training = [sample for sample in samples if sample["metadata"]["material_id"] == material and sample["metadata"]["split"] == "train"]
    validation = [sample for sample in samples if sample["metadata"]["material_id"] == material and sample["metadata"]["split"] == "validation"]
    if sample_id:
        training = [sample for sample in training if sample["metadata"]["sample_id"] == sample_id]
    if validation_id:
        validation = [sample for sample in validation if sample["metadata"]["sample_id"] == validation_id]
    if not training:
        raise ValueError(f"No eligible training crop for {material!r} / {sample_id!r}")
    if not skip_validation and not validation:
        raise ValueError("Same-material disjoint validation crop missing; explicit --skip-validation required to omit it")
    return training[0], None if skip_validation else validation[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = choose_device(args.device)
    training, validation = select_material_samples(args.dataset, args.material, args.sample_id, args.validation_sample_id, args.skip_validation, args.allow_unreviewed)
    # Source hash/precision/transfer/alpha preflight runs before GPU allocations.
    for sample in [training] + ([validation] if validation else []):
        source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_transparent_input, return_mask=True)
        if list(source.shape[-2:]) != [args.expected_size, args.expected_size]:
            raise ValueError(f"Diagnostic requires native {args.expected_size}² prepared crops, got {list(source.shape[-2:])}: {sample['metadata_path']}")
        del source, target, mask
    args.output.mkdir(parents=True, exist_ok=False)
    source, target, mask = load_pair(training, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
    if validation:
        vx, vy, vm = load_pair(validation, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
    else:
        vx = vy = vm = None
    report: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA, "architecture": ARCHITECTURE,
        "diagnostic_scope": "Repeated training crop capacity only; optional unseen region of the same known material is reported separately",
        "not_generalization_validation": True, "material_id": args.material,
        "training_sample_id": training["metadata"]["sample_id"],
        "validation_sample_id": None if validation is None else validation["metadata"]["sample_id"],
        "training_crop_rectangle": training["metadata"]["crop_rectangle_top_left_xywh"],
        "validation_crop_rectangle": None if validation is None else validation["metadata"]["crop_rectangle_top_left_xywh"],
        "native_dimensions": [args.expected_size, args.expected_size], "no_augmentation": True,
        "source_images_modified": False, "target_rescaled": False,
        "target_conversion": "Unchanged uint16 codes divided by 65535 into Float32 for the active crop",
        "dataset_index_sha256": digest(args.dataset / "dataset.json"),
        "sample_manifest_sha256": {sample["metadata"]["sample_id"]: digest(sample["metadata_path"]) for sample in [training] + ([validation] if validation else [])},
        "sample_map_sha256": {sample["metadata"]["sample_id"]: {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")} for sample in [training] + ([validation] if validation else [])},
        "seed": args.seed, "learning_rate": args.learning_rate, "batch_size": 1,
        "requested_steps_per_variant": args.steps, "base_channels": args.base_channels,
        "device": str(device), "torch": str(torch.__version__), "cached_tensors": "Only active training and optional one validation Float32 pair; no decoded dataset cache or permanent float target copies",
        "alpha_masking_enabled": args.mask_transparent_input,
        "training_loss_valid_pixel_fraction": training["loss_valid_pixel_fraction"],
        "validation_loss_valid_pixel_fraction": None if validation is None else validation["loss_valid_pixel_fraction"],
        "objectives": {"current_l1": "Centered height L1 + 16 × multiscale gradient L1 + 4 × radius-4 high-pass L1", "relative_squared": "Centered height MSE / target centered energy + mean(multiscale gradient MSE / each target gradient energy) + high-pass MSE / target high-pass energy; energies are detached loss-unit constants, never transforms on stored target", "squared_energy_floors": {"height": 1e-6, "gradient": 1e-7, "highpass": 1e-7}},
        "production_promotion": False, "variants": {},
    }
    write_json(args.output / "run.json", report)
    cancelled = False
    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for variant in args.variants:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = MaterialHeightNet(args.base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        variant_output = args.output / variant
        variant_output.mkdir()
        model.eval()
        with torch.no_grad():
            initial = fit_metrics(model(source), target, mask)
            initial_validation = None if validation is None else fit_metrics(model(vx), vy, vm)
        print(json.dumps({"event": "initial", "variant": variant, "training_fit": initial, "known_material_disjoint_region": initial_validation}), flush=True)
        model.train()
        started = time.monotonic()
        first_step = None
        upstream_step = None
        completed_steps = 0
        history: list[dict[str, Any]] = []
        sampled_peak = memory(device)
        for step in range(args.steps):
            if cancelled or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
                break
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source)
            if variant == "current_l1":
                loss, components = height_loss(prediction, target, 16, True, 4, mask)
            else:
                loss, components = squared_objective(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite repeated-crop objective: {variant} step {step + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            upstream_parameter = model.encoder[0].layers[0].weight
            upstream_before = upstream_parameter.detach().clone() if step == 1 else None
            if step == 1:
                upstream_step = {"step": 2, "encoder_gradient_finite": bool(torch.isfinite(upstream_parameter.grad).all()), "encoder_gradient_nonzero": bool(torch.count_nonzero(upstream_parameter.grad)), "encoder_gradient_norm": float(upstream_parameter.grad.norm())}
            before = model.head.weight.detach().clone() if step == 0 else None
            optimizer.step()
            if step == 1:
                upstream_step["encoder_parameters_changed"] = bool(torch.any(upstream_before != upstream_parameter))
            completed_steps = step + 1
            if step == 0:
                first_step = {"forward_backward_optimizer_succeeded": True, "native_dimensions": [int(source.shape[-1]), int(source.shape[-2])], "head_parameters_changed": bool(torch.any(before != model.head.weight)), "finite_gradient_norm": float(gradient_norm), "loss": float(loss.detach())}
                if not first_step["head_parameters_changed"]:
                    raise ValueError("Diagnostic optimizer did not change checkpoint parameters")
            if step == 0 or completed_steps % args.log_every == 0 or completed_steps == args.steps:
                current_memory = memory(device)
                for key, value in current_memory.items():
                    if isinstance(value, int):
                        sampled_peak[key] = max(int(sampled_peak.get(key, 0)), value)
                with torch.no_grad():
                    metrics = fit_metrics(model(source), target, mask)
                record = {"event": "training", "variant": variant, "step": completed_steps, "elapsed_seconds": time.monotonic() - started, "loss": float(loss.detach()), "loss_components": components, "training_fit": metrics, "memory": current_memory}
                history.append(record)
                print(json.dumps(record), flush=True)
            del prediction, loss
        model.eval()
        with torch.no_grad():
            final_prediction = model(source)
            final = fit_metrics(final_prediction, target, mask)
            final_validation = None if validation is None else fit_metrics(model(vx), vy, vm)
        save_prediction(variant_output, final_prediction, source, target, "final-training-fit")
        checkpoint = {"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": args.base_channels}, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "step": completed_steps, "diagnostic_schema": DIAGNOSTIC_SCHEMA, "objective_variant": variant, "material_id": args.material, "training_sample_id": report["training_sample_id"], "diagnostic_only_not_production": True, "target_encoding": report["target_conversion"], "seed": args.seed, "sample_map_sha256": report["sample_map_sha256"], "loss": report["objectives"][variant]}
        torch.save(checkpoint, variant_output / "checkpoint.final.pt")
        result = {"initial_training_fit": initial, "final_training_fit": final, "initial_known_material_disjoint_region": initial_validation, "final_known_material_disjoint_region": final_validation, "completed_steps": completed_steps, "elapsed_seconds": time.monotonic() - started, "first_real_step": first_step, "second_step_upstream_gradients": upstream_step, "sampled_peak_memory": sampled_peak, "checkpoint_sha256": digest(variant_output / "checkpoint.final.pt"), "history": history}
        write_json(variant_output / "summary.json", result)
        report["variants"][variant] = {key: value for key, value in result.items() if key != "history"}
        write_json(args.output / "summary.json", report)
        print(json.dumps({"event": "variant_complete", "variant": variant, "steps": completed_steps, "final_training_fit": final, "final_known_material_disjoint_region": final_validation}), flush=True)
        del model, optimizer, final_prediction
        if device.type == "mps":
            torch.mps.empty_cache()
        if cancelled:
            break
    report["cancelled"] = cancelled
    report["artifact_bytes"] = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--material", default="white_stucco_02")
    parser.add_argument("--sample-id")
    parser.add_argument("--validation-sample-id")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-minutes", type=float, default=5, help="Per-variant bound; 0 disables time bound")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--expected-size", type=int, default=1024, help="Required exact native square crop size; never resizes")
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--mask-transparent-input", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 2000 or args.log_every < 1 or args.max_minutes < 0 or args.learning_rate <= 0 or args.expected_size < 16 or len(set(args.variants)) != len(args.variants):
        parser.error("Invalid limits, learning rate, crop size or duplicate variants")
    result = run(args)
    print(json.dumps({"event": "complete", "summary": str(args.output / "summary.json"), "artifact_bytes": result["artifact_bytes"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
