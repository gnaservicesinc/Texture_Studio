#!/usr/bin/env python3
"""Matched frozen-DINOv2 versus zero-feature native-height capacity diagnostic.

This trains only a small native-pixel head, not the encoder or a LoRA. The same
initial head weights, repeated training crop, raw uint16 height, objective and
optimizer are used in both variants. A disjoint known-material region measures
transfer within that material; this cannot establish fresh-material accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from diagnose_material_fit import fit_metrics, select_material_samples
from frozen_dino_height import ARCHITECTURE, DIAGNOSTIC_SCHEMA, CODE_REVISION, MODEL_REVISION, ConditionedHeightNet, extract_features, load_frozen_encoder, state_sha256
from material_height_model import squared_objective
from train_material_height import choose_device, digest, load_pair, memory, save_prediction, write_json

VARIANTS = ("zero_features", "frozen_features")
ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace, encoder_loader: Callable = load_frozen_encoder) -> dict[str, Any]:
    if args.expected_size > 1024:
        raise ValueError("This bounded frozen-feature diagnostic supports native crops up to1024; 2K needs a separate artifact budget")
    device = choose_device(args.device)
    encoder_device = choose_device(args.encoder_device or args.device)
    training, validation = select_material_samples(args.dataset, args.material, args.sample_id, args.validation_sample_id, False, args.allow_unreviewed)
    pairs = []
    for sample in (training, validation):
        source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_transparent_input, return_mask=True)
        if tuple(source.shape[-2:]) != (args.expected_size, args.expected_size):
            raise ValueError(f"Expected exact native {args.expected_size}² crop: {sample['metadata_path']}")
        pairs.append((source, target, mask))
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    feature_started = time.monotonic()
    encoder, encoder_provenance = encoder_loader(args.model_directory, args.code_directory, encoder_device)
    feature_cache = [extract_features(encoder, source.to(encoder_device), args.encoder_size).to(device) for source, _, _ in pairs]
    if any(parameter.grad is not None or parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Frozen encoder acquired gradients")
    encoder_memory = memory(encoder_device)
    del encoder
    if encoder_device.type == "mps":
        torch.mps.empty_cache()
    source, target, mask = (value.to(device) if value is not None else None for value in pairs[0])
    vx, vy, vm = (value.to(device) if value is not None else None for value in pairs[1])
    del pairs
    feature_seconds = time.monotonic() - feature_started
    torch.manual_seed(args.seed)
    initial_model = ConditionedHeightNet(args.base_channels, 768, args.projection_channels)
    initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
    initial_state_hash = state_sha256(initial_state)
    trainable_count = sum(parameter.numel() for parameter in initial_model.parameters())
    del initial_model
    report: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA, "architecture": ARCHITECTURE,
        "diagnostic_scope": "Repeated training crop capacity; unseen disjoint region of the same known material reported separately",
        "validation_scope": "unseen regions of known materials", "not_fresh_material_generalization": True,
        "training_sample_id": training["metadata"]["sample_id"], "validation_sample_id": validation["metadata"]["sample_id"],
        "material_id": args.material, "dataset_index_sha256": digest(args.dataset / "dataset.json"),
        "sample_manifest_sha256": {sample["metadata"]["sample_id"]: digest(sample["metadata_path"]) for sample in (training, validation)},
        "sample_map_sha256": {sample["metadata"]["sample_id"]: {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")} for sample in (training, validation)},
        "crop_rectangles_top_left_xywh": {sample["metadata"]["sample_id"]: sample["metadata"]["crop_rectangle_top_left_xywh"] for sample in (training, validation)},
        "source_images_modified": False, "target_rescaled": False,
        "target_encoding": "Raw source uint16 codes / 65535 into active Float32; no gamma, per-crop range scaling or source modification",
        "native_head_input": "Full native linear RGB; no resize or augmentation",
        "encoder_input": "Encoder-only linear-to-sRGB conversion; whole crop bicubic antialias resize with no center crop; ImageNet mean(0.485,0.456,0.406)/std(0.229,0.224,0.225)",
        "native_dimensions": [args.expected_size, args.expected_size], "encoder_dimensions": [args.encoder_size, args.encoder_size],
        "feature_grid_shape": list(feature_cache[0].shape), "feature_cache_bytes": sum(value.numel() * value.element_size() for value in feature_cache),
        "features": "Frozen final-layer normalized patch tokens; learned low-grid 1×1 projection before bilinear upsampling to native head bottleneck",
        "not_lora": True, "encoder": encoder_provenance, "encoder_extraction_seconds": feature_seconds, "encoder_extraction_memory": encoder_memory,
        "trainable_parameters": trainable_count, "base_channels": args.base_channels, "projection_channels": args.projection_channels,
        "initial_head_state_sha256": initial_state_hash, "matched_initialization": True,
        "objective": "relative_squared", "objective_description": "Centered height, native/multiscale gradients and radius4 highpass MSE divided by fixed target energies; equal component weights; no target-value scaling",
        "energy_floors": {"height": 1e-6, "gradient": 1e-7, "highpass": 1e-7},
        "batch_size": 1, "learning_rate": args.learning_rate, "weight_decay": 1e-4, "seed": args.seed,
        "steps_per_variant": args.steps, "device": str(device), "encoder_device": str(encoder_device), "torch_version": str(torch.__version__),
        "alpha_masking_enabled": args.mask_transparent_input,
        "training_loss_valid_pixel_fraction": training["loss_valid_pixel_fraction"], "validation_loss_valid_pixel_fraction": validation["loss_valid_pixel_fraction"],
        "production_promotion": False, "variants": {},
    }
    implementation = args.output / "implementation"
    implementation.mkdir()
    report["implementation"] = {}
    for name in ("diagnose_frozen_dino_height.py", "frozen_dino_height.py", "diagnose_material_fit.py", "material_height_model.py", "train_material_height.py", "material_dataset.py"):
        original = Path(__file__).parent / name
        (implementation / name).write_bytes(original.read_bytes())
        report["implementation"][name] = digest(original)
    write_json(args.output / "run.json", report)
    cancelled = False
    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for variant in args.variants:
        model = ConditionedHeightNet(args.base_channels, 768, args.projection_channels)
        model.load_state_dict(initial_state, strict=True)
        if state_sha256(model.state_dict()) != initial_state_hash:
            raise ValueError("Variant did not receive identical initial head weights")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        features = feature_cache if variant == "frozen_features" else [torch.zeros_like(value) for value in feature_cache]
        model.eval()
        with torch.no_grad():
            initial_prediction = model(source, features[0])
            initial_is_flat = bool(torch.equal(initial_prediction, torch.full_like(initial_prediction, 0.5)))
            if not initial_is_flat:
                raise ValueError("Matched control initial prediction must be identically0.5")
            initial = fit_metrics(initial_prediction, target, mask)
            initial_validation = fit_metrics(model(vx, features[1]), vy, vm)
            del initial_prediction
        print(json.dumps({"event": "initial", "variant": variant, "training_fit": initial, "known_material_disjoint_region": initial_validation}), flush=True)
        started, completed_steps = time.monotonic(), 0
        peak, history, gradient_proof = memory(device), [], {}
        model.train()
        for step in range(args.steps):
            if cancelled or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
                break
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source, features[0])
            loss, components = squared_objective(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite diagnostic objective: {variant}, step{step+1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            upstream = model.encoder[0].layers[0].weight
            if step in (0, 1):
                before_head, before_upstream = model.head.weight.detach().clone(), upstream.detach().clone()
            optimizer.step()
            completed_steps = step + 1
            if step in (0, 1):
                gradient_proof[f"step_{step+1}"] = {"finite_gradient_norm": float(grad_norm), "head_gradient_norm": float(model.head.weight.grad.norm()), "head_parameters_changed": bool(torch.any(before_head != model.head.weight)), "native_rgb_encoder_gradient_nonzero": bool(torch.count_nonzero(upstream.grad)), "native_rgb_encoder_gradient_norm": float(upstream.grad.norm()), "native_rgb_encoder_parameters_changed": bool(torch.any(before_upstream != upstream)), "feature_projection_gradient_norm": float(model.feature_projection.weight.grad.norm()), "cached_encoder_features_require_grad": features[0].requires_grad, "native_output_dimensions": list(prediction.shape[-2:])}
                if step == 0 and not gradient_proof["step_1"]["head_parameters_changed"]:
                    raise ValueError("Diagnostic optimizer did not update the native head")
                if step == 1 and variant == "frozen_features" and not bool(torch.count_nonzero(model.feature_projection.weight.grad)):
                    raise ValueError("Frozen feature projection did not receive a real training gradient")
            if step == 0 or completed_steps % args.log_every == 0 or completed_steps == args.steps:
                current = memory(device)
                for key, value in current.items():
                    if isinstance(value, int):
                        peak[key] = max(int(peak.get(key, 0)), value)
                with torch.no_grad():
                    metrics = fit_metrics(model(source, features[0]), target, mask)
                record = {"event": "training", "variant": variant, "step": completed_steps, "elapsed_seconds": time.monotonic() - started, "loss": float(loss.detach()), "loss_components": components, "training_fit": metrics, "memory": current}
                history.append(record)
                print(json.dumps(record), flush=True)
            del prediction, loss
        elapsed_training = time.monotonic() - started
        model.eval()
        with torch.no_grad():
            final_prediction = model(source, features[0])
            final_validation_prediction = model(vx, features[1])
            final = fit_metrics(final_prediction, target, mask)
            final_validation = fit_metrics(final_validation_prediction, vy, vm)
        directory = args.output / variant
        directory.mkdir()
        # Export one training and one disjoint-region result per variant only.
        for label, prediction, rgb, height, valid in (("final-training-fit", final_prediction, source, target, mask), ("final-known-material-disjoint-region", final_validation_prediction, vx, vy, vm)):
            save_prediction(directory / label, prediction, rgb, height, label)
            export_metadata = directory / label / f"{label}.prediction.json"
            metadata = json.loads(export_metadata.read_text())
            metadata.update({"schema": DIAGNOSTIC_SCHEMA, "diagnostic_only_not_production": True, "condition_variant": variant})
            write_json(export_metadata, metadata)
            if valid is not None:
                np.save(directory / label / "loss-valid.bool.npy", valid.detach().cpu().numpy()[0, 0].astype(bool), allow_pickle=False)
                write_json(directory / label / "loss-valid.json", {"meaning": "Input-alpha loss/metric eligibility with 8px margin, not model-estimated confidence", "excluded_prediction_pixels_unassessed": True})
        checkpoint = {"schema": DIAGNOSTIC_SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": args.base_channels, "feature_channels": 768, "projection_channels": args.projection_channels}, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "encoder": encoder_provenance, "encoder_size": args.encoder_size, "variant": variant, "step": completed_steps, "initial_state_sha256": initial_state_hash, "diagnostic_only_not_production": True, "sample_map_sha256": report["sample_map_sha256"], "target_encoding": report["target_encoding"]}
        torch.save(checkpoint, directory / "checkpoint.final.pt")
        result = {"initial_training_fit": initial, "initial_prediction_identically_0_5": initial_is_flat, "final_training_fit": final, "initial_known_material_disjoint_region": initial_validation, "final_known_material_disjoint_region": final_validation, "completed_steps": completed_steps, "training_seconds": elapsed_training, "seconds_per_training_step_including_periodic_metrics": elapsed_training / max(completed_steps, 1), "gradient_proof": gradient_proof, "sampled_peak_memory": peak, "checkpoint_sha256": digest(directory / "checkpoint.final.pt"), "initial_head_state_sha256": initial_state_hash}
        write_json(directory / "summary.json", dict(result, history=history))
        report["variants"][variant] = result
        write_json(args.output / "summary.json", report)
        print(json.dumps({"event": "variant_complete", "variant": variant, "steps": completed_steps, "final_training_fit": final, "final_known_material_disjoint_region": final_validation}), flush=True)
        del model, optimizer, final_prediction, final_validation_prediction, features
        if device.type == "mps":
            torch.mps.empty_cache()
        if cancelled:
            break
    report["cancelled"] = cancelled
    report["matched_completed_steps"] = len({value["completed_steps"] for value in report["variants"].values()}) == 1 and len(report["variants"]) == len(args.variants)
    report["artifact_bytes"] = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    if report["artifact_bytes"] > 200 * 1024 * 1024:
        raise ValueError("Diagnostic exceeded its 200MiB artifact bound")
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--material", default="white_stucco_02")
    parser.add_argument("--sample-id")
    parser.add_argument("--validation-sample-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
    parser.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-minutes", type=float, default=5, help="Per-variant bound; 0 disables")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--projection-channels", type=int, default=12)
    parser.add_argument("--encoder-size", type=int, default=518)
    parser.add_argument("--expected-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--encoder-device", choices=("mps", "cpu"))
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--mask-transparent-input", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or args.log_every < 1 or args.max_minutes < 0 or args.learning_rate <= 0 or not 16 <= args.expected_size <= 1024 or len(set(args.variants)) != len(args.variants) or not 28 <= args.encoder_size <= 518 or args.encoder_size % 14:
        parser.error("Invalid step/time/size/learning limits or duplicate variants")
    report = run(args)
    print(json.dumps({"event": "complete", "summary": str(args.output / "summary.json"), "artifact_bytes": report["artifact_bytes"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
