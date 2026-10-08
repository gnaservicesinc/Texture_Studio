#!/usr/bin/env python3
"""Matched frozen versus rank-8 DINOv2 material-height quality experiment.

Both branches warm start from the same checked 300-update stucco head and give
each of four native crops equal balanced updates. Validation is on disjoint
regions of those known materials. Initial weights remain eligible for selection;
this experiment does not promote an app model or evaluate fresh photographs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from diagnose_material_curriculum import MATERIALS, cpu_tree, mean_metrics, round_schedule, save_checkpoint_atomic
from diagnose_material_fit import fit_metrics
from frozen_dino_height import CODE_HASHES, CODE_REVISION, MODEL_REVISION, encoder_input, state_sha256
from material_height_model import squared_objective
from probe_material_adapters import MAX_DRIVER_BYTES, ROOT, base_fingerprint, differentiable_features, install_adapters, load_checked_head, load_frozen_encoder, optimizer_tensor_bytes, synchronize
from train_material_height import choose_device, digest, find_samples, load_pair, memory, save_prediction, write_json

SCHEMA = "texture-studio-four-material-adaptation-diagnostic-v1"
ARTIFACT_BUDGET = 220 * 1024**2
CHECKPOINT_BUDGET = 25 * 1024**2


class StageLimit(RuntimeError):
    def __init__(self, reason: str, stage: str, observed: dict[str, Any]):
        super().__init__(f"{reason} at {stage}")
        self.reason, self.stage, self.observed = reason, stage, observed


def balanced_schedule(updates_per_crop: int, seed: int) -> list[int]:
    if not 2 <= updates_per_crop <= 1000:
        raise ValueError("Use 2–1000 updates per crop")
    result = round_schedule([updates_per_crop] * 4, random.Random(seed))
    if Counter(result) != Counter({index: updates_per_crop for index in range(4)}):
        raise ValueError("Balanced schedule must have exact equal per-crop update counts")
    return result


def selection_score(evaluation: dict[str, Any]) -> float:
    result = evaluation["known_material_disjoint_regions"]["mean_metrics"]["relative_squared_sum"]
    if result is None or not np.isfinite(result):
        raise ValueError("Selection needs finite mean known-region relative-squared objective")
    return float(result)


@torch.no_grad()
def inference_features(encoder: torch.nn.Module, source: torch.Tensor, size: int, hidden: int) -> torch.Tensor:
    tokens = encoder.forward_features(encoder_input(source, size))["x_norm_patchtokens"]
    side = size // 14
    if tokens.shape != (source.shape[0], side * side, hidden) or not torch.isfinite(tokens).all():
        raise ValueError("Nonfinite or wrong-shaped normalized encoder patch features")
    return tokens.transpose(1, 2).reshape(source.shape[0], hidden, side, side).contiguous()


def adapter_state(adapters: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {path: {name: getattr(module, name).detach().cpu().clone() for name in ("lora_A", "lora_B")} for path, module in adapters.items()}


def verify_selected_files(expected: dict[str, str]) -> None:
    for path, checksum in expected.items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Selected source/checkpoint changed during adaptation: {path}")


def run(args: argparse.Namespace, encoder_loader: Callable = load_frozen_encoder, head_loader: Callable = load_checked_head, encoder_contract: tuple[int, int] = (768, 12)) -> dict[str, Any]:
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("MPS adaptation refuses CPU fallback")
    if not 16 <= args.expected_size <= 1024 or not 0 < args.max_minutes <= 12 or not 0 < args.max_driver_bytes <= MAX_DRIVER_BYTES or args.evaluate_every < 1 or args.checkpoint_every < 1:
        raise ValueError("Use native ≤1K, a positive time limit of at most 12 minutes per variant, driver ≤30 GB and positive intervals")
    if not 28 <= args.encoder_size <= 518 or args.encoder_size % 14:
        raise ValueError("Encoder working size must be a multiple of 14 between 28 and 518")
    schedule = balanced_schedule(args.updates_per_crop, args.seed)
    device = choose_device(args.device)
    index_path = args.dataset / "dataset.json"
    index_bytes = index_path.read_bytes()
    found = find_samples(args.dataset, args.allow_unreviewed)
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    if digest(index_path) != index_hash:
        raise ValueError("Dataset index changed during adaptation sample selection")
    training, validation, source_files, manifests = [], [], {}, {}
    for material in MATERIALS:
        for suffix, split, destination in (("001", "train", training), ("003", "validation", validation)):
            identity = material + "_auto_" + suffix
            matches = [sample for sample in found if sample["metadata"]["sample_id"] == identity and sample["metadata"]["material_id"] == material and sample["metadata"]["split"] == split]
            if len(matches) != 1:
                raise ValueError(f"Need exact {identity} / {split}")
            sample = matches[0]
            payload = sample["metadata_path"].read_bytes()
            if json.loads(payload) != sample["metadata"]:
                raise ValueError(f"Selected manifest changed: {sample['metadata_path']}")
            manifests[identity] = payload
            source_files[str(sample["metadata_path"].resolve())] = hashlib.sha256(payload).hexdigest()
            for role, path_name in (("input", "input_path"), ("height", "height_path")):
                source_files[str(sample[path_name].resolve())] = sample["metadata"]["map_metadata"][role]["sample_sha256"]
            source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_transparent_input, return_mask=True)
            if tuple(source.shape[-2:]) != (args.expected_size, args.expected_size):
                raise ValueError("Adaptation never resizes native source or height targets")
            del source, target, mask
            destination.append(sample)
    source_files[str(args.head_checkpoint.resolve())] = digest(args.head_checkpoint)
    verify_selected_files(source_files)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "dataset-snapshot"
    snapshot.mkdir()
    (snapshot / "dataset.json").write_bytes(index_bytes)
    for identity, payload in manifests.items():
        (snapshot / (identity + ".json")).write_bytes(payload)
    implementation = args.output / "implementation"
    implementation.mkdir()
    code_hashes = {}
    for name in ("diagnose_material_adaptation.py", "probe_material_adapters.py", "frozen_dino_height.py", "diagnose_material_curriculum.py", "diagnose_material_fit.py", "material_height_model.py", "train_material_height.py", "material_dataset.py"):
        path = Path(__file__).parent / name
        (implementation / name).write_bytes(path.read_bytes())
        code_hashes[name] = digest(path)
    report: dict[str, Any] = {"schema": SCHEMA, "scope": "Matched four-material frozen-versus-LoRA experiment; unseen regions of known materials", "fresh_material_generalization_tested": False, "production_promotion": False, "materials": list(MATERIALS), "source_images_modified": False, "target_rescaled": False, "target_encoding": "Raw uint16 codes / 65535 into active Float32; no gamma or per-crop range normalization", "native_dimensions": [args.expected_size, args.expected_size], "encoder_dimensions": [args.encoder_size, args.encoder_size], "encoder_resize_only": True, "no_augmentation": True, "warm_start_bias": "Both heads already trained 300 times on white_stucco_02_auto_001; this favors stucco and is not training from scratch", "dataset_index_sha256_at_selection": index_hash, "selected_files_sha256": source_files, "sample_crop_rectangles_top_left_xywh": {sample["metadata"]["sample_id"]: sample["metadata"]["crop_rectangle_top_left_xywh"] for sample in training + validation}, "loss_valid_pixel_fractions": {sample["metadata"]["sample_id"]: sample["loss_valid_pixel_fraction"] for sample in training + validation}, "schedule": schedule, "schedule_sha256": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest(), "requested_updates_per_crop": args.updates_per_crop, "requested_steps_per_variant": len(schedule), "seed": args.seed, "head_learning_rate": 0.001, "adapter_learning_rate": 0.0001, "head_weight_decay": 1e-4, "adapter_weight_decay": 0.0, "objective": "relative_squared", "selection": "Minimum actual mean known-region relative-squared sum among initial and complete evaluated snapshots; all four regions equally weighted", "checkpoint_every_steps": args.checkpoint_every, "evaluate_every_steps": args.evaluate_every, "max_minutes_per_variant_soft_guard": args.max_minutes, "max_driver_bytes_soft_guard": args.max_driver_bytes, "soft_guard_limits": "Sampled before/after forwards, before backward and after optimizer; cannot predict an in-flight allocation peak", "artifact_budget_bytes": ARTIFACT_BUDGET, "checkpoint_budget_bytes": CHECKPOINT_BUDGET, "device": str(device), "torch_version": str(torch.__version__), "implementation_sha256": code_hashes, "variants": {}, "cancelled": False}
    write_json(args.output / "run.json", report)
    encoder, encoder_info = encoder_loader(args.model_directory, args.code_directory, device)
    initial_head, head_info = head_loader(args.head_checkpoint, torch.device("cpu"))
    report["encoder"], report["learned_head"] = encoder_info, head_info
    for sample in (training[0], validation[0]):
        identity = sample["metadata"]["sample_id"]
        expected = head_info["sample_map_sha256"].get(identity)
        actual = {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")}
        if expected != actual:
            raise ValueError("Warm-start stucco samples differ from the checked head's source pairs")
    shared_head_state = cpu_tree(initial_head.state_dict())
    initial_head_hash = state_sha256(shared_head_state)
    del initial_head
    frozen_base_hash = base_fingerprint(encoder)
    hidden, blocks = encoder_contract
    encoder.eval()
    peak_cache = memory(device)
    cache_started = time.monotonic()
    feature_cache = {}
    for sample in training + validation:
        current = memory(device)
        if current.get("mps_driver_bytes", 0) > args.max_driver_bytes:
            raise StageLimit("stopped_memory_soft_guard", "frozen_feature_cache_before_forward", current)
        source, target, mask = load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
        feature_cache[sample["metadata"]["sample_id"]] = inference_features(encoder, source, args.encoder_size, hidden).cpu().detach()
        synchronize(device)
        current = memory(device)
        for key, value in current.items():
            if isinstance(value, int):
                peak_cache[key] = max(int(peak_cache.get(key, 0)), value)
        if current.get("mps_driver_bytes", 0) > args.max_driver_bytes:
            raise StageLimit("stopped_memory_soft_guard", "frozen_feature_cache_after_forward", current)
        del source, target, mask
    encoder.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    report.update({"initial_shared_head_state_sha256": initial_head_hash, "original_pretrained_base_fingerprint": frozen_base_hash, "frozen_feature_cache_bytes": sum(value.numel() * value.element_size() for value in feature_cache.values()), "frozen_feature_cache_device": "cpu", "frozen_feature_cache_shapes": {key: list(value.shape) for key, value in feature_cache.items()}, "frozen_feature_cache_detached": all(not value.requires_grad for value in feature_cache.values()), "frozen_feature_cache_seconds": time.monotonic() - cache_started, "frozen_feature_cache_memory": peak_cache})
    cached_training = [load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True) for sample in training]
    target_hashes = [hashlib.sha256(target.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for _, target, _ in cached_training]
    report["cached_training_tensor_bytes"] = sum(value.numel() * value.element_size() for pair in cached_training for value in pair if value is not None)
    cancelled = False
    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
    old_handlers = {name: signal.signal(name, stop) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        for variant in ("frozen", "lora"):
            if cancelled:
                break
            head, _ = head_loader(args.head_checkpoint, device)
            head.load_state_dict(shared_head_state, strict=True)
            if state_sha256(head.state_dict()) != initial_head_hash:
                raise ValueError("Both variants must share exact checked initial head weights")
            adapters = {}
            if variant == "lora":
                encoder.to(device)
                torch.manual_seed(args.seed)
                adapters = install_adapters(encoder, 8, 8, hidden, blocks)
                if base_fingerprint(encoder) != frozen_base_hash:
                    raise ValueError("Adapter insertion changed original Base parameters")
            encoder.eval()
            directory = args.output / variant
            directory.mkdir()
            groups = [{"params": list(head.parameters()), "lr": 0.001, "weight_decay": 1e-4}]
            adapter_parameters = [parameter for module in adapters.values() for parameter in (module.lora_A, module.lora_B)]
            if adapter_parameters:
                groups.append({"params": adapter_parameters, "lr": 0.0001, "weight_decay": 0.0})
            optimizer = torch.optim.AdamW(groups)
            adapter_count = sum(parameter.numel() for parameter in adapter_parameters)
            head_count = sum(parameter.numel() for parameter in head.parameters())
            if encoder_contract == (768, 12) and (head_count != 328957 or (adapters and adapter_count != 442368)):
                raise ValueError("Unexpected exact Base/native-head parameter count")
            started, completed_steps, counts = time.monotonic(), 0, [0] * 4
            peak, history, proofs = memory(device), [], {}
            selected_score, selected_step, last_evaluation = None, None, None
            status = "complete"
            def guard(stage: str) -> None:
                synchronize(device)
                current = memory(device)
                for key, value in current.items():
                    if isinstance(value, int):
                        peak[key] = max(int(peak.get(key, 0)), value)
                if current.get("mps_driver_bytes", 0) > args.max_driver_bytes:
                    raise StageLimit("stopped_memory_soft_guard", stage, current)
                if time.monotonic() - started >= args.max_minutes * 60:
                    raise StageLimit("stopped_time_soft_guard", stage, current)
                if cancelled:
                    raise StageLimit("cancelled", stage, current)
            def features(sample: dict[str, Any], source: torch.Tensor, differentiable: bool = False) -> torch.Tensor:
                if not adapters:
                    return feature_cache[sample["metadata"]["sample_id"]].to(device)
                return differentiable_features(encoder, source, args.encoder_size, hidden) if differentiable else inference_features(encoder, source, args.encoder_size, hidden)
            @torch.no_grad()
            def evaluate_all(step: int) -> dict[str, Any]:
                previous_mode = head.training
                head.eval()
                results = {}
                try:
                    for key, samples, cache in (("training_fit", training, cached_training), ("known_material_disjoint_regions", validation, None)):
                        records = []
                        for index, sample in enumerate(samples):
                            guard("evaluation_before_forward")
                            source, target, mask = cache[index] if cache is not None else load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
                            prediction = head(source, features(sample, source))
                            guard("evaluation_after_forward")
                            metrics = fit_metrics(prediction, target, mask)
                            metrics["relative_squared_sum"] = sum(metrics[name] for name in ("relative_centered_height_mse", "relative_multiscale_gradient_mse", "relative_detail_highpass_mse"))
                            records.append({"material_id": sample["metadata"]["material_id"], "sample_id": sample["metadata"]["sample_id"], "loss_valid_pixel_fraction": sample["loss_valid_pixel_fraction"], "metrics": metrics})
                            del prediction, source, target, mask
                        results[key] = {"samples": records, "mean_metrics": mean_metrics(records), "sample_count": len(records)}
                finally:
                    head.train(previous_mode)
                return {"step": step, "completed_steps": step, "per_crop_update_counts": list(counts), **results}
            def payload(include_optimizer: bool, complete: bool) -> dict[str, Any]:
                result = {"schema": SCHEMA, "variant": variant, "head_config": head_info["model_config"], "head_state": cpu_tree(head.state_dict()), "adapter_state": adapter_state(adapters), "encoder": encoder_info, "learned_head": head_info, "step": completed_steps, "per_crop_update_counts": list(counts), "schedule_sha256": report["schedule_sha256"], "selected_known_region_objective": selected_score, "selected_step": selected_step, "last_complete_evaluation_step": None if last_evaluation is None else last_evaluation["step"], "complete_schedule": complete, "diagnostic_only_not_production": True, "initial_head_state_sha256": initial_head_hash, "selected_files_sha256": source_files, "target_encoding": report["target_encoding"]}
                result["encoder_size"] = args.encoder_size
                if include_optimizer:
                    result["optimizer_state"] = cpu_tree(optimizer.state_dict())
                    result["torch_rng_state"] = torch.get_rng_state()
                return result
            def record_evaluation(evaluation: dict[str, Any]) -> None:
                nonlocal selected_score, selected_step, last_evaluation
                score = selection_score(evaluation)
                last_evaluation = evaluation
                if selected_score is None or score < selected_score:
                    selected_score, selected_step = score, completed_steps
                    save_checkpoint_atomic(directory / "checkpoint.selected.pt", payload(False, False))
                record = dict(evaluation, score=score, selected_score=selected_score, selected_step=selected_step, elapsed_seconds=time.monotonic() - started)
                history.append(record)
                write_json(directory / "progress.json", {"variant": variant, "history": history})
                print(json.dumps({"event": "evaluation", "variant": variant, **record}), flush=True)
            initial_prediction_identity = None
            try:
                if adapters:
                    # B=0 must preserve both shared features and head predictions.
                    head.eval()
                    for index, sample in enumerate(training + validation):
                        guard("zero_adapter_identity_before_forward")
                        pair = cached_training[index] if index < 4 else load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
                        source, target, mask = pair
                        with torch.no_grad():
                            cached_feature = feature_cache[sample["metadata"]["sample_id"]].to(device)
                            live_feature = features(sample, source)
                            if not torch.equal(cached_feature, live_feature) or not torch.equal(head(source, cached_feature), head(source, live_feature)):
                                raise ValueError("Zero adapters changed shared initial encoder features/predictions")
                        guard("zero_adapter_identity_after_forward")
                        del source, target, mask, cached_feature, live_feature, pair
                    initial_prediction_identity = True
                record_evaluation(evaluate_all(0))
                save_checkpoint_atomic(directory / "checkpoint.latest.pt", payload(True, False))
                head.train()
                for crop_index in schedule:
                    guard("training_before_forward")
                    optimizer.zero_grad(set_to_none=True)
                    source, target, mask = cached_training[crop_index]
                    before = {path: {name: getattr(module, name).detach().clone() for name in ("lora_A", "lora_B")} for path, module in adapters.items()} if completed_steps < 2 else None
                    prediction = head(source, features(training[crop_index], source, differentiable=True))
                    loss, components = squared_objective(prediction, target, mask)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite supervised adaptation objective")
                    guard("training_after_forward_before_backward")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(list(head.parameters()) + adapter_parameters, 1.0, error_if_nonfinite=True)
                    guard("training_after_backward_before_optimizer")
                    if completed_steps < 2:
                        proof = {"step": completed_steps + 1, "gradient_norm_before_clipping": float(gradient_norm), "head_gradient_norm_after_clipping": float(head.head.weight.grad.norm()), "adapters": {}}
                        for path, module in adapters.items():
                            proof["adapters"][path] = {}
                            for name in ("lora_A", "lora_B"):
                                gradient = getattr(module, name).grad
                                if gradient is None or not torch.isfinite(gradient).all():
                                    raise ValueError("Missing/nonfinite adapter learning gradient")
                                nonzero = bool(torch.count_nonzero(gradient))
                                proof["adapters"][path][name] = {"gradient_nonzero": nonzero, "gradient_norm_after_clipping": float(gradient.norm())}
                                if name == "lora_B" or completed_steps == 1:
                                    if not nonzero:
                                        raise ValueError("B(step1)/A(step2) true adapter gradient path missing")
                    optimizer.step()
                    completed_steps += 1
                    counts[crop_index] += 1
                    if completed_steps <= 2:
                        for path, module in adapters.items():
                            for name in ("lora_A", "lora_B"):
                                proof["adapters"][path][name]["parameters_changed"] = bool(torch.any(before[path][name] != getattr(module, name)))
                        if adapters and (not any(value["lora_B"]["parameters_changed"] for value in proof["adapters"].values()) or (completed_steps == 2 and not any(value["lora_A"]["parameters_changed"] for value in proof["adapters"].values()))):
                            raise ValueError("Real adapter optimizer failed to change B(step1)/A(step2)")
                        proofs[f"step_{completed_steps}"] = proof
                    # Persist a completed update before the post-optimizer guard.
                    if completed_steps % args.checkpoint_every == 0:
                        save_checkpoint_atomic(directory / "checkpoint.latest.pt", payload(True, False))
                        print(json.dumps({"event": "training", "variant": variant, "step": completed_steps, "per_crop_updates": counts, "loss": float(loss.detach()), "loss_components": components, "elapsed_seconds": time.monotonic() - started}), flush=True)
                    del prediction, loss, before
                    guard("training_after_optimizer")
                    if completed_steps % args.evaluate_every == 0 or completed_steps == len(schedule):
                        record_evaluation(evaluate_all(completed_steps))
                        save_checkpoint_atomic(directory / "checkpoint.latest.pt", payload(True, False))
            except StageLimit as limit:
                status = limit.reason
                report["stopped_stage"] = {"variant": variant, "stage": limit.stage, "reason": limit.reason, "observed": limit.observed, "completed_updates_retained": completed_steps}
            complete = completed_steps == len(schedule) and status == "complete" and last_evaluation is not None and last_evaluation["step"] == completed_steps
            if complete and counts != [args.updates_per_crop] * 4:
                raise ValueError("Complete adaptation schedule has unequal crop updates")
            if base_fingerprint(encoder) != frozen_base_hash:
                raise ValueError("Original pretrained encoder tensors changed")
            verify_selected_files(source_files)
            save_checkpoint_atomic(directory / "checkpoint.latest.pt", payload(True, complete))
            os.replace(directory / "checkpoint.latest.pt", directory / "checkpoint.final.pt")
            exported = []
            training_complete = complete
            if complete:
                head.eval()
                try:
                    for label, sample, pair in (("final-training-fit", training[0], cached_training[0]), ("final-known-material-disjoint-region", validation[0], None)):
                        guard("export_before_forward")
                        source, target, mask = pair if pair is not None else load_pair(sample, device, mask_input_alpha=args.mask_transparent_input, return_mask=True)
                        with torch.no_grad():
                            prediction = head(source, features(sample, source))
                        guard("export_after_forward")
                        target_directory = directory / label
                        save_prediction(target_directory, prediction, source, target, label)
                        metadata_path = target_directory / (label + ".prediction.json")
                        metadata = json.loads(metadata_path.read_text())
                        metadata.update({"schema": SCHEMA, "variant": variant, "sample_id": sample["metadata"]["sample_id"], "diagnostic_only_not_production": True, "snapshot": "final weights; selected validation weights may be initial or earlier"})
                        write_json(metadata_path, metadata)
                        if mask is not None:
                            np.save(target_directory / "loss-valid.bool.npy", mask.detach().cpu().numpy()[0, 0].astype(bool), allow_pickle=False)
                        exported.append(sample["metadata"]["sample_id"])
                        del source, target, mask, prediction
                        guard("export_after_write")
                except StageLimit as limit:
                    status, complete = limit.reason, False
                    report["stopped_stage"] = {"variant": variant, "stage": limit.stage, "reason": limit.reason, "observed": limit.observed, "completed_updates_retained": completed_steps}
            result = {"status": status, "complete": complete, "training_and_final_evaluation_complete": training_complete, "completed_steps": completed_steps, "per_crop_update_counts": counts, "initial_head_state_sha256": initial_head_hash, "zero_adapter_initial_features_and_predictions_identical": initial_prediction_identity, "adapter_rank": 8 if adapters else None, "adapter_alpha": 8 if adapters else None, "adapter_parameters": adapter_count, "trainable_head_parameters": head_count, "initial_evaluation": None if not history else history[0], "final_complete_evaluation": last_evaluation, "final_training_fit": None if last_evaluation is None else last_evaluation["training_fit"], "final_known_material_disjoint_regions": None if last_evaluation is None else last_evaluation["known_material_disjoint_regions"], "selected_step": selected_step, "selected_known_region_objective": selected_score, "selection_improved_initial": bool(history and selected_score < history[0]["score"]), "phase_history": history, "gradient_proof": proofs, "original_pretrained_base_unchanged": True, "elapsed_seconds": time.monotonic() - started, "sampled_peak_memory": peak, "optimizer_tensor_state_bytes": optimizer_tensor_bytes(optimizer), "exported_sample_ids": exported, "final_checkpoint_sha256": digest(directory / "checkpoint.final.pt"), "selected_checkpoint_sha256": None if not (directory / "checkpoint.selected.pt").exists() else digest(directory / "checkpoint.selected.pt")}
            write_json(directory / "summary.json", result)
            report["variants"][variant] = result
            report["artifact_bytes"] = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
            report["checkpoint_bytes"] = sum(path.stat().st_size for path in args.output.rglob("checkpoint.*.pt"))
            if report["artifact_bytes"] > ARTIFACT_BUDGET or report["checkpoint_bytes"] > CHECKPOINT_BUDGET:
                raise ValueError("Bounded adaptation artifact/checkpoint budget exceeded")
            write_json(args.output / "summary.json", report)
            print(json.dumps({"event": "variant_complete", "variant": variant, "complete": complete, "steps": completed_steps, "selected_step": selected_step, "selected_known_region_objective": selected_score}), flush=True)
            del head, optimizer
            if device.type == "mps":
                torch.mps.empty_cache()
            if not complete:
                break
    finally:
        for name, old_handler in old_handlers.items():
            signal.signal(name, old_handler)
    for expected, (_, target, _) in zip(target_hashes, cached_training):
        if hashlib.sha256(target.detach().cpu().contiguous().numpy().tobytes()).hexdigest() != expected:
            raise ValueError("Active native height target changed")
    verify_selected_files(source_files)
    if encoder_info.get("checkpoint_path") and digest(Path(encoder_info["checkpoint_path"])) != encoder_info["checkpoint_sha256"]:
        raise ValueError("Cached pretrained Base checkpoint changed")
    if encoder_info.get("official_source_directory"):
        for path, expected in CODE_HASHES.items():
            if digest(Path(encoder_info["official_source_directory"]) / path) != expected:
                raise ValueError("Cached official source changed")
    report["cancelled"] = cancelled
    report["matched_complete_schedules"] = all(report["variants"].get(name, {}).get("complete", False) for name in ("frozen", "lora"))
    report["dataset_index_sha256_after"] = digest(index_path)
    report["dataset_index_changed_during_run"] = report["dataset_index_sha256_after"] != index_hash
    report["source_files_and_native_targets_unchanged"] = True
    report["original_pretrained_base_fingerprint_after"] = base_fingerprint(encoder)
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, default=ROOT / "out/material-training/frozen-dino-stucco-01/frozen_features/checkpoint.final.pt")
    parser.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
    parser.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
    parser.add_argument("--updates-per-crop", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--expected-size", type=int, default=1024)
    parser.add_argument("--encoder-size", type=int, default=518)
    parser.add_argument("--evaluate-every", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=12)
    parser.add_argument("--max-driver-bytes", type=int, default=MAX_DRIVER_BYTES)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--mask-transparent-input", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"event": "complete", "summary": str(args.output / "summary.json"), "matched_complete_schedules": result["matched_complete_schedules"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, StageLimit) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
