#!/usr/bin/env python3
"""Two-update DINOv2 Base LoRA gradient/memory feasibility probe.

Uses cached pinned weights and a checked, already learned native height head.
This does not measure output quality, generalization, GIANT feasibility or full
fine-tuning memory. Source data and all input checkpoints remain read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from diagnose_material_curriculum import cpu_tree, save_checkpoint_atomic
from frozen_dino_height import ARCHITECTURE as HEAD_ARCHITECTURE, CODE_HASHES, CODE_REVISION, DIAGNOSTIC_SCHEMA as HEAD_SCHEMA, MODEL_REVISION, MODEL_SHA256, ConditionedHeightNet, encoder_input, file_sha256, load_frozen_encoder, state_sha256
from material_height_model import squared_objective
from material_resources import configure_training_resources, resolve_training_budget, training_resources
from train_material_height import choose_device, digest, find_samples, load_pair, memory, write_json

SCHEMA = "texture-studio-material-adapter-feasibility-v1"
HEAD_SHA256 = "9c6038d82fc0e371112226c4d85b709da3d586fe5e0f4ccc80eb0a82485f8f5d"
ROOT = Path(__file__).resolve().parents[1]
MAX_DRIVER_BYTES = training_resources()["maximum_training_bytes"]


class DifferentiableHeightNet(ConditionedHeightNet):
    """The checked head's exact computation, allowing feature-input gradients.

The frozen-feature diagnostic deliberately rejects features with a graph. This
new subclass retains its state layout and all operators without that guard.
"""
    def forward(self, rgb: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or features.ndim != 4 or features.shape[:2] != (rgb.shape[0], self.feature_channels):
            raise ValueError("Expected native RGB plus batch-aligned feature grid")
        skips, value = [], rgb
        for level, block in enumerate(self.encoder):
            if level:
                value = F.avg_pool2d(value, 2)
            value = block(value)
            skips.append(value)
        projected = self.feature_projection(features)
        projected = F.interpolate(projected, size=value.shape[-2:], mode="bilinear", align_corners=False)
        value = self.feature_fusion(torch.cat((value, projected), dim=1))
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return torch.sigmoid(self.head(value))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or alpha <= 0:
            raise ValueError("LoRA requires a linear layer, positive rank and alpha")
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + ((value @ self.lora_A.T) @ self.lora_B.T) * self.scale


def install_adapters(encoder: nn.Module, rank: int = 8, alpha: float = 8, hidden: int = 768, blocks: int = 12) -> dict[str, LoRALinear]:
    if not hasattr(encoder, "blocks") or len(encoder.blocks) != blocks:
        raise ValueError("Encoder block count does not match the checked contract")
    encoder.requires_grad_(False)
    targets = {}
    # Only exact attention children are wrapped; patch projection is excluded.
    for index, block in enumerate(encoder.blocks):
        for name, output in (("qkv", hidden * 3), ("proj", hidden)):
            original = getattr(block.attn, name, None)
            if not isinstance(original, nn.Linear) or original.in_features != hidden or original.out_features != output or original.weight.dtype != torch.float32:
                raise ValueError(f"Unexpected DINOv2 attention linear layout: block {index} {name}")
            targets[f"blocks.{index}.attn.{name}"] = original
    result = {}
    for path, original in targets.items():
        index, name = int(path.split(".")[1]), path.split(".")[-1]
        wrapped = LoRALinear(original, rank, alpha)
        setattr(encoder.blocks[index].attn, name, wrapped)
        result[path] = wrapped
    expected_count = blocks * rank * (hidden + hidden * 3 + hidden + hidden)
    if sum(module.lora_A.numel() + module.lora_B.numel() for module in result.values()) != expected_count:
        raise ValueError("Unexpected exact adapter parameter count")
    return result


def base_fingerprint(encoder: nn.Module) -> str:
    canonical = {}
    for name, parameter in encoder.named_parameters():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError(f"Original encoder parameter was not frozen: {name}")
        original_name = re.sub(r"(\.attn\.(?:qkv|proj))\.base\.", r"\1.", name)
        canonical[original_name] = parameter
    return state_sha256(canonical)


def load_checked_head(path: Path, device: torch.device, expected_sha256: str = HEAD_SHA256) -> tuple[nn.Module, dict[str, Any]]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Known learned-head checkpoint SHA256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != HEAD_SCHEMA or checkpoint.get("architecture") != HEAD_ARCHITECTURE or checkpoint.get("model_config") != {"base_channels": 12, "feature_channels": 768, "projection_channels": 12} or checkpoint.get("variant") != "frozen_features" or checkpoint.get("step") != 300 or checkpoint.get("diagnostic_only_not_production") is not True:
        raise ValueError("Learned head must be the checked 300-update frozen-feature diagnostic")
    if checkpoint.get("encoder", {}).get("checkpoint_sha256") != MODEL_SHA256 or checkpoint["encoder"].get("official_meta_code_revision") != CODE_REVISION or checkpoint.get("encoder_size") != 518:
        raise ValueError("Learned head encoder provenance differs from the pinned Base encoder")
    model = DifferentiableHeightNet(**checkpoint["model_config"])
    state = checkpoint.get("model_state", {})
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Learned head state must be finite Float32")
    model.load_state_dict(state, strict=True)
    if not torch.count_nonzero(model.head.weight):
        raise ValueError("Learned native head must have nonzero output weights to test adapter gradients")
    return model.to(device), {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": expected_sha256, "source_schema": HEAD_SCHEMA, "source_step": 300, "source_variant": "frozen_features", "model_config": checkpoint["model_config"], "sample_map_sha256": checkpoint["sample_map_sha256"], "diagnostic_only_not_production": True}


def differentiable_features(encoder: nn.Module, linear_rgb: torch.Tensor, size: int = 518, hidden: int = 768) -> torch.Tensor:
    # Call official differentiable forward_features directly: never the cached
    # no_grad extract_features helper used by the frozen-head diagnostic.
    tokens = encoder.forward_features(encoder_input(linear_rgb, size))["x_norm_patchtokens"]
    side = size // 14
    if tokens.shape != (linear_rgb.shape[0], side * side, hidden) or not torch.isfinite(tokens).all() or not tokens.requires_grad:
        raise ValueError("Expected finite differentiable normalized patch features")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], hidden, side, side).contiguous()


def optimizer_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(value.numel() * value.element_size() for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run(args: argparse.Namespace, encoder_loader: Callable = load_frozen_encoder, head_loader: Callable = load_checked_head, encoder_contract: tuple[int, int] = (768, 12)) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" and args.device == "mps":
        raise ValueError("MPS probe refuses CPU fallback; unset PYTORCH_ENABLE_MPS_FALLBACK")
    if not 0 < args.max_seconds <= 120 or not 16 <= args.expected_size <= 1024 or args.rank != 8 or args.alpha != 8 or args.learning_rate <= 0 or not 28 <= args.encoder_size <= 518 or args.encoder_size % 14:
        raise ValueError("Use rank 8 / alpha 8, native≤1K, aligned encoder≤518, time≤120s and positive learning rate")
    device = choose_device(args.device)
    args.max_driver_bytes, resource_limits = resolve_training_budget(args.max_driver_bytes, device)
    resource_limits.update(configure_training_resources(args.max_driver_bytes, device))
    found = find_samples(args.dataset, args.allow_unreviewed)
    matches = [sample for sample in found if sample["metadata"]["sample_id"] == args.sample_id and sample["metadata"]["split"] == "train"]
    if len(matches) != 1:
        raise ValueError("Need one exact eligible training sample for this feasibility probe")
    sample = matches[0]
    manifest_payload = sample["metadata_path"].read_bytes()
    if json.loads(manifest_payload) != sample["metadata"]:
        raise ValueError("Selected sample manifest changed before probe preflight")
    source, target, mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_transparent_input, return_mask=True)
    if tuple(source.shape[-2:]) != (args.expected_size, args.expected_size):
        raise ValueError("Probe never resizes the native source/target pair")
    target_original_digest = hashlib.sha256(target.contiguous().numpy().tobytes()).hexdigest()
    source_files = {
        str(sample["metadata_path"].resolve()): hashlib.sha256(manifest_payload).hexdigest(),
        str(sample["input_path"].resolve()): sample["metadata"]["map_metadata"]["input"]["sample_sha256"],
        str(sample["height_path"].resolve()): sample["metadata"]["map_metadata"]["height"]["sample_sha256"],
        str(args.head_checkpoint.resolve()): digest(args.head_checkpoint),
    }
    for path, expected in source_files.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Selected sample/head changed during probe preflight: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema": SCHEMA, "scope": "Two genuine differentiable adapter/head updates; gradient and local MPS memory/time feasibility only", "quality_training": False, "fresh_material_generalization_tested": False, "giant_model_feasibility_tested": False, "full_encoder_finetuning_memory_tested": False, "production_promotion": False, "source_images_modified": False, "target_rescaled": False, "target_encoding": "Unchanged raw uint16 source codes / 65535 into active Float32", "native_dimensions": [args.expected_size, args.expected_size], "encoder_dimensions": [args.encoder_size, args.encoder_size], "encoder_resize_only": True, "device": str(device), "torch_version": str(torch.__version__), "rank": args.rank, "alpha": args.alpha, "steps_requested": 2, "max_seconds_soft_guard": args.max_seconds, "max_driver_bytes_soft_guard": args.max_driver_bytes, "soft_guard_limits": "Checked between synchronous stages; does not predict allocation peaks or interrupt an in-flight Metal operation", "sample_id": args.sample_id, "sample_crop_rectangle_top_left_xywh": sample["metadata"]["crop_rectangle_top_left_xywh"], "loss_valid_pixel_fraction": sample["loss_valid_pixel_fraction"], "selected_source_sha256": source_files, "dataset_index_sha256_at_selection": digest(args.dataset / "dataset.json"), "steps": [], "status": "preflight_complete"}
    report["resources"] = resource_limits
    write_json(args.output / "run.json", report)
    implementation = args.output / "implementation"
    implementation.mkdir()
    report["implementation_sha256"] = {}
    for name in ("probe_material_adapters.py", "material_resources.py", "frozen_dino_height.py", "material_height_model.py", "train_material_height.py", "material_dataset.py", "diagnose_material_curriculum.py", "diagnose_material_fit.py"):
        path = Path(__file__).parent / name
        (implementation / name).write_bytes(path.read_bytes())
        report["implementation_sha256"][name] = digest(path)
    (args.output / "sample.snapshot.json").write_bytes(manifest_payload)
    started = time.monotonic()
    encoder, encoder_info = encoder_loader(args.model_directory, args.code_directory, device)
    head, head_info = head_loader(args.head_checkpoint, device)
    report["encoder"], report["learned_head"] = encoder_info, head_info
    expected_maps = head_info["sample_map_sha256"].get(args.sample_id)
    actual_maps = {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")}
    if expected_maps != actual_maps:
        raise ValueError("Probe pair differs from the verified learned-head source pair")
    frozen_before = base_fingerprint(encoder)
    torch.manual_seed(args.seed)
    hidden, blocks = encoder_contract
    adapters = install_adapters(encoder, args.rank, args.alpha, hidden, blocks)
    if base_fingerprint(encoder) != frozen_before:
        raise ValueError("Adapter insertion changed original pretrained tensors")
    encoder.eval()  # deterministic official blocks; autograd remains enabled.
    head.train()
    adapter_parameters = [parameter for module in adapters.values() for parameter in (module.lora_A, module.lora_B)]
    adapter_count = sum(parameter.numel() for parameter in adapter_parameters)
    head_count = sum(parameter.numel() for parameter in head.parameters())
    report.update({"original_pretrained_fingerprint_before": frozen_before, "exact_target_modules": list(adapters), "adapter_parameters": adapter_count, "trainable_head_parameters": head_count, "trainable_total_parameters": adapter_count + head_count, "encoder_frozen_provenance_describes_original_base_before_adapter_insertion": True, "optimizer_learning_rates": {"adapters": args.learning_rate, "head": args.learning_rate}, "adapter_weight_decay": 0.0, "head_weight_decay": 1e-4, "initialization": "A Kaiming uniform; B exactly zero; alpha/rank=1. Zero adapter weight decay separates parameter changes from decoupled decay", "head_initial_state_sha256": state_sha256(head.state_dict())})
    if encoder_contract == (768, 12) and (adapter_count != 442368 or head_count != 328957):
        raise ValueError("Pinned Base adapter/native-head parameter count mismatch")
    optimizer = torch.optim.AdamW([{"params": adapter_parameters, "weight_decay": 0.0}, {"params": head.parameters(), "weight_decay": 1e-4}], lr=args.learning_rate)
    source, target, mask = (value.to(device) if value is not None else None for value in (source, target, mask))
    synchronize(device)
    peak = memory(device)
    status = "complete"
    for step in (1, 2):
        if time.monotonic() - started >= args.max_seconds:
            status = "stopped_time_soft_guard"
            break
        before_forward_memory = memory(device)
        for key, value in before_forward_memory.items():
            if isinstance(value, int):
                peak[key] = max(int(peak.get(key, 0)), value)
        if before_forward_memory.get("mps_driver_bytes", 0) > args.max_driver_bytes:
            report["stopped_before_forward"] = {"step": step, "memory": before_forward_memory, "forward_performed": False}
            status = "stopped_memory_soft_guard"
            break
        optimizer.zero_grad(set_to_none=True)
        before = {path: {name: getattr(module, name).detach().clone() for name in ("lora_A", "lora_B")} for path, module in adapters.items()}
        head_weight_before = head.head.weight.detach().clone()
        synchronize(device)
        step_started = time.monotonic()
        features = differentiable_features(encoder, source, args.encoder_size, hidden)
        prediction = head(source, features)
        loss, components = squared_objective(prediction, target, mask)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite differentiable material objective")
        synchronize(device)
        forward_seconds = time.monotonic() - step_started
        forward_memory = memory(device)
        for key, value in forward_memory.items():
            if isinstance(value, int):
                peak[key] = max(int(peak.get(key, 0)), value)
        if forward_memory.get("mps_driver_bytes", 0) > args.max_driver_bytes or time.monotonic() - started >= args.max_seconds:
            report["stopped_after_forward"] = {"step": step, "forward_seconds": forward_seconds, "memory": forward_memory, "backward_performed": False}
            status = "stopped_memory_soft_guard" if forward_memory.get("mps_driver_bytes", 0) > args.max_driver_bytes else "stopped_time_soft_guard"
            del prediction, loss, features
            break
        backward_started = time.monotonic()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter_parameters + list(head.parameters()), 1.0, error_if_nonfinite=True)
        gradients = {}
        for path, module in adapters.items():
            gradients[path] = {}
            for name in ("lora_A", "lora_B"):
                gradient = getattr(module, name).grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError(f"Missing/nonfinite adapter gradient: {path}.{name}")
                gradients[path][name] = {"gradient_norm_after_clipping": float(gradient.norm()), "gradient_nonzero": bool(torch.count_nonzero(gradient))}
            if not gradients[path]["lora_B"]["gradient_nonzero"] or (step == 2 and not gradients[path]["lora_A"]["gradient_nonzero"]):
                raise ValueError(f"Genuine B(step1)/A(step2) adapter learning path absent: {path}")
        optimizer.step()
        synchronize(device)
        backward_optimizer_seconds = time.monotonic() - backward_started
        changed_a = changed_b = False
        for path, module in adapters.items():
            for name in ("lora_A", "lora_B"):
                changed = bool(torch.any(before[path][name] != getattr(module, name)))
                gradients[path][name]["parameters_changed"] = changed
                changed_a = changed_a or (name == "lora_A" and changed)
                changed_b = changed_b or (name == "lora_B" and changed)
        if not changed_b or (step == 2 and not changed_a):
            raise ValueError("Adapter optimizer did not change B(step1) and A(step2) tensors")
        current = memory(device)
        for key, value in current.items():
            if isinstance(value, int):
                peak[key] = max(int(peak.get(key, 0)), value)
        record = {"step": step, "native_dimensions": list(prediction.shape[-2:]), "loss": float(loss.detach()), "loss_components": components, "features_requires_grad": features.requires_grad, "gradient_norm_before_clipping": float(norm), "head_gradient_norm_after_clipping": float(head.head.weight.grad.norm()), "head_output_weights_changed": bool(torch.any(head_weight_before != head.head.weight)), "adapter_gradients": gradients, "any_A_parameters_changed": changed_a, "any_B_parameters_changed": changed_b, "forward_seconds_synchronized": forward_seconds, "backward_optimizer_seconds_synchronized": backward_optimizer_seconds, "step_seconds_synchronized": time.monotonic() - step_started, "memory_after_forward": forward_memory, "memory_after_optimizer": current, "optimizer_tensor_state_bytes": optimizer_tensor_bytes(optimizer)}
        report["steps"].append(record)
        write_json(args.output / "progress.json", report)
        print(json.dumps({"event": "real_adapter_step", **record}), flush=True)
        del prediction, loss, features, before
        if current.get("mps_driver_bytes", 0) > args.max_driver_bytes:
            report["stopped_after_optimizer"] = {"step": step, "memory": current, "completed_update_retained": True}
            status = "stopped_memory_soft_guard"
            break
        if time.monotonic() - started >= args.max_seconds:
            status = "stopped_time_soft_guard"
            break
    frozen_after = base_fingerprint(encoder)
    if frozen_after != frozen_before:
        raise ValueError("Original pretrained encoder tensors changed during adapter probe")
    for path, expected in source_files.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Read-only source/head checkpoint changed during probe: {path}")
    if encoder_info.get("checkpoint_path"):
        if digest(Path(encoder_info["checkpoint_path"])) != encoder_info["checkpoint_sha256"]:
            raise ValueError("Cached pretrained encoder checkpoint changed during probe")
    if encoder_info.get("official_source_directory"):
        for path, expected in CODE_HASHES.items():
            if digest(Path(encoder_info["official_source_directory"]) / path) != expected:
                raise ValueError("Cached official encoder source changed during probe")
    target_after = hashlib.sha256(target.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if target_after != target_original_digest:
        raise ValueError("Probe altered the active native height target")
    report.update({"status": status, "completed_updates": len(report["steps"]), "original_pretrained_fingerprint_after": frozen_after, "original_pretrained_tensors_unchanged": True, "target_float32_batch_unchanged": True, "source_files_unchanged": True, "sampled_peak_memory": peak, "elapsed_seconds_including_load_fingerprints": time.monotonic() - started, "optimizer_tensor_state_bytes": optimizer_tensor_bytes(optimizer), "head_final_state_sha256": state_sha256(head.state_dict())})
    checkpoint = {"schema": SCHEMA, "feasibility_only_not_quality_model": True, "production_promotion": False, "encoder": encoder_info, "learned_head": head_info, "rank": args.rank, "alpha": args.alpha, "adapter_state": {path: {"lora_A": module.lora_A.detach().cpu(), "lora_B": module.lora_B.detach().cpu()} for path, module in adapters.items()}, "head_state": cpu_tree(head.state_dict()), "head_config": head_info["model_config"], "completed_updates": len(report["steps"]), "selected_source_sha256": source_files, "target_encoding": report["target_encoding"]}
    save_checkpoint_atomic(args.output / "adapter-head.feasibility.pt", checkpoint)
    report["saved_checkpoint_sha256"] = digest(args.output / "adapter-head.feasibility.pt")
    report["saved_checkpoint_bytes"] = (args.output / "adapter-head.feasibility.pt").stat().st_size
    report["checkpoint_contains_pretrained_base_tensors"] = False
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", default="white_stucco_02_auto_001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, default=ROOT / "out/material-training/frozen-dino-stucco-01/frozen_features/checkpoint.final.pt")
    parser.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
    parser.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--expected-size", type=int, default=1024)
    parser.add_argument("--encoder-size", type=int, default=518)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-driver-bytes", type=int, default=None, help="Memory budget in bytes; default adapts to physical/Metal memory with OS headroom")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--mask-transparent-input", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"event": "complete", "summary": str(args.output / "summary.json"), "status": report["status"], "completed_updates": report["completed_updates"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
