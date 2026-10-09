#!/usr/bin/env python3
"""Research-only DINOv2 material experiments, outside the shipped backend."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnose_material_curriculum import cpu_tree, mean_metrics, save_checkpoint_atomic
from diagnose_material_fit import fit_metrics
from frozen_dino_height import ARCHITECTURE, CODE_REVISION, DIAGNOSTIC_SCHEMA, MODEL_REVISION, MODEL_SHA256, ConditionedHeightNet, extract_features, load_frozen_encoder, state_sha256
from material_dataset import read_png
from material_fixed_adapters import POLICY as FIXED_ADAPTER_POLICY, adapter_sha256, apply_fixed_adapters, encoder_size as fixed_encoder_size, validate_adapters
from material_height_model import squared_objective, weighted_mean
from material_resources import configure_training_resources, decoded_cache_budget, resolve_training_budget, training_resources
from train_material_height import checked_relative, choose_device, digest, find_samples, load_pair, memory, opengl_normal_from_height, write_float_exr, write_json

SCHEMA = "texture-studio-material-training-cycle-v1"
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("height", "roughness", "normal")
# Compatibility for scripts importing this name; the limit is machine-derived.
MAX_DRIVER_BYTES = training_resources()["maximum_training_bytes"]
ADAPTATION_SCHEMA = "texture-studio-four-material-adaptation-diagnostic-v1"



class MaterialMapHead(ConditionedHeightNet):
    """Independent scalar reflectance/height or directly supervised normal head."""
    def __init__(self, base_channels: int = 12, feature_channels: int = 768,
                 projection_channels: int = 12, target: str = "height") -> None:
        if target not in TARGETS:
            raise ValueError("Unknown material-map target")
        super().__init__(base_channels, feature_channels, projection_channels)
        self.target = target
        if target == "normal":
            self.head = nn.Conv2d(base_channels, 3, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            with torch.no_grad():
                self.head.bias[2] = 1

    def forward(self, rgb: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        result = super().forward(rgb, features)
        if self.target == "normal":
            # Unit tangent-space direction, encoded in [0,1]; independent of
            # any height derivative. Raw target codes remain untouched.
            result = F.normalize(result * 2 - 1, dim=1, eps=1e-6) * 0.5 + 0.5
        return result


def balanced_schedule(sample_count: int, updates: int, seed: int) -> list[int]:
    if sample_count < 1 or updates < 1:
        raise ValueError("Need training samples and positive updates per crop")
    rng, result = random.Random(seed), []
    for _ in range(updates):
        current = list(range(sample_count))
        rng.shuffle(current)
        result.extend(current)
    return result


def _encoder_contract(checkpoint: dict[str, Any]) -> None:
    encoder = checkpoint.get("encoder", {})
    if encoder.get("checkpoint_sha256") != MODEL_SHA256 or encoder.get("official_meta_code_revision") != CODE_REVISION:
        raise ValueError("Head checkpoint requires the pinned official DINOv2 Base encoder")


def load_cycle_head(path: Path, device: torch.device, expected_sha256: str | None = None,
                    transfer_target: str | None = None) -> tuple[MaterialMapHead, dict[str, Any]]:
    """Load a native head and preserve any declared, fixed encoder adapters."""
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Native material checkpoint exceeds its 64 MiB bound")
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError(f"Warm-start checkpoint checksum mismatch: {path}")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("schema") == "texture-studio-material-inference-package-v1":
        checkpoint = checkpoint.get("inference_checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Material package must contain a declared inference checkpoint")
    _encoder_contract(checkpoint)
    schema = checkpoint.get("schema")
    if schema == DIAGNOSTIC_SCHEMA:
        if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("variant") != "frozen_features":
            raise ValueError("Only a frozen-feature native head can warm start this cycle")
        config, state = checkpoint.get("model_config", {}), checkpoint.get("model_state", {})
        source_target = "height"
    elif schema in (SCHEMA, ADAPTATION_SCHEMA):
        if checkpoint.get("variant", "frozen") not in ("frozen", "frozen_features", "lora"):
            raise ValueError("Unknown checkpoint encoder variant")
        config, state = checkpoint.get("head_config", {}), checkpoint.get("head_state", {})
        source_target = checkpoint.get("target", config.get("target", "height"))
    else:
        raise ValueError("Unsupported native material-head checkpoint schema")
    config = dict(config)
    config.setdefault("target", source_target)
    if set(config) != {"base_channels", "feature_channels", "projection_channels", "target"} or config["feature_channels"] != 768 or config["target"] != source_target:
        raise ValueError("Checkpoint head configuration is incompatible")
    if (type(config["base_channels"]) is not int or not 4 <= config["base_channels"] <= 64 or config["base_channels"] % 4
            or type(config["projection_channels"]) is not int or not 1 <= config["projection_channels"] <= 64):
        raise ValueError("Checkpoint head configuration exceeds bounded native dimensions")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Head state must contain finite Float32 tensors")
    model = MaterialMapHead(**config)
    model.load_state_dict(state, strict=True)
    reset_output = transfer_target is not None and transfer_target != source_target
    if reset_output:
        replacement = MaterialMapHead(**dict(config, target=transfer_target))
        body = {key: value for key, value in state.items() if not key.startswith("head.")}
        replacement.load_state_dict(dict(replacement.state_dict(), **body), strict=True)
        model, config = replacement, dict(config, target=transfer_target)
    fixed_adapters = checkpoint.get("adapter_state", {})
    if checkpoint.get("variant") == "lora":
        validate_adapters(fixed_adapters)
        size = fixed_encoder_size(checkpoint, path)
    elif fixed_adapters:
        raise ValueError("Frozen checkpoint contains unexplained encoder adapters")
    else:
        size = checkpoint.get("encoder_size", 518)
    if schema == ADAPTATION_SCHEMA:
        size = fixed_encoder_size(checkpoint, path)
    if type(size) is not int or not 28 <= size <= 518 or size % 14:
        raise ValueError("Checkpoint requires a bounded aligned encoder size")
    model.fixed_adapter_state = cpu_tree(fixed_adapters)
    return model.to(device), {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": checksum,
        "source_schema": schema, "source_step": checkpoint.get("step"), "source_target": source_target,
        "head_config": config, "output_layer_reset_for_new_target": reset_output,
        "optimizer_reset": True, "source_encoder_size": size,
        "fixed_adapter_sha256": adapter_sha256(fixed_adapters) if fixed_adapters else None,
        "encoder_refinement_policy": FIXED_ADAPTER_POLICY if fixed_adapters else "refine_material_head_with_frozen_encoder"}


def load_target_pair(sample: dict[str, Any], device: torch.device, target: str,
                     mask_transparent_input: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    source, height, mask = load_pair(sample, device, mask_input_alpha=mask_transparent_input, return_mask=True)
    if target == "height":
        sample["target_sample_bits"] = 16
        return source, height, mask
    del height
    details = sample["metadata"].get("map_metadata", {}).get(target, {})
    filename = sample["metadata"].get("maps", {}).get(target)
    if not filename or details.get("encoding") != "linear_data":
        raise ValueError(f"Need an explicitly linear numeric {target} map: {sample['metadata_path']}")
    path = checked_relative(sample["metadata_path"].parent, filename)
    codes, metadata = read_png(path)
    if metadata["file_sha256"] != details.get("sample_sha256"):
        raise ValueError(f"Prepared map checksum mismatch ({target}): {path}")
    maximum = np.iinfo(codes.dtype).max
    # Auxiliary alpha can contain tiny integer rounding differences even in
    # published opaque maps. This is a declared numeric-data interpretation:
    # retain each scalar/RGB code independently and never premultiply it.
    # Eight UInt16 codes are 0.0001221 of opacity; UInt8 remains exactly opaque.
    tolerance = 8 if codes.dtype == np.uint16 else 0
    has_alpha = codes.shape[-1] == 4 or (target == "roughness" and codes.shape[-1] == 2)
    alpha = codes[..., -1] if has_alpha else None
    alpha_policy = {"policy": "independent_numeric_components_near_opaque_auxiliary_alpha_v1",
        "numeric_components_preserved": True, "alpha_used_to_scale_numeric_components": False,
        "original_png_preserved": True, "alpha_present": alpha is not None,
        "maximum_alpha_code": int(maximum), "accepted_alpha_deficit_codes": tolerance,
        "minimum_observed_alpha_code": None if alpha is None else int(alpha.min()),
        "nonopaque_alpha_pixel_fraction": 0.0 if alpha is None else float(np.mean(alpha < maximum)),
        "meaningful_transparency_rejected": True}
    if target == "roughness":
        policy = details.get("scalar_alpha_policy", {})
        independent_alpha = (policy.get("scalar_component") == "grayscale"
            and policy.get("alpha_preserved_in_png") is True
            and policy.get("alpha_used_to_scale_scalar") is False
            and policy.get("uint16_near_opaque_tolerance_codes") == 8)
        if codes.shape[-1] == 2:
            if not independent_alpha:
                raise ValueError("Grayscale+alpha roughness needs declared independent scalar/alpha preservation policy")
            if np.any(codes[..., 1] < maximum - tolerance):
                raise ValueError("Nonopaque roughness alpha must be reviewed")
        if codes.shape[-1] in (3, 4):
            if np.any(codes[..., :3] != codes[..., 0:1]):
                raise ValueError("Roughness RGB components differ; scalar conversion is not inferred")
            if codes.shape[-1] == 4 and np.any(codes[..., 3] < maximum - tolerance):
                raise ValueError("Roughness alpha exceeds the declared near-opaque auxiliary tolerance")
        elif codes.shape[-1] not in (1, 2):
            raise ValueError("Roughness must be scalar or identical opaque RGB")
        codes = codes[..., :1]
    else:
        if codes.shape[-1] not in (3, 4) or (codes.shape[-1] == 4 and np.any(codes[..., 3] < maximum - tolerance)):
            raise ValueError("Direct normal target needs RGB and alpha within the declared near-opaque auxiliary tolerance")
        original = details.get("source", {}).get("suffix", "")
        converted = any(operation.get("type") == "directx_to_opengl" for operation in details.get("transforms", []))
        if original not in ("nor_gl", "NormalGL") and not converted and details.get("normal_convention") != "OpenGL +Y":
            raise ValueError("Direct normal target needs an explicit OpenGL convention or recorded integer green inversion")
        codes = codes[..., :3]
    numeric = np.ascontiguousarray(codes.astype(np.float32) / np.float32(maximum))
    values = torch.from_numpy(numeric.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    if values.shape[-2:] != source.shape[-2:]:
        raise ValueError("Native input/target dimensions differ; targets are never resized")
    sample["target_sample_bits"] = metadata["sample_bits"]
    sample["target_auxiliary_alpha_policy"] = alpha_policy
    return source, values, mask


def map_objective(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, role: str) -> tuple[torch.Tensor, dict[str, float]]:
    if role == "height":
        return squared_objective(prediction, target, mask)
    absolute = weighted_mean((prediction - target).square().mean(dim=1, keepdim=True), mask)
    if role == "roughness":
        # Roughness has an absolute reflectance parameter: offset must matter.
        gradient_terms = []
        for axis in (-1, -2):
            error = torch.diff(prediction, dim=axis) - torch.diff(target, dim=axis)
            neighbor = None if mask is None else (mask[..., :, 1:] * mask[..., :, :-1] if axis == -1 else mask[..., 1:, :] * mask[..., :-1, :])
            gradient_terms.append(weighted_mean(error.square(), neighbor))
        gradient = torch.stack(gradient_terms).mean()
        total = absolute + 4 * gradient
        return total, {"absolute_roughness_mse": float(absolute.detach()), "native_gradient_mse": float(gradient.detach())}
    target_squared_length = (target * 2 - 1).square().sum(dim=1, keepdim=True)
    # Interpolation in published normal maps can shorten valid direction
    # vectors. A length below 0.5 is not evidence of an invalid direction.
    # Normalize directions for angular supervision only; raw encoded target
    # samples remain unchanged for the independent component MSE term.
    if bool((target_squared_length <= 1e-12).any()):
        raise ValueError("Normal target contains invalid near-zero tangent vectors")
    pvector, tvector = F.normalize(prediction * 2 - 1, dim=1, eps=1e-6), F.normalize(target * 2 - 1, dim=1, eps=1e-6)
    angular = weighted_mean(1 - (pvector * tvector).sum(dim=1, keepdim=True).clamp(-1, 1), mask)
    return angular + absolute, {"angular_cosine_loss": float(angular.detach()), "encoded_normal_mse": float(absolute.detach()),
        "target_vector_minimum_length": float(target_squared_length.min().sqrt().detach()),
        "target_vector_fraction_shorter_than_0_5": float((target_squared_length < 0.25).float().mean().detach()),
        "angular_term_normalizes_direction_only": True}


@torch.no_grad()
def map_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, role: str) -> dict[str, Any]:
    if role == "height":
        result = fit_metrics(prediction, target, mask)
    else:
        result = {"mae": float(weighted_mean((prediction - target).abs().mean(dim=1, keepdim=True), mask))}
    loss, components = map_objective(prediction, target, mask, role)
    result.update(components)
    result["objective"] = float(loss)
    return result


class CycleLimit(RuntimeError):
    pass


class NativePairCache:
    """Reuse unchanged decoded Float32 pairs, with bounded CPU-only storage.

    Samples enter lazily in their existing feature/training order. This changes
    neither optimizer batching nor the saved schedule, and never resizes maps.
    """
    def __init__(self, loader: Callable, maximum_bytes: int) -> None:
        self.loader, self.maximum_bytes = loader, maximum_bytes
        self.pairs: OrderedDict[str, tuple] = OrderedDict()
        self.bytes, self.hits, self.misses, self.pressure_evictions = 0, 0, 0, 0

    @staticmethod
    def tensor_bytes(pair: tuple) -> int:
        return sum(value.numel() * value.element_size() for value in pair if value is not None)

    def get(self, sample: dict, device: torch.device) -> tuple:
        identity = sample["metadata"]["sample_id"]
        pair = self.pairs.pop(identity, None)
        if pair is None:
            self.misses += 1
            pair = self.loader(sample)
            size = self.tensor_bytes(pair)
            if size <= self.maximum_bytes:
                while self.pairs and self.bytes + size > self.maximum_bytes:
                    _, evicted = self.pairs.popitem(last=False)
                    self.bytes -= self.tensor_bytes(evicted)
                self.pairs[identity] = pair
                self.bytes += size
        else:
            self.hits += 1
            self.pairs[identity] = pair
        return tuple(None if value is None else value.to(device) for value in pair)

    def report(self) -> dict:
        return {"device": "cpu", "maximum_bytes": self.maximum_bytes,
                "retained_bytes": self.bytes, "retained_pairs": len(self.pairs),
                "hits": self.hits, "decodes": self.misses, "pressure_evictions": self.pressure_evictions,
                "native_values_unchanged": True,
                "prefilled": False, "eviction": "least_recently_used"}

    def reclaim(self, bytes_needed: int) -> None:
        target = max(0, self.bytes - bytes_needed)
        while self.pairs and self.bytes > target:
            _, pair = self.pairs.popitem(last=False)
            self.bytes -= self.tensor_bytes(pair)
            self.pressure_evictions += 1


class Guard:
    def __init__(self, device: torch.device, minutes: float, driver_bytes: int) -> None:
        self.device, self.minutes, self.driver_bytes = device, minutes, driver_bytes
        self.started, self.peak, self.cancelled = time.monotonic(), {}, False
        self.retained_cpu_bytes: Callable[[], int] = lambda: 0
        self.reclaim_cpu_bytes: Callable[[int], None] = lambda _bytes: None

    def check(self, stage: str) -> None:
        current = memory(self.device)
        retained = self.retained_cpu_bytes()
        excess = current.get("mps_driver_bytes", 0) + retained - self.driver_bytes
        if excess > 0:
            self.reclaim_cpu_bytes(excess)
            retained = self.retained_cpu_bytes()
        current["retained_cpu_cache_bytes"] = retained
        # These CPU tensors are deliberately not resident on the MPS device,
        # so unlike RSS they do not overlap Metal driver accounting.
        current["tracked_unified_bytes"] = current.get("mps_driver_bytes", 0) + retained
        for key, value in current.items():
            if isinstance(value, int):
                self.peak[key] = max(self.peak.get(key, 0), value)
        if current["tracked_unified_bytes"] > self.driver_bytes:
            raise CycleLimit("memory soft guard at " + stage)
        if time.monotonic() - self.started >= self.minutes * 60:
            raise CycleLimit("time soft guard at " + stage)
        if self.cancelled:
            raise CycleLimit("cancelled at " + stage)


def select_samples(args: argparse.Namespace) -> tuple[list, list, dict, dict]:
    # Unselected materials' review state does not authorize or reject the
    # requested selection. Scientific dataset/split checks still run globally.
    found = find_samples(args.dataset, True)
    requested = sorted(set(args.materials or [s["metadata"]["material_id"] for s in found]))
    selected = sorted((s for s in found if s["metadata"]["material_id"] in requested), key=lambda s: s["metadata"]["sample_id"])
    if not args.allow_unreviewed and any(s["metadata"]["split"] == "train" and s["metadata"]["status"] not in ("approved", "accepted") for s in selected):
        raise ValueError("Selected samples need quality review or explicit --allow-unreviewed")
    training, validation = ([s for s in selected if s["metadata"]["split"] == split] for split in ("train", "validation"))
    if {s["metadata"]["material_id"] for s in selected} != set(requested) or not training or not validation:
        raise ValueError("Requested materials need eligible native training and validation crops")
    bindings, manifests = {}, {}
    for sample in selected:
        raw = sample["metadata_path"].read_bytes()
        if json.loads(raw) != sample["metadata"]:
            raise ValueError("Sample manifest changed during selection")
        identity = sample["metadata"]["sample_id"]
        manifests[identity] = raw
        bindings[str(sample["metadata_path"].resolve())] = hashlib.sha256(raw).hexdigest()
        for role in {"input", "height", args.target}:
            filename = sample["metadata"]["maps"].get(role)
            if not filename:
                raise ValueError(f"Missing {role} map: {identity}")
            bindings[str(checked_relative(sample["metadata_path"].parent, filename).resolve())] = sample["metadata"]["map_metadata"][role]["sample_sha256"]
        source, target, mask = load_target_pair(sample, torch.device("cpu"), args.target, args.mask_transparent_input)
        if tuple(source.shape[-2:]) != (args.expected_size, args.expected_size):
            raise ValueError(f"Expected exact native {args.expected_size}², never resized: {identity}")
        del source, target, mask
    return training, validation, bindings, manifests


def verify_bindings(bindings: dict[str, str]) -> None:
    for name, checksum in bindings.items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Selected file changed: {name}")


def cycle_identity(args: argparse.Namespace, training: list, validation: list, bindings: dict) -> dict[str, Any]:
    identity = {"target": args.target, "expected_size": args.expected_size, "encoder_size": args.encoder_size,
        "training_sample_ids": [s["metadata"]["sample_id"] for s in training], "validation_sample_ids": [s["metadata"]["sample_id"] for s in validation],
        "selected_files_sha256": bindings, "seed": args.seed, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "mask_transparent_input": args.mask_transparent_input,
        "objective": {"height": "relative_squared", "roughness": "absolute_mse_plus_4_native_gradient_mse", "normal": "direct_opengl_angular_plus_encoded_mse"}[args.target]}
    if args.target != "height":
        identity["target_auxiliary_alpha_policy"] = {sample["metadata"]["sample_id"]: sample["target_auxiliary_alpha_policy"] for sample in training + validation}
    return identity


def validate_resume(checkpoint: dict, identity: dict, schedule: list[int], config: dict) -> None:
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("identity") != identity or checkpoint.get("head_config") != config:
        raise ValueError("Resume identity differs: selected maps, native size, target, encoder, architecture or optimizer settings")
    old_schedule = checkpoint.get("schedule")
    step = checkpoint.get("step")
    if not isinstance(old_schedule, list) or not isinstance(step, int) or step < 0 or step > len(old_schedule) or len(schedule) < len(old_schedule) or schedule[:len(old_schedule)] != old_schedule:
        raise ValueError("Resume requires the original balanced schedule, optionally explicitly extended")
    expected = [Counter(schedule[:step])[index] for index in range(len(identity["training_sample_ids"]))]
    if checkpoint.get("per_crop_update_counts") != expected or not isinstance(checkpoint.get("optimizer_state"), dict):
        raise ValueError("Resume checkpoint has inconsistent counts or no optimizer state")


def export_map(directory: Path, prediction: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
               role: str, sample: dict, checkpoint_step: int, write_npy: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    value = prediction.detach().cpu().numpy()[0].transpose(1, 2, 0)
    if role != "normal":
        value = value[..., 0]
    if not np.isfinite(value).all():
        raise ValueError("Cannot export nonfinite material map")
    write_float_exr(directory / f"selected.{role}.float32.exr", value)
    if write_npy:
        np.save(directory / f"selected.{role}.float32.npy", value, allow_pickle=False)
    if role == "height":
        normal = opengl_normal_from_height(value)
        write_float_exr(directory / "selected.normal_opengl.float32.exr", normal)
        if write_npy:
            np.save(directory / "selected.normal_opengl.float32.npy", normal, allow_pickle=False)
    target_path = checked_relative(sample["metadata_path"].parent, sample["metadata"]["maps"][role])
    write_json(directory / "cycle-export.json", {"target": role, "checkpoint_step": checkpoint_step,
        "sample_id": sample["metadata"]["sample_id"], "native_dimensions": list(prediction.shape[-2:]),
        "input_png_path": str(sample["input_path"].resolve()), "input_png_sha256": sample["metadata"]["map_metadata"]["input"]["sample_sha256"],
        "input_encoding": sample["metadata"]["map_metadata"]["input"]["encoding"],
        "target_png_path": str(target_path.resolve()), "target_png_sha256": sample["metadata"]["map_metadata"][role]["sample_sha256"],
        "float_npy_duplicates_written": write_npy,
        **({"target_auxiliary_alpha_policy": sample["target_auxiliary_alpha_policy"]} if role != "height" else {}),
        "target_source_bits": sample["target_sample_bits"], "storage": "FLOAT32 EXR lossless ZIP exact round-trip verified", "blender_color_space": "Non-Color",
        "normal_origin": "Mathematical OpenGL derivative of predicted height, artistic 0.03 height/patch-width amplitude" if role == "height" else "Directly supervised OpenGL normal map; not height-derived" if role == "normal" else None,
        "precision_note": "Float export does not add information to the original integer-bit-depth training target"})


def run_train(args: argparse.Namespace, encoder_loader: Callable = load_frozen_encoder,
              head_loader: Callable = load_cycle_head, feature_extractor: Callable = extract_features) -> dict:
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Unset PYTORCH_ENABLE_MPS_FALLBACK; native MPS training never falls back to CPU")
    if not 16 <= args.expected_size <= 2048 or not 28 <= args.encoder_size <= 518 or args.encoder_size % 14 or not 0 < args.max_minutes <= 240 or args.learning_rate <= 0 or args.weight_decay < 0 or min(args.checkpoint_every, args.evaluate_every) < 1 or args.prediction_limit < 0:
        raise ValueError("Use native size ≤2048, aligned encoder ≤518, positive bounded time/memory and training intervals")
    if args.warm_start is not None and args.resume_checkpoint is not None:
        raise ValueError("Choose a fresh warm start or optimizer resume, not both")
    device = choose_device(args.device)
    args.max_driver_bytes, resource_limits = resolve_training_budget(args.max_driver_bytes, device)
    resource_limits.update(configure_training_resources(args.max_driver_bytes, device))
    training, validation, bindings, manifests = select_samples(args)
    schedule = balanced_schedule(len(training), args.updates_per_crop, args.seed)
    warm = args.resume_checkpoint or args.warm_start
    if warm is None:
        torch.manual_seed(args.seed)
        head = MaterialMapHead(args.base_channels, 768, args.projection_channels, args.target).to(device)
        head_info = {"head_config": {"base_channels": args.base_channels, "feature_channels": 768, "projection_channels": args.projection_channels, "target": args.target}, "optimizer_reset": True, "from_scratch": True}
    else:
        head, head_info = head_loader(warm, device, args.warm_start_sha256, args.target)
        args.encoder_size = head_info["source_encoder_size"]
        if type(args.encoder_size) is not int or not 28 <= args.encoder_size <= 518 or args.encoder_size % 14:
            raise ValueError("Selected checkpoint requires a bounded aligned encoder size")
        head_info["encoder_size_policy"] = "selected_checkpoint_size_preserved"
    fixed_adapters = getattr(head, "fixed_adapter_state", {})
    fixed_hash = adapter_sha256(fixed_adapters) if fixed_adapters else None
    identity = cycle_identity(args, training, validation, bindings)
    if fixed_adapters:
        identity["fixed_encoder_adapters_sha256"] = fixed_hash
    config = head_info["head_config"]
    initial = state_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    step, counts, history, best_score, best_step, best_state = 0, [0] * len(training), [], None, None, None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        validate_resume(checkpoint, identity, schedule, config)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step, counts = checkpoint["step"], list(checkpoint["per_crop_update_counts"])
        history = list(checkpoint.get("history", []))
        best_score, best_step, best_state = checkpoint.get("best_validation_score"), checkpoint.get("best_validation_step"), checkpoint.get("best_head_state")
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        head_info["optimizer_reset"] = False
    else:
        torch.manual_seed(args.seed)
    verify_bindings(bindings)
    if args.output.exists():
        raise ValueError("Choose a new output directory; previous cycles are preserved")
    args.output.mkdir(parents=True)
    snapshot = args.output / "dataset-snapshot"
    snapshot.mkdir()
    for name, payload in manifests.items():
        (snapshot / (name + ".json")).write_bytes(payload)
    implementation = args.output / "implementation"
    implementation.mkdir()
    code_hashes = {}
    for name in ("material_training_cycle.py", "material_resources.py", "material_fixed_adapters.py", "probe_material_adapters.py", "frozen_dino_height.py", "material_height_model.py", "train_material_height.py", "material_dataset.py", "diagnose_material_fit.py", "diagnose_material_curriculum.py"):
        path = Path(__file__).parent / name
        (implementation / name).write_bytes(path.read_bytes())
        code_hashes[name] = digest(path)
    guard = Guard(device, args.max_minutes, args.max_driver_bytes)
    pairs = NativePairCache(lambda sample: load_target_pair(sample, torch.device("cpu"), args.target, args.mask_transparent_input),
                            decoded_cache_budget(args.max_driver_bytes))
    guard.retained_cpu_bytes = lambda: pairs.bytes + sum(f.numel() * f.element_size() for f in features.values())
    guard.reclaim_cpu_bytes = pairs.reclaim
    report = {"schema": SCHEMA, "identity": identity, "head_config": config, "warm_start": head_info,
        "native_dimensions": [args.expected_size] * 2, "target": args.target, "target_resized": False,
        "target_precision_bits": {s["metadata"]["sample_id"]: s["target_sample_bits"] for s in training + validation},
        "validation_scope": sorted({s["validation_scope"] for s in validation}), "fresh_material_generalization_tested": all(s["validation_scope"] == "unseen materials" for s in validation),
        "selection_policy": args.selection, "selection_note": "Final is user default; metrics are diagnostic. Explicit best-validation selection uses actual evaluated initial/final/periodic snapshots only.",
        "initial_head_state_sha256": initial, "requested_steps": len(schedule), "requested_updates_per_crop": args.updates_per_crop,
        "encoder_resize_only": True, "no_augmentation": True, "source_images_modified": False, "production_promotion": False,
        "encoder_refinement_policy": FIXED_ADAPTER_POLICY if fixed_adapters else "refine_material_head_with_frozen_encoder",
        "fixed_adapter_sha256": fixed_hash, "encoder_adapters_trained": False,
        "selected_crop_rectangles": {s["metadata"]["sample_id"]: s["metadata"]["crop_rectangle_top_left_xywh"] for s in training + validation},
        "implementation_sha256": code_hashes, "dataset_index_sha256_at_selection": digest(args.dataset / "dataset.json"), "device": str(device), "torch_version": str(torch.__version__),
        "started_from_step": step, "feature_cache": "Detached frozen encoder grids on CPU; bounded lazy native-pair CPU cache reuses exact Float32 values",
        "guard": {"max_minutes": args.max_minutes, "max_driver_bytes": args.max_driver_bytes, "note": "Sampled guards include retained CPU caches plus Metal driver allocations; allocator also enforces the GPU limit"}, "resources": resource_limits, "prediction_limit": args.prediction_limit,
        "export_split": args.export_split, "float_npy_duplicates_written": args.write_npy}
    write_json(args.output / "run.json", report)
    encoder_info, features = {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION}, {}
    status, stop_reason = "complete", None
    def payload() -> dict:
        return {"schema": SCHEMA, "head_config": config, "head_state": cpu_tree(head.state_dict()), "target": args.target,
            "variant": "lora" if fixed_adapters else "frozen", "adapter_state": cpu_tree(fixed_adapters),
            "encoder_refinement_policy": FIXED_ADAPTER_POLICY if fixed_adapters else "refine_material_head_with_frozen_encoder",
            "encoder": encoder_info, "encoder_size": args.encoder_size, "step": step, "identity": identity,
            "training_sample_ids": identity["training_sample_ids"], "validation_sample_ids": identity["validation_sample_ids"],
            "schedule": schedule, "per_crop_update_counts": counts, "optimizer_state": cpu_tree(optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state(), "complete_schedule": step == len(schedule), "history": history,
            "best_validation_score": best_score, "best_validation_step": best_step, "best_head_state": cpu_tree(best_state),
            "run_settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "implementation_sha256": code_hashes, "production_promotion": False}
    previous_handlers = {kind: signal.signal(kind, lambda _sig, _frame: setattr(guard, "cancelled", True)) for kind in (signal.SIGINT, signal.SIGTERM)}
    @torch.no_grad()
    def evaluate() -> dict:
        head.eval()
        result = {"step": step, "per_crop_update_counts": list(counts)}
        try:
            for name, samples in (("training_fit", training), ("validation", validation)):
                records = []
                for sample in samples:
                    guard.check("evaluation before forward")
                    x, y, mask = pairs.get(sample, device)
                    prediction = head(x, features[sample["metadata"]["sample_id"]].to(device))
                    guard.check("evaluation after forward")
                    records.append({"material_id": sample["metadata"]["material_id"], "sample_id": sample["metadata"]["sample_id"], "metrics": map_metrics(prediction, y, mask, args.target), "loss_valid_pixel_fraction": sample["loss_valid_pixel_fraction"]})
                    del x, y, mask, prediction
                result[name] = {"samples": records, "mean_metrics": mean_metrics(records), "sample_count": len(records)}
        finally:
            head.train()
        return result
    def record_evaluation() -> None:
        nonlocal best_score, best_step, best_state
        entry = evaluate()
        score = entry["validation"]["mean_metrics"]["objective"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation diagnostic")
        if best_score is None or score < best_score:
            best_score, best_step, best_state = score, step, cpu_tree(head.state_dict())
        history.append(entry)
        write_json(args.output / "progress.json", {"history": history})
        print(json.dumps({"event": "evaluation", **entry}), flush=True)
    try:
        guard.check("before encoder load")
        encoder, encoder_info = encoder_loader(args.model_directory, args.code_directory, device)
        if fixed_adapters:
            from probe_material_adapters import base_fingerprint
            frozen_base_before = base_fingerprint(encoder)
        fixed_modules = apply_fixed_adapters(encoder, fixed_adapters) if fixed_adapters else {}
        guard.check("after encoder load")
        for sample in training + validation:
            guard.check("feature cache before forward")
            x, y, mask = pairs.get(sample, device)
            features[sample["metadata"]["sample_id"]] = feature_extractor(encoder, x, args.encoder_size).cpu().detach()
            guard.check("feature cache after forward")
            del x, y, mask
        if any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
            raise ValueError("Frozen encoder acquired gradients")
        if fixed_modules:
            actual = {name: {field: getattr(module, field).detach().cpu() for field in ("lora_A", "lora_B")} for name, module in fixed_modules.items()}
            if adapter_sha256(actual) != fixed_hash:
                raise ValueError("Fixed encoder adapters changed during feature caching")
            report["fixed_adapters_verified_unchanged"] = True
            if base_fingerprint(encoder) != frozen_base_before:
                raise ValueError("Original pretrained encoder changed during fixed-adapter feature caching")
            report["pretrained_base_verified_unchanged"] = True
            del actual
        del fixed_modules
        del encoder
        if device.type == "mps":
            torch.mps.empty_cache()
        report["feature_cache_bytes"] = sum(f.numel() * f.element_size() for f in features.values())
        report["encoder"] = encoder_info
        record_evaluation()
        save_checkpoint_atomic(args.output / "checkpoint.latest.pt", payload())
        while step < len(schedule):
            guard.check("training before forward")
            crop = schedule[step]
            sample = training[crop]
            x, y, mask = pairs.get(sample, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(x, features[sample["metadata"]["sample_id"]].to(device))
            loss, components = map_objective(prediction, y, mask, args.target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised objective")
            guard.check("after forward before backward")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
            guard.check("after backward before optimizer")
            optimizer.step()
            step += 1
            counts[crop] += 1
            if step % args.checkpoint_every == 0:
                save_checkpoint_atomic(args.output / "checkpoint.latest.pt", payload())
                print(json.dumps({"event": "training", "step": step, "loss": float(loss.detach()), "components": components, "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - guard.started}), flush=True)
            del x, y, mask, prediction, loss
            guard.check("after optimizer")
            if step % args.evaluate_every == 0 or step == len(schedule):
                record_evaluation()
                save_checkpoint_atomic(args.output / "checkpoint.latest.pt", payload())
    except CycleLimit as limit:
        status, stop_reason = "stopped", str(limit)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        status, stop_reason = "stopped", "actual runtime allocation failure: " + str(error)
    finally:
        for kind, handler in previous_handlers.items():
            signal.signal(kind, handler)
    # A guard/cancelled run retains every completed update and optimizer.
    save_checkpoint_atomic(args.output / "checkpoint.latest.pt", payload())
    save_checkpoint_atomic(args.output / "checkpoint.final.pt", dict(payload(), optimizer_state=None, best_head_state=None))
    selected_step = step
    if args.selection == "best-validation":
        if best_state is None:
            selected_step = None
        else:
            head.load_state_dict(best_state, strict=True)
            selected_step = best_step
    if selected_step is not None:
        selected = dict(payload(), head_state=cpu_tree(head.state_dict()), step=selected_step,
            per_crop_update_counts=[Counter(schedule[:selected_step])[i] for i in range(len(training))],
            optimizer_state=None, best_head_state=None, complete_schedule=selected_step == len(schedule), selection_policy=args.selection)
        save_checkpoint_atomic(args.output / "checkpoint.selected.pt", selected)
    exported, export_stop = [], None
    try:
        with torch.no_grad():
            # Pair one training and one validation region first for useful
            # visual comparison, then bound any additional exports explicitly.
            if args.export_split == "paired":
                ordered = [item for pair in zip(training, validation) for item in pair]
                ordered += [item for item in training + validation if item not in ordered]
            else:
                ordered = validation if args.export_split == "validation" else training if args.export_split == "train" else training + validation
            available = [sample for sample in ordered if sample["metadata"]["sample_id"] in features]
            for sample in available[:args.prediction_limit] if selected_step is not None else []:
                guard.check("export before forward")
                x, y, mask = pairs.get(sample, device)
                prediction = head(x, features[sample["metadata"]["sample_id"]].to(device))
                guard.check("export after forward")
                directory = args.output / "predictions" / sample["metadata"]["sample_id"]
                export_map(directory, prediction, x, y, args.target, sample, selected_step, args.write_npy)
                if mask is not None:
                    np.save(directory / "loss-valid.bool.npy", mask.cpu().numpy()[0, 0].astype(bool), allow_pickle=False)
                exported.append(sample["metadata"]["sample_id"])
                del x, y, mask, prediction
    except CycleLimit as limit:
        export_stop = str(limit)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        export_stop = "actual export allocation failure: " + str(error)
    verify_bindings(bindings)
    report.update({"status": status, "stop_reason": stop_reason, "completed_steps": step,
        "complete_schedule": step == len(schedule), "per_crop_update_counts": dict(zip(identity["training_sample_ids"], counts)),
        "history": history, "selected_step": selected_step, "best_validation_step": best_step, "best_validation_objective": best_score,
        "exported_sample_ids": exported, "export_stop_reason": export_stop, "sampled_peak_memory": guard.peak,
        "decoded_pair_cache": pairs.report(),
        "elapsed_seconds": time.monotonic() - guard.started, "source_files_verified_unchanged": True,
        "dataset_index_changed_during_run": digest(args.dataset / "dataset.json") != report["dataset_index_sha256_at_selection"],
        "selection_checkpoint_sha256": digest(args.output / "checkpoint.selected.pt") if selected_step is not None else None})
    write_json(args.output / "summary.json", report)
    return report


def run_probe(args: argparse.Namespace, encoder_loader: Callable = load_frozen_encoder,
              feature_extractor: Callable = extract_features,
              native_sizes: tuple[int, int] = (1024, 2048)) -> dict:
    """Actual optimizer updates at both prepared native sizes; quality untested."""
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("MPS probe refuses CPU fallback")
    if not 1 <= args.steps <= 5 or not 0 < args.max_minutes <= 2 or not 28 <= args.encoder_size <= 518 or args.encoder_size % 14 or args.learning_rate <= 0:
        raise ValueError("Probe permits 1–5 real updates per native size and at most 2 minutes")
    device = choose_device(args.device)
    args.max_driver_bytes, resource_limits = resolve_training_budget(args.max_driver_bytes, device)
    resource_limits.update(configure_training_resources(args.max_driver_bytes, device))
    # Both native sample contracts are checked before any encoder allocation.
    selected, bindings = [], {}
    for size, dataset in zip(native_sizes, (args.dataset_1024, args.dataset_2048)):
        candidates = [s for s in find_samples(dataset, args.allow_unreviewed) if s["metadata"]["split"] == "train" and s["metadata"]["material_id"] == args.material]
        if not candidates:
            raise ValueError(f"Missing native training sample at {size}")
        sample = sorted(candidates, key=lambda s: s["metadata"]["sample_id"])[0]
        x, y, mask = load_target_pair(sample, torch.device("cpu"), "height", args.mask_transparent_input)
        if tuple(x.shape[-2:]) != (size, size):
            raise ValueError("Probe requires independently prepared native crops, never resized targets")
        raw = sample["metadata_path"].read_bytes()
        if json.loads(raw) != sample["metadata"]:
            raise ValueError("Selected probe manifest changed")
        bindings[str(sample["metadata_path"].resolve())] = hashlib.sha256(raw).hexdigest()
        for role in ("input", "height"):
            bindings[str(sample[role + "_path"].resolve())] = sample["metadata"]["map_metadata"][role]["sample_sha256"]
        selected.append((size, sample))
        del x, y, mask
    bindings[str(args.warm_start.resolve())] = digest(args.warm_start)
    verify_bindings(bindings)
    template, head_info = load_cycle_head(args.warm_start, torch.device("cpu"), args.warm_start_sha256, "height")
    args.encoder_size = head_info["source_encoder_size"]
    initial = cpu_tree(template.state_dict())
    report = {"schema": SCHEMA + "-size-probe", "scope": "Native update timing/memory feasibility only; no quality or promotion", "warm_start": head_info, "resources": resource_limits, "sizes": {}, "selected_files_sha256": bindings, "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in ("material_training_cycle.py", "material_resources.py", "frozen_dino_height.py", "material_height_model.py", "train_material_height.py")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run.json", report)
    setup_guard = Guard(device, args.max_minutes, args.max_driver_bytes)
    try:
        setup_guard.check("probe before encoder load")
        encoder, pins = encoder_loader(args.model_directory, args.code_directory, device)
        fixed_adapters = getattr(template, "fixed_adapter_state", {})
        if fixed_adapters:
            args.encoder_size = head_info["source_encoder_size"]
            apply_fixed_adapters(encoder, fixed_adapters)
        setup_guard.check("probe after encoder load")
        report["encoder"] = pins
    except (CycleLimit, RuntimeError) as error:
        if isinstance(error, RuntimeError) and not isinstance(error, CycleLimit) and "out of memory" not in str(error).lower():
            raise
        report.update(status="stopped during setup", stop_reason=str(error), sampled_setup_peak_memory=setup_guard.peak)
        verify_bindings(bindings)
        report["source_files_verified_unchanged"] = True
        write_json(args.output / "summary.json", report)
        return report
    for size, sample in selected:
        x, y, mask = load_target_pair(sample, device, "height", args.mask_transparent_input)
        if tuple(x.shape[-2:]) != (size, size):
            raise ValueError("Probe requires independently prepared native crops, never resized targets")
        try:
            setup_guard.check("probe feature cache before forward")
            features = feature_extractor(encoder, x, args.encoder_size)
            setup_guard.check("probe feature cache after forward")
        except (CycleLimit, RuntimeError) as error:
            if isinstance(error, RuntimeError) and not isinstance(error, CycleLimit) and "out of memory" not in str(error).lower():
                raise
            report["sizes"][str(size)] = {"status": "stopped during feature setup", "stop_reason": str(error), "steps": [], "mean_measured_seconds": None, "sampled_peak_memory": setup_guard.peak}
            break
        head = MaterialMapHead(**head_info["head_config"]).to(device)
        head.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        guard = Guard(device, args.max_minutes, args.max_driver_bytes)
        records, status = [], "complete"
        try:
            # First actual optimizer update warms kernels/state; subsequent
            # synchronized timings remain separately visible, not averaged away.
            for index in range(args.steps + 1):
                guard.check("probe before forward")
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                prediction = head(x, features)
                loss, _ = squared_objective(prediction, y, mask)
                guard.check("probe after forward")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
                guard.check("probe after backward")
                before = head.head.weight.detach().clone()
                optimizer.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                records.append({"warmup": index == 0, "seconds": time.monotonic() - started, "loss": float(loss.detach()), "finite_gradient_norm": float(norm), "head_parameters_changed": bool(torch.any(before != head.head.weight)), "native_output_dimensions": list(prediction.shape[-2:])})
                del prediction, loss, before
                guard.check("probe after optimizer")
        except CycleLimit as limit:
            status = str(limit)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            status = "actual runtime allocation failure: " + str(error)
        report["sizes"][str(size)] = {"status": status, "steps": records, "mean_measured_seconds": float(np.mean([r["seconds"] for r in records if not r["warmup"]])) if len(records) > 1 else None, "sample_id": sample["metadata"]["sample_id"], "sample_manifest_sha256": digest(sample["metadata_path"]), "map_sha256": {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")}, "source_crop_rectangle": sample["metadata"]["crop_rectangle_top_left_xywh"], "sampled_peak_memory": guard.peak}
        write_json(args.output / "summary.json", report)
        del head, optimizer, features, x, y, mask
        if device.type == "mps":
            torch.mps.empty_cache()
        if status != "complete":
            break
    verify_bindings(bindings)
    report["source_files_verified_unchanged"] = True
    write_json(args.output / "summary.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    for command in ("train", "resume", "probe"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", choices=("mps", "cpu"), default="mps")
        p.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
        p.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
        p.add_argument("--encoder-size", type=int, default=518)
        p.add_argument("--max-minutes", type=float, default=2 if command == "probe" else 15)
        p.add_argument("--max-driver-bytes", type=int, default=None, help="Training memory in bytes; default adapts to physical/Metal memory, with OS headroom")
        p.add_argument("--allow-unreviewed", action="store_true")
        p.add_argument("--mask-transparent-input", action="store_true")
        p.add_argument("--learning-rate", type=float, default=0.001)
        p.add_argument("--warm-start-sha256")
        if command == "probe":
            p.add_argument("--dataset-1024", type=Path, required=True)
            p.add_argument("--dataset-2048", type=Path, required=True)
            p.add_argument("--material", default="white_stucco_02")
            p.add_argument("--warm-start", type=Path, required=True)
            p.add_argument("--steps", type=int, default=3)
            continue
        p.add_argument("--dataset", type=Path, required=command == "train")
        p.add_argument("--material", dest="materials", action="append")
        p.add_argument("--target", choices=TARGETS, default="height")
        p.add_argument("--expected-size", type=int, choices=(1024, 2048), default=1024)
        p.add_argument("--updates-per-crop", type=int, default=100)
        p.add_argument("--seed", type=int, default=2307)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--base-channels", type=int, default=12)
        p.add_argument("--projection-channels", type=int, default=12)
        p.add_argument("--checkpoint-every", type=int, default=100)
        p.add_argument("--evaluate-every", type=int, default=400)
        p.add_argument("--prediction-limit", type=int, default=2, help="Bounded selected-checkpoint exports; 0 = metrics/checkpoints only")
        p.add_argument("--export-split", choices=("validation", "train", "paired", "all"), default="validation")
        p.add_argument("--write-npy", action="store_true", help="Also write redundant Float32 NPY diagnostics; default exports only lossless FLOAT EXR")
        p.add_argument("--selection", choices=("final", "best-validation"), default="final")
        p.add_argument("--warm-start", type=Path)
        p.add_argument("--resume-checkpoint", type=Path, required=command == "resume")
    return main


def main() -> None:
    arguments = parser()
    args = arguments.parse_args()
    if args.command == "resume":
        # Resume inherits identity settings unless a flag was explicitly given.
        stored = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        explicit_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        for key, value in stored.get("run_settings", {}).items():
            flag = "--" + key.replace("_", "-")
            if key in ("command", "output", "resume_checkpoint", "warm_start", "warm_start_sha256") or flag in explicit_flags or (key == "materials" and "--material" in explicit_flags):
                continue
            if key in ("dataset", "model_directory", "code_directory") and value is not None:
                value = Path(value)
            setattr(args, key, value)
        args.warm_start = None
    result = run_probe(args) if args.command == "probe" else run_train(args)
    print(json.dumps({"event": "complete", "output": str(args.output), "status": result.get("status"), "selected_step": result.get("selected_step")}), flush=True)


if __name__ == "__main__":
    main()
