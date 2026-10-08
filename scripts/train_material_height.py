#!/usr/bin/env python3
"""Train/evaluate a native-resolution material-height pilot on prepared PNGs.

Original maps are read only. Input diffuse transfer is explicit; numeric height
is raw integer codes divided by 65535, with no per-crop normalization or gamma.
The uint16 source remains on disk; only the active sample becomes Float32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from material_height_model import ARCHITECTURE, SCHEMA, MaterialHeightNet, detail_mask, height_loss, multiscale_gradient_loss, squared_objective, weighted_center, weighted_mean


def read_native_png(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    # Shared precision-preserving decoder; importing lazily also permits --help
    # before the preparation dependency has been installed.
    from material_dataset import read_png
    return read_png(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_relative(parent: Path, relative: str) -> Path:
    value = (parent / relative).resolve()
    if not value.is_relative_to(parent.resolve()):
        raise ValueError(f"Map path leaves its sample directory: {relative}")
    return value


def rectangles_overlap(first: list[int], second: list[int]) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def verify_region_split(samples: list[dict[str, Any]], index: dict[str, Any]) -> None:
    """Prove region validation never reuses training pixels from either map."""
    if index.get("validation_scope") != "unseen_regions_of_known_materials":
        raise ValueError("Held-out region policy must explicitly declare its validation scope")
    sources: dict[str, list[tuple[str, str, list[int]]]] = {}
    by_material: dict[str, set[str]] = {}
    for sample in samples:
        metadata = sample["metadata"]
        identity, split = metadata["sample_id"], metadata["split"]
        if metadata.get("split_strategy") != "heldout-region-v1" or metadata.get("validation_scope") != "unseen_regions_of_known_materials":
            raise ValueError(f"Sample does not match explicit held-out region policy: {identity}")
        if metadata.get("source_region_role") != split:
            raise ValueError(f"Sample source region role differs from split: {identity}")
        rectangle = metadata.get("crop_rectangle_top_left_xywh")
        if not isinstance(rectangle, list) or len(rectangle) != 4 or any(type(value) is not int for value in rectangle):
            raise ValueError(f"Held-out region needs an integer source rectangle: {identity}")
        x, y, width, height = rectangle
        if x < 0 or y < 0 or width < 1 or height < 1 or metadata.get("sample_pixel_dimensions") != [width, height]:
            raise ValueError(f"Invalid source crop rectangle: {identity}")
        by_material.setdefault(metadata["material_id"], set()).add(split)
        for role in ("input", "height"):
            details = metadata.get("map_metadata", {}).get(role, {})
            source = details.get("source", {})
            source_hash = source.get("file_sha256")
            if not isinstance(source_hash, str) or len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
                raise ValueError(f"Held-out region needs verified parent identity ({role}): {identity}")
            if type(source.get("width")) is not int or type(source.get("height")) is not int or x + width > source["width"] or y + height > source["height"]:
                raise ValueError(f"Source rectangle exceeds parent dimensions ({role}): {identity}")
            for other_id, other_split, other_rectangle in sources.get(source_hash, []):
                if split != other_split and rectangles_overlap(rectangle, other_rectangle):
                    raise ValueError(f"Training/validation source regions overlap: {identity} and {other_id}")
            sources.setdefault(source_hash, []).append((identity, split, rectangle))
    policy = index.get("automatic_validation")
    if policy:
        from material_validation_policy import POLICY
        if policy.get("policy") != POLICY or policy.get("fraction") != 0.05:
            raise ValueError("Unknown automatic material-check policy")
        expected = set(policy.get("material_ids", []))
        if not expected or expected - {entry["material_id"] for entry in index["samples"]}:
            raise ValueError("Automatic check material identities are invalid")
        incomplete = [material for material, splits in by_material.items()
                      if splits != ({"train", "validation"} if material in expected else {"train"})]
    else:
        incomplete = [material for material, splits in by_material.items() if splits != {"train", "validation"}]
    if incomplete:
        raise ValueError(f"Region validation requires training and validation regions of every material: {incomplete}")


def find_samples(dataset: Path, allow_unreviewed: bool) -> list[dict[str, Any]]:
    index = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if index.get("schema_version") != 2 or not isinstance(index.get("samples"), list):
        raise ValueError("Training requires the prepared schema-2 dataset index")
    found = []
    seen_ids = set()
    for item in index["samples"]:
        if item.get("status") in ("excluded", "rejected"):
            continue
        metadata_path = checked_relative(dataset, item["path"]) / "sample.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("sample_id", "material_id", "split", "status"):
            if metadata.get(key) != item.get(key):
                raise ValueError(f"Dataset index/sample disagree on {key}: {metadata_path}")
        if metadata["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample identity in dataset index: {metadata['sample_id']}")
        seen_ids.add(metadata["sample_id"])
        maps = metadata.get("maps", {})
        # A starter file without real maps is never a training sample.
        if not maps.get("input") or not maps.get("height"):
            raise ValueError(f"Indexed sample is missing input/height maps: {metadata_path}")
        input_path = checked_relative(metadata_path.parent, maps["input"])
        height_path = checked_relative(metadata_path.parent, maps["height"])
        if not input_path.is_file() or not height_path.is_file():
            raise ValueError(f"Indexed sample's input/height files are missing: {metadata_path}")
        if not metadata.get("source_precision_verified") or not metadata.get("crop_values_verified"):
            raise ValueError(f"Unverified sample is not eligible for training: {metadata_path}")
        if metadata.get("status") not in ("approved", "accepted") and not allow_unreviewed:
            raise ValueError(f"Sample needs quality review or explicit --allow-unreviewed: {metadata_path}")
        split = metadata.get("split")
        if split not in ("train", "validation"):
            raise ValueError(f"Sample must have a train/validation assignment: {metadata_path}")
        found.append({"metadata": metadata, "metadata_path": metadata_path, "input_path": input_path, "height_path": height_path,
                      "validation_scope": "unseen regions of known materials" if index.get("split_strategy") == "heldout-region-v1" else "unseen materials"})
    if not found:
        raise ValueError("No verified prepared samples found")
    material_splits: dict[str, set[str]] = {}
    for sample in found:
        material_splits.setdefault(sample["metadata"]["material_id"], set()).add(sample["metadata"]["split"])
    leaked = [material for material, splits in material_splits.items() if len(splits) > 1]
    if index.get("split_strategy") == "heldout-region-v1":
        verify_region_split(found, index)
    elif leaked:
        raise ValueError(f"Material identities cross splits: {leaked}")
    if not all(any(s["metadata"]["split"] == split for s in found) for split in ("train", "validation")):
        raise ValueError("Training requires both training and validation samples")
    return found


def diffuse_encoding(metadata: dict[str, Any]) -> str:
    details = metadata.get("map_metadata", metadata.get("map_details", {}))
    input_details = details.get("input", details.get("diffuse", {}))
    encoding = input_details.get("encoding", input_details.get("transfer_function"))
    if encoding is None:
        encoding = metadata.get("input_encoding", metadata.get("diffuse_encoding"))
    if encoding in ("srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"):
        return "srgb"
    if encoding in ("linear", "linear_rgb", "linear_color", "linear_light"):
        return "linear"
    raise ValueError(f"Diffuse transfer must be declared explicitly, got {encoding!r}")


def linear_rgb(codes: np.ndarray, encoding: str, allow_nonopaque: bool = False) -> np.ndarray:
    if codes.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(f"Unsupported diffuse dtype: {codes.dtype}")
    if codes.ndim == 2:
        codes = np.repeat(codes[..., None], 3, axis=-1)
    elif codes.shape[-1] == 1:
        codes = np.repeat(codes, 3, axis=-1)
    if codes.shape[-1] not in (3, 4):
        raise ValueError("Diffuse must have RGB or opaque RGBA channels")
    maximum = float(np.iinfo(codes.dtype).max)
    # Published Poly Haven blue_metal_plate alpha differs by up to four uint16
    # codes from opaque. Preserve its RGB codes directly and ignore this tiny
    # alpha rounding error; meaningful transparency remains ineligible.
    alpha_tolerance = 8 if codes.dtype == np.uint16 else 0
    if not allow_nonopaque and codes.shape[-1] == 4 and np.any(codes[..., 3] < int(maximum) - alpha_tolerance):
        raise ValueError("Transparent diffuse samples must be reviewed before training")
    rgb = codes[..., :3].astype(np.float32) / maximum
    if encoding == "srgb":
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def load_pair(sample: dict[str, Any], device: torch.device, augment: bool = False, mask_input_alpha: bool = False, return_mask: bool = False) -> tuple:
    diffuse, diffuse_metadata = read_native_png(sample["input_path"])
    height, height_metadata = read_native_png(sample["height_path"])
    for role, actual in (("input", diffuse_metadata), ("height", height_metadata)):
        expected = sample["metadata"].get("map_metadata", {}).get(role, {}).get("sample_sha256")
        if not isinstance(expected, str) or actual["file_sha256"] != expected:
            raise ValueError(f"Prepared map checksum mismatch ({role}): {sample['metadata_path']}")
    if height.dtype != np.uint16:
        raise ValueError(f"Height must retain original 16-bit codes: {sample['height_path']} ({height.dtype})")
    if height.ndim == 3:
        if height.shape[-1] == 2:
            policy = sample["metadata"].get("map_metadata", {}).get("height", {}).get("scalar_alpha_policy", {})
            if (policy.get("scalar_component") != "grayscale" or policy.get("alpha_preserved_in_png") is not True
                    or policy.get("alpha_used_to_scale_scalar") is not False
                    or policy.get("uint16_near_opaque_tolerance_codes") != 8):
                raise ValueError("Grayscale+alpha height needs declared independent scalar/alpha preservation policy")
            if np.any(height[..., 1] < 65527):
                raise ValueError("Nonopaque height alpha must be reviewed before training")
        elif height.shape[-1] in (3, 4):
            if np.any(height[..., :3] != height[..., 0:1]):
                raise ValueError("RGB height channels differ; scalar conversion is not inferred")
            if height.shape[-1] == 4 and np.any(height[..., 3] != 65535):
                raise ValueError("Nonopaque height alpha must be reviewed before training")
        elif height.shape[-1] != 1:
            raise ValueError("Height must be scalar or identical opaque RGB channels")
        height = height[..., 0]
    if sample["metadata"].get("map_metadata", {}).get("height", {}).get("encoding") != "linear_data":
        raise ValueError("Height must declare raw linear_data encoding")
    valid = None
    sample["source_alpha_excluded_pixel_fraction"] = 0.0
    if mask_input_alpha and diffuse.shape[-1] == 4:
        maximum = np.iinfo(diffuse.dtype).max
        tolerance = 8 if diffuse.dtype == np.uint16 else 0
        raw_valid = diffuse[..., 3] >= maximum - tolerance
        sample["source_alpha_excluded_pixel_fraction"] = float(1 - raw_valid.mean())
        if not np.all(raw_valid):
            import cv2
            valid = cv2.erode(raw_valid.astype(np.uint8), np.ones((17, 17), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(np.float32)
            if float(valid.mean()) < 0.9:
                raise ValueError(f"More than 10% of diffuse crop excluded by transparency/context mask: {sample['input_path']}")
    try:
        rgb = linear_rgb(diffuse, diffuse_encoding(sample["metadata"]), allow_nonopaque=mask_input_alpha)
    except ValueError as error:
        raise ValueError(f"{error}: {sample['input_path']}") from error
    height = height.astype(np.float32) / np.float32(65535)
    if rgb.shape[:2] != height.shape:
        raise ValueError("Input/height dimensions differ")
    dimensions = sample["metadata"].get("sample_pixel_dimensions")
    if dimensions and list(reversed(height.shape)) != dimensions:
        raise ValueError("Decoded dimensions differ from sample metadata")
    if augment:
        turns = random.randrange(4)
        rgb, height = np.rot90(rgb, turns), np.rot90(height, turns)
        if valid is not None:
            valid = np.rot90(valid, turns)
        if random.getrandbits(1):
            rgb, height = rgb[:, ::-1], height[:, ::-1]
            if valid is not None:
                valid = valid[:, ::-1]
    source = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    target = torch.from_numpy(np.ascontiguousarray(height)).unsqueeze(0).unsqueeze(0).to(device)
    mask = None if valid is None else torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).unsqueeze(0).to(device)
    sample["loss_valid_pixel_fraction"] = 1.0 if valid is None else float(valid.mean())
    return (source, target, mask) if return_mask else (source, target)


def memory(device: torch.device) -> dict[str, int | str]:
    usage: dict[str, int | str] = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if sys.platform != "darwin":
        usage["process_peak_rss_bytes"] = int(usage["process_peak_rss_bytes"]) * 1024
    if device.type == "mps":
        torch.mps.synchronize()
        usage.update(mps_tensor_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory(), mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    usage["allocation_note"] = "RSS and Metal driver allocations can overlap; do not add them. Driver usage is a sampled peak, not an OS-wide pressure measurement."
    return usage


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; explicit --device cpu is required to choose CPU")
    return torch.device(name)


@torch.no_grad()
def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    prediction, target = prediction.float(), target.float()
    dx, dy = prediction[..., :, 1:] - prediction[..., :, :-1], prediction[..., 1:, :] - prediction[..., :-1, :]
    tx, ty = target[..., :, 1:] - target[..., :, :-1], target[..., 1:, :] - target[..., :-1, :]
    mx = None if mask is None else mask[..., :, 1:] * mask[..., :, :-1]
    my = None if mask is None else mask[..., 1:, :] * mask[..., :-1, :]
    gradient_energy = (weighted_mean(dx.abs(), mx) + weighted_mean(dy.abs(), my)) / 2
    target_energy = (weighted_mean(tx.abs(), mx) + weighted_mean(ty.abs(), my)) / 2
    residual_p = prediction - F.avg_pool2d(F.pad(prediction, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    residual_t = target - F.avg_pool2d(F.pad(target, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    return {
        "height_mae": float(weighted_mean((prediction - target).abs(), mask)),
        "centered_height_mae": float(weighted_mean((weighted_center(prediction, mask) - weighted_center(target, mask)).abs(), mask)),
        "height_rmse": float(weighted_mean((prediction - target) ** 2, mask).sqrt()),
        "native_gradient_mae": float((weighted_mean((dx - tx).abs(), mx) + weighted_mean((dy - ty).abs(), my)) / 2),
        "multiscale_gradient_mae": float(multiscale_gradient_loss(prediction, target, mask)),
        "detail_highpass_mae_radius_4": float(weighted_mean((residual_p - residual_t).abs(), detail_mask(mask))),
        "gradient_energy": float(gradient_energy),
        "target_gradient_energy": float(target_energy),
        "gradient_energy_ratio": float(gradient_energy / target_energy.clamp_min(1e-12)),
        "prediction_std": float(weighted_mean(weighted_center(prediction, mask) ** 2, mask).sqrt()),
        "target_std": float(weighted_mean(weighted_center(target, mask) ** 2, mask).sqrt()),
    }


def write_float_exr(path: Path, values: np.ndarray) -> None:
    """Write lossless FLOAT data channels and verify an exact decoded round trip."""
    import OpenEXR
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim == 2:
        channels = {"Y": values}
    elif values.ndim == 3 and values.shape[-1] == 3:
        channels = {name: np.ascontiguousarray(values[..., index]) for index, name in enumerate("RGB")}
    else:
        raise ValueError("Float EXR needs scalar or RGB data")
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage, "textureStudioEncoding": "linear_data"}
    # OpenEXR's constructor replaces values in its supplied dict with Channel
    # objects. Keep the original arrays separately for the numeric proof.
    OpenEXR.File(header, dict(channels)).write(str(path))
    decoded = OpenEXR.File(str(path), separate_channels=True)
    for name, original in channels.items():
        if decoded.channels()[name].pixels.dtype != np.float32 or not np.array_equal(decoded.channels()[name].pixels, original):
            raise ValueError(f"FLOAT EXR changed numeric samples: {path}")


def opengl_normal_from_height(values: np.ndarray, amplitude_to_width: float = 0.03) -> np.ndarray:
    y_gradient, x_gradient = np.gradient(values)
    # Square source pixels: displacement amplitude is relative to patch width,
    # so both axis slopes use the same texel spacing even for rectangular inputs.
    vector = np.stack((-x_gradient * values.shape[1] * amplitude_to_width, y_gradient * values.shape[1] * amplitude_to_width, np.ones_like(values)), axis=-1)
    vector /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)
    return vector * np.float32(0.5) + np.float32(0.5)


def save_prediction(directory: Path, prediction: torch.Tensor, source: torch.Tensor, target: torch.Tensor | None, label: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = prediction.detach().float().cpu().numpy()[0, 0]
    if not np.isfinite(values).all():
        raise ValueError("Cannot export nonfinite material height")
    np.save(directory / f"{label}.height.float32.npy", values, allow_pickle=False)
    write_float_exr(directory / f"{label}.height.float32.exr", values)
    # Display-only previews use an 8-bit fixed 0..1 mapping. Numeric training
    # and exported Float32 predictions never pass through these previews.
    import cv2
    def preview(path: Path, array: np.ndarray) -> None:
        visual = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        if not cv2.imwrite(str(path), visual):
            raise OSError(f"Could not write preview: {path}")
    preview(directory / f"{label}.height.preview.png", values)
    # Tangent derivatives use normalized UV pixel spacing. Default artistic
    # displacement amplitude is 3% of the square patch width; it is declared,
    # not inferred physical scale. +Y is up while image rows run downwards.
    normal = opengl_normal_from_height(values)
    np.save(directory / f"{label}.normal_opengl.float32.npy", normal, allow_pickle=False)
    write_float_exr(directory / f"{label}.normal_opengl.float32.exr", normal)
    preview(directory / f"{label}.normal_opengl.preview.png", normal[..., ::-1])
    if target is not None and not (directory / "target.height.float32.npy").exists():
        np.save(directory / "target.height.float32.npy", target.detach().float().cpu().numpy()[0, 0], allow_pickle=False)
        preview(directory / "target.height.preview.png", target.detach().float().cpu().numpy()[0, 0])
    if not (directory / "input.preview.png").exists():
        color = source.detach().float().cpu().numpy()[0].transpose(1, 2, 0)
        color = np.where(color <= 0.0031308, color * 12.92, 1.055 * np.power(np.maximum(color, 0), 1 / 2.4) - 0.055)
        preview(directory / "input.preview.png", color[..., ::-1])
    write_json(directory / f"{label}.prediction.json", {"schema": SCHEMA, "height": f"{label}.height.float32.exr", "raw_height": f"{label}.height.float32.npy", "height_encoding": "linear_relative_source_0_1", "exr_storage": "FLOAT32 lossless ZIP, exact numeric round-trip verified", "blender_color_space": "Non-Color", "no_per_image_normalization": True, "normal_convention": "OpenGL +Y", "normal_height_to_patch_width_ratio": 0.03, "normal_physical_scale_known": False, "preview_encoding": "fixed_0_1_to_uint8_display_only"})


@torch.no_grad()
def evaluate(model: MaterialHeightNet, samples: list[dict[str, Any]], device: torch.device, output: Path, label: str, baseline_directory: Path | None = None, export_limit: int = 8, mask_input_alpha: bool = False, objective: str = "current-l1") -> dict[str, Any]:
    model.eval()
    records = []
    for sample_number, sample in enumerate(samples):
        source, target, mask = load_pair(sample, device, mask_input_alpha=mask_input_alpha, return_mask=True)
        prediction = model(source)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite model prediction")
        metadata = sample["metadata"]
        record = {"sample_id": metadata["sample_id"], "material_id": metadata["material_id"], "valid_pixel_fraction": 1.0 if mask is None else float(mask.mean()), "source_alpha_excluded_pixel_fraction": sample.get("source_alpha_excluded_pixel_fraction", 0.0), "trained_model": prediction_metrics(prediction, target, mask), "flat_0_5": prediction_metrics(torch.full_like(target, 0.5), target, mask), "linear_luminance": prediction_metrics((source[:, 0:1] * 0.2126 + source[:, 1:2] * 0.7152 + source[:, 2:3] * 0.0722), target, mask)}
        if objective == "relative-squared":
            objective_loss, objective_components = squared_objective(prediction, target, mask)
            record["objective_loss"] = float(objective_loss)
            record["objective_components"] = objective_components
        directory = output / "predictions" / metadata["sample_id"]
        if sample_number < export_limit:
            save_prediction(directory, prediction, source, target, label)
            if mask is not None:
                np.save(directory / "confidence.valid.bool.npy", mask.cpu().numpy()[0, 0].astype(bool), allow_pickle=False)
                import cv2
                if not cv2.imwrite(str(directory / "confidence.valid.preview.png"), mask.cpu().numpy()[0, 0].astype(np.uint8) * 255):
                    raise OSError(f"Could not write validity mask: {directory}")
                write_json(directory / "confidence.valid.json", {"meaning": "Opaque input pixels and 8-pixel context margin eligible for supervised loss/metrics; not model-estimated confidence", "excluded_prediction_pixels_unassessed": True, "valid_pixel_fraction": record["valid_pixel_fraction"]})
        if baseline_directory:
            path = baseline_directory / f"{metadata['sample_id']}.npy"
            if path.is_file():
                raw = np.load(path, allow_pickle=False)
                if raw.dtype != np.float32 or raw.shape != tuple(target.shape[-2:]) or not np.isfinite(raw).all():
                    raise ValueError(f"DA3 baseline must be finite Float32 height at exact sample size: {path}")
                baseline = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)
                record["da3_height"] = prediction_metrics(baseline, target, mask)
                record["da3_baseline_sha256"] = digest(path)
                if sample_number < export_limit:
                    save_prediction(directory, baseline, source, target, "da3")
            else:
                record["da3_baseline_missing"] = str(path)
        records.append(record)
    methods = sorted({key for record in records for key in ("trained_model", "flat_0_5", "linear_luminance", "da3_height") if key in record})
    aggregate = {method: {metric: float(np.mean([record[method][metric] for record in records if method in record])) for metric in records[0]["trained_model"]} for method in methods}
    evaluated_splits = sorted({sample["metadata"]["split"] for sample in samples})
    result = {"label": label, "evaluated_splits": evaluated_splits, "validation_scope": "training regions already used for optimization" if evaluated_splits == ["train"] else samples[0].get("validation_scope", "unseen materials"), "sample_count": len(records), "material_count": len({record["material_id"] for record in records}), "prediction_exports": min(len(records), export_limit), "metrics_cover_all_samples": True, "aggregation": "equal weight per sample; not scientific height calibration", "aggregate": aggregate, "samples": records}
    result["exported_sample_ids"] = [sample["metadata"]["sample_id"] for sample in samples[:export_limit]]
    result["mean_valid_pixel_fraction"] = float(np.mean([record["valid_pixel_fraction"] for record in records]))
    result["alpha_masking_enabled"] = mask_input_alpha
    result["objective"] = objective
    if objective == "relative-squared":
        result["mean_objective_loss"] = float(np.mean([record["objective_loss"] for record in records]))
    write_json(output / f"evaluation.{label}.json", result)
    return result


def load_checkpoint(path: Path, device: torch.device) -> tuple[MaterialHeightNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported material-height checkpoint schema")
    model = MaterialHeightNet(int(checkpoint["model_config"]["base_channels"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device), checkpoint


def score_validation(result: dict[str, Any], args: argparse.Namespace) -> float:
    if args.objective == "relative-squared":
        return float(result["mean_objective_loss"])
    metrics = result["aggregate"]["trained_model"]
    return float(metrics["centered_height_mae" if args.offset_invariant_loss else "height_mae"] + args.gradient_weight * metrics["multiscale_gradient_mae"] + args.highpass_weight * metrics["detail_highpass_mae_radius_4"])


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = find_samples(args.dataset, args.allow_unreviewed)
    training = [sample for sample in samples if sample["metadata"]["split"] == "train"]
    validation = [sample for sample in samples if sample["metadata"]["split"] == "validation"]
    # Validate all active maps, transfers and alpha eligibility before reserving
    # GPU memory or starting a run. This keeps errors concrete and checkable.
    for sample in samples:
        preflight_source, preflight_target, preflight_mask = load_pair(sample, torch.device("cpu"), mask_input_alpha=args.mask_input_alpha, return_mask=True)
        del preflight_source, preflight_target, preflight_mask
    initial_checkpoint = None
    if args.initial_checkpoint:
        model, initial_checkpoint = load_checkpoint(args.initial_checkpoint, torch.device("cpu"))
        if int(initial_checkpoint["model_config"]["base_channels"]) != args.base_channels:
            raise ValueError("Initial checkpoint base_channels differs from explicit training architecture")
        model = model.to(device)
    else:
        model = MaterialHeightNet(args.base_channels).to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    provenance = {"schema": SCHEMA, "architecture": ARCHITECTURE, "model_config": {"base_channels": args.base_channels}, "dataset": str(args.dataset.resolve()), "dataset_index_sha256": digest(args.dataset / "dataset.json") if (args.dataset / "dataset.json").is_file() else None, "sample_manifest_sha256": {sample["metadata"]["sample_id"]: digest(sample["metadata_path"]) for sample in samples}, "train_materials": sorted({sample["metadata"]["material_id"] for sample in training}), "validation_materials": sorted({sample["metadata"]["material_id"] for sample in validation}), "train_samples": len(training), "validation_samples": len(validation), "quality_review_required_for_production": True, "allow_unreviewed_for_requested_pilot": args.allow_unreviewed, "source_images_modified": False, "input": "linear_RGB_at_native_crop_resolution", "height": "raw_uint16_codes_divided_by_65535_in_Float32_no_gamma_no_per_crop_stretch", "loss": {"height": "L1_Float32", "gradient": "L1_Float32_finite_differences_scales_1_2_4_8", "gradient_weight": args.gradient_weight}, "seed": args.seed, "torch": str(torch.__version__), "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "license": "GPL-3.0-or-later code; curated-source licenses remain in dataset metadata", "limitations": ["Small curated reflectance-to-height pilot, not a trained DA3 LoRA.", "Not trained to remove lighting from casual photos.", "Relative source height offsets/scales may differ among materials.", "Held-out materials do not establish phone-photo generalization."]}
    provenance["input_alpha_policy"] = "Preserve raw RGB; ignore only near-opaque uint16 alpha >=65527 (rounding tolerance), reject meaningful transparency."
    if args.mask_input_alpha:
        provenance["input_alpha_policy"] = "Preserve raw straight RGB, no alpha compositing. Exclude nonopaque alpha (uint16<65527 or uint8<255) plus 8-pixel context margin from height/gradient loss and evaluation. Reject samples with <90% valid pixels. Multiscale uses fully valid pooling blocks; high-pass requires full 9x9 support."
    provenance["alpha_masking_enabled"] = args.mask_input_alpha
    provenance["sample_alpha_eligibility"] = {sample["metadata"]["sample_id"]: {"source_alpha_excluded_pixel_fraction": sample["source_alpha_excluded_pixel_fraction"], "loss_valid_pixel_fraction": sample["loss_valid_pixel_fraction"]} for sample in samples}
    provenance["validation_scope"] = samples[0]["validation_scope"]
    provenance["prediction_export_limit_per_evaluation"] = args.export_limit
    provenance["validation_prediction_sample_ids"] = [sample["metadata"]["sample_id"] for sample in validation[:args.export_limit]]
    if provenance["validation_scope"] == "unseen regions of known materials":
        provenance["limitations"] = ["Small curated reflectance-to-height pilot, not a trained DA3 LoRA.", "Not trained to remove lighting from casual photos.", "Relative source height offsets/scales may differ among materials.", "Validation measures unseen nonoverlapping regions of training materials; it does not establish unseen-material or phone-photo generalization."]
    provenance["sample_map_sha256"] = {sample["metadata"]["sample_id"]: {role: sample["metadata"]["map_metadata"][role]["sample_sha256"] for role in ("input", "height")} for sample in samples}
    provenance["sample_integrity_policy"] = "Each decoded active map checksum must match its verified preparation record."
    provenance["loss"]["offset_invariant_height_loss"] = args.offset_invariant_loss
    provenance["loss"]["detail_highpass_weight"] = args.highpass_weight
    provenance["objective"] = args.objective
    provenance["optimizer_state_resumed"] = False
    provenance["optimizer_initialization"] = "Fresh AdamW state at requested learning rate; optional checkpoint supplies model weights only"
    provenance["initial_checkpoint"] = None if args.initial_checkpoint is None else {"path": str(args.initial_checkpoint.resolve()), "sha256": digest(args.initial_checkpoint), "trained_steps": initial_checkpoint.get("step"), "schema": initial_checkpoint["schema"], "architecture": initial_checkpoint["architecture"], "base_channels": initial_checkpoint["model_config"]["base_channels"]}
    provenance["validation_every_epochs"] = args.validation_every_epochs
    implementation_directory = args.output / "implementation"
    implementation_directory.mkdir()
    provenance["implementation_sources"] = {}
    for filename in ("train_material_height.py", "material_height_model.py", "material_dataset.py"):
        source_path = Path(__file__).resolve().parent / filename
        source_bytes = source_path.read_bytes()
        snapshot_path = implementation_directory / filename
        snapshot_path.write_bytes(source_bytes)
        provenance["implementation_sources"][filename] = {"original_path": str(source_path), "snapshot_relative_path": "implementation/" + filename, "sha256": hashlib.sha256(source_bytes).hexdigest()}
    provenance["implementation_snapshot_scope"] = "On-disk source captured at run initialization; preserves reproducible implementation files for this run"
    provenance["best_selection_policy"] = "Best validated objective score including the actual initial model at new-run step zero"
    provenance["time_budget_scope"] = "Optimizer phase after preflight and initial validation; scheduled validation is included, final/best/training-fit reports can extend total wall time"
    if args.objective == "relative-squared":
        provenance["loss"] = {"objective": "relative-squared", "height": "Centered Float32 MSE / fixed detached target centered energy", "gradient": "Mean Float32 MSE / fixed detached target energy at each axis and scale 1/2/4/8", "highpass": "Float32 radius-4 high-pass MSE / fixed detached target high-pass energy", "energy_floors": {"height": 1e-6, "gradient": 1e-7, "highpass": 1e-7}, "loss_terms_equal_weight": True, "source_targets_rescaled": False, "offset_invariant_height_loss": True}
    provenance["output_height_origin"] = "Raw sigmoid prediction; absolute height origin is unconstrained when offset-invariant loss is selected. predict --neutral-origin adds a declared constant only, preserving amplitudes."
    write_json(args.output / "run.json", provenance)
    write_json(args.output / "dataset.snapshot.json", {"dataset_index": json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8")), "sample_metadata": {sample["metadata"]["sample_id"]: sample["metadata"] for sample in samples}})
    initial = evaluate(model, validation, device, args.output, "initial" if args.initial_checkpoint else "untrained", args.baseline_directory, args.export_limit, args.mask_input_alpha, args.objective)
    initial_score = score_validation(initial, args)
    torch.save({**provenance, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "step": 0, "epoch": 0, "validation_score": initial_score, "validation_step": 0, "validation_evaluated_for_checkpoint": True}, args.output / "checkpoint.best.pt")
    print(json.dumps({"event": "initial_validation", "metrics": initial["aggregate"], "memory": memory(device)}), flush=True)
    started = time.monotonic()
    cancelled = False
    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    best_score = initial_score
    step = 0
    samples_per_epoch = len(training)
    peak_memory: dict[str, int | str] = memory(device)
    history_path = args.output / "training.jsonl"
    first_step = None
    for epoch in range(args.epochs):
        order = list(training)
        random.shuffle(order)
        model.train()
        for sample in order:
            if cancelled or step >= args.max_steps or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
                break
            source, target, mask = load_pair(sample, device, augment=True, mask_input_alpha=args.mask_input_alpha, return_mask=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source)
            if args.objective == "relative-squared":
                loss, components = squared_objective(prediction, target, mask)
            else:
                loss, components = height_loss(prediction, target, args.gradient_weight, args.offset_invariant_loss, args.highpass_weight, mask)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            before = model.head.weight.detach().clone() if step == 0 else None
            optimizer.step()
            step += 1
            if step == 1:
                first_step = {"forward_backward_optimizer_succeeded": True, "native_dimensions": [int(source.shape[-1]), int(source.shape[-2])], "batch_size": 1, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "head_parameters_changed": bool(torch.any(before != model.head.weight.detach())), "memory": memory(device)}
                write_json(args.output / "training-memory-probe.json", first_step)
            if step == 1 or step % 10 == 0:
                current_memory = memory(device)
                for key, value in current_memory.items():
                    if isinstance(value, int):
                        peak_memory[key] = max(int(peak_memory.get(key, 0)), value)
                record = {"event": "training", "step": step, "epoch": epoch + 1, "sample_id": sample["metadata"]["sample_id"], "loss": float(loss.detach()), **components, "valid_pixel_fraction": sample["loss_valid_pixel_fraction"], "source_alpha_excluded_pixel_fraction": sample["source_alpha_excluded_pixel_fraction"], "gradient_norm": float(gradient_norm), "elapsed_seconds": time.monotonic() - started, "memory": current_memory}
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            del source, target, prediction, loss, mask
        terminal = cancelled or step >= args.max_steps or epoch + 1 == args.epochs or bool(args.max_minutes and time.monotonic() - started >= args.max_minutes * 60)
        should_validate = terminal or (epoch + 1) % args.validation_every_epochs == 0
        score = None
        if should_validate:
            validation_result = evaluate(model, validation, device, args.output, "latest", args.baseline_directory, args.export_limit, args.mask_input_alpha, args.objective)
            score = score_validation(validation_result, args)
        checkpoint = {**provenance, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "step": step, "epoch": epoch + 1, "validation_score": score, "validation_step": step if should_validate else None, "validation_evaluated_for_checkpoint": should_validate}
        torch.save(checkpoint, args.output / "checkpoint.last.pt")
        if should_validate and score < best_score:
            best_score = score
            torch.save(checkpoint, args.output / "checkpoint.best.pt")
        if should_validate:
            print(json.dumps({"event": "validation", "epoch": epoch + 1, "step": step, "score": score, "objective": args.objective, "metrics": validation_result["aggregate"]}), flush=True)
        if terminal or (args.max_minutes and time.monotonic() - started >= args.max_minutes * 60):
            break
    if step == 0:
        raise ValueError("No training steps completed")
    best_model, best_checkpoint = load_checkpoint(args.output / "checkpoint.best.pt", device)
    result = evaluate(best_model, validation, device, args.output, "best", args.baseline_directory, args.export_limit, args.mask_input_alpha, args.objective)
    training_result = evaluate(best_model, training, device, args.output, "training-fit", None, args.export_limit, args.mask_input_alpha, args.objective) if args.evaluate_training else None
    summary = {**provenance, "completed_steps": step, "nominal_samples_per_epoch": samples_per_epoch, "elapsed_seconds": time.monotonic() - started, "cancelled_after_current_step": cancelled, "first_real_training_step": first_step, "sampled_peak_memory": peak_memory, "best_checkpoint_sha256": digest(args.output / "checkpoint.best.pt"), "initial_validation": initial["aggregate"], "best_validation": result["aggregate"], "initial_to_best_height_mae_change": result["aggregate"]["trained_model"]["height_mae"] - initial["aggregate"]["trained_model"]["height_mae"], "initial_to_best_native_gradient_mae_change": result["aggregate"]["trained_model"]["native_gradient_mae"] - initial["aggregate"]["trained_model"]["native_gradient_mae"], "automatic_production_promotion": False}
    write_json(args.output / "summary.json", summary)
    summary["validation_mean_valid_pixel_fraction"] = result["mean_valid_pixel_fraction"]
    summary["best_checkpoint_training_step"] = best_checkpoint["step"]
    summary["best_checkpoint_validated_at_same_step"] = best_checkpoint["validation_evaluated_for_checkpoint"] and best_checkpoint["validation_step"] == best_checkpoint["step"]
    summary["initial_validation_objective_score"] = initial_score
    summary["best_validation_objective_score"] = score_validation(result, args)
    summary["selected_checkpoint_validation_score"] = best_checkpoint["validation_score"]
    summary["best_improved_over_initial_objective"] = best_checkpoint["validation_score"] < initial_score
    write_json(args.output / "summary.json", summary)
    if training_result:
        summary["training_fit_metrics"] = training_result["aggregate"]
        summary["training_fit_sample_count"] = len(training)
        write_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "complete", "steps": step, "summary": str(args.output / "summary.json"), "best_validation": result["aggregate"]}), flush=True)


def probe_native_sizes(args: argparse.Namespace) -> None:
    """Measure actual training on two native parent crops without changing maps."""
    device = choose_device(args.device)
    diffuse, diffuse_header = read_native_png(args.source_input)
    height_codes, height_header = read_native_png(args.source_height)
    if diffuse.shape[:2] != height_codes.shape[:2]:
        raise ValueError("Probe parent maps must have identical registered dimensions")
    if height_codes.dtype != np.uint16:
        raise ValueError("Probe parent height must retain uint16 source codes")
    if height_codes.shape[-1] == 2:
        if np.any(height_codes[..., 1] < 65527):
            raise ValueError("Probe grayscale+alpha height has non-opaque alpha requiring review")
    elif height_codes.shape[-1] in (3, 4):
        if np.any(height_codes[..., :3] != height_codes[..., 0:1]):
            raise ValueError("Probe RGB height channels differ")
        if height_codes.shape[-1] == 4 and np.any(height_codes[..., 3] != 65535):
            raise ValueError("Probe height alpha is not opaque")
    elif height_codes.shape[-1] != 1:
        raise ValueError("Probe height must be scalar, near-opaque grayscale+alpha, or identical opaque RGB")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "texture-studio-native-training-size-probe-v1", "architecture": ARCHITECTURE, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint), "source_input": str(args.source_input.resolve()), "source_input_sha256": diffuse_header["file_sha256"], "source_height": str(args.source_height.resolve()), "source_height_sha256": height_header["file_sha256"], "source_dimensions": [diffuse.shape[1], diffuse.shape[0]], "input_encoding": args.input_encoding, "target_encoding": "untouched_uint16_codes_divided_by_65535_in_memory_Float32", "source_images_modified": False, "source_images_resized": False, "batch_size": 1, "arithmetic": "Float32 model, optimizer, target and loss", "device": str(device), "torch": str(torch.__version__), "warmup_steps": args.warmup_steps, "measured_steps": args.steps, "probe_weights_promoted": False, "results": []}
    if height_codes.shape[-1] == 2:
        report["scalar_alpha_policy"] = {"scalar_component": "grayscale", "alpha_preserved_in_source": True,
            "alpha_used_to_scale_scalar": False, "uint16_near_opaque_tolerance_codes": 8}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Probe checkpoint schema is incompatible")
    report["parameter_count"] = sum(value.numel() for value in checkpoint["model_state"].values())
    report["loss"] = checkpoint["loss"]
    for size in args.sizes:
        result: dict[str, Any] = {"crop_rectangle_top_left_xywh": [0, 0, size, size], "native_size": size, "timing_seconds": [], "successful": False}
        if size > min(diffuse.shape[:2]):
            result["error"] = "Parent map is smaller than requested native crop"
            report["results"].append(result)
            continue
        model = optimizer = source = target = prediction = loss = None
        try:
            rgb = linear_rgb(diffuse[:size, :size], args.input_encoding)
            target_array = np.ascontiguousarray(height_codes[:size, :size, 0], dtype=np.float32) / np.float32(65535)
            source = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
            target = torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0).to(device)
            model = MaterialHeightNet(int(checkpoint["model_config"]["base_channels"])).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
            before = model.head.weight.detach().clone()
            memories = []
            for step in range(args.warmup_steps + args.steps):
                if device.type == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(source)
                if prediction.shape != target.shape:
                    raise ValueError("Native prediction dimensions changed")
                if checkpoint.get("objective") == "relative-squared":
                    loss, components = squared_objective(prediction, target)
                else:
                    loss, components = height_loss(prediction, target, checkpoint["loss"]["gradient_weight"], checkpoint["loss"].get("offset_invariant_height_loss", False), checkpoint["loss"].get("detail_highpass_weight", 0))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite native-size probe loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                duration = time.perf_counter() - started
                memories.append(memory(device))
                if step >= args.warmup_steps:
                    result["timing_seconds"].append(duration)
                result["last_loss"] = float(loss.detach())
                result["last_gradient_norm"] = float(gradient_norm)
                result["last_loss_components"] = components
            result.update(successful=True, forward_backward_optimizer_succeeded=True, head_parameters_changed=bool(torch.any(before != model.head.weight.detach())), mean_seconds_per_step=float(np.mean(result["timing_seconds"])), median_seconds_per_step=float(np.median(result["timing_seconds"])), sampled_peak_memory={key: max(int(row[key]) for row in memories) for key in memories[0] if isinstance(memories[0][key], int)}, memory_note=memories[0]["allocation_note"])
            if not result["head_parameters_changed"]:
                raise ValueError("Probe optimizer did not change real checkpoint parameters")
        except RuntimeError as error:
            result["error"] = str(error)
            result["out_of_memory"] = "out of memory" in str(error).lower()
            result["fallback_used"] = False
        finally:
            model = optimizer = source = target = prediction = loss = None
            if device.type == "mps":
                torch.mps.empty_cache()
        report["results"].append(result)
        write_json(args.output / "probe.json", report)
        print(json.dumps(result, allow_nan=False), flush=True)
    succeeded = {result["native_size"]: result for result in report["results"] if result["successful"]}
    if 1024 in succeeded and 2048 in succeeded:
        report["2048_to_1024_mean_step_time_ratio"] = succeeded[2048]["mean_seconds_per_step"] / succeeded[1024]["mean_seconds_per_step"]
    write_json(args.output / "probe.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True, help="New run directory; existing outputs are never overwritten")
    training.add_argument("--device", choices=("mps", "cpu"), default="mps")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--max-steps", type=int, default=600)
    training.add_argument("--max-minutes", type=float, default=15)
    training.add_argument("--base-channels", type=int, default=12)
    training.add_argument("--objective", choices=("current-l1", "relative-squared"), default="current-l1")
    training.add_argument("--initial-checkpoint", type=Path, help="Load matching model weights with a fresh optimizer; does not resume optimizer state")
    training.add_argument("--validation-every-epochs", type=int, default=1, help="Full validation cadence; initial and final validation always run")
    training.add_argument("--learning-rate", type=float, default=0.0003)
    training.add_argument("--gradient-weight", type=float, default=8)
    training.add_argument("--offset-invariant-loss", action="store_true", help="Remove a constant height offset inside the loss only; raw targets and their amplitudes remain untouched")
    training.add_argument("--highpass-weight", type=float, default=0, help="Additional native-pixel radius-4 high-pass L1 loss weight")
    training.add_argument("--seed", type=int, default=2307)
    training.add_argument("--allow-unreviewed", action="store_true")
    training.add_argument("--baseline-directory", type=Path)
    training.add_argument("--evaluate-training", action="store_true", help="Measure best checkpoint on all training samples as a separately labeled fitting diagnostic")
    training.add_argument("--prediction-limit", "--export-limit", dest="export_limit", type=int, default=8, help="Maximum prediction artifact sets per evaluation; metrics still cover every sample")
    training.add_argument("--mask-transparent-input", dest="mask_input_alpha", action="store_true", help="Mask nonopaque input pixels plus an 8-pixel margin from loss/metrics, preserve straight RGB and raw height")
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--dataset", type=Path, required=True)
    evaluating.add_argument("--checkpoint", type=Path, required=True)
    evaluating.add_argument("--output", type=Path, required=True)
    evaluating.add_argument("--device", choices=("mps", "cpu"), default="mps")
    evaluating.add_argument("--allow-unreviewed", action="store_true")
    evaluating.add_argument("--baseline-directory", type=Path)
    evaluating.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluating.add_argument("--prediction-limit", "--export-limit", dest="export_limit", type=int, default=8)
    evaluating.add_argument("--mask-transparent-input", dest="mask_input_alpha", action="store_true")
    evaluating.add_argument("--objective", choices=("current-l1", "relative-squared"), help="Defaults to checkpoint's recorded objective")
    predicting = commands.add_parser("predict")
    predicting.add_argument("--input", type=Path, required=True)
    predicting.add_argument("--input-encoding", choices=("srgb", "linear"), required=True)
    predicting.add_argument("--checkpoint", type=Path, required=True)
    predicting.add_argument("--output", type=Path, required=True)
    predicting.add_argument("--device", choices=("mps", "cpu"), default="mps")
    predicting.add_argument("--neutral-origin", action="store_true", help="Add a constant to predicted height so its mean is 0.5; no scale change or clipping")
    probing = commands.add_parser("probe", help="Measure real forward/backward/optimizer steps on native crops from matching original parent maps")
    probing.add_argument("--source-input", type=Path, required=True)
    probing.add_argument("--source-height", type=Path, required=True)
    probing.add_argument("--input-encoding", choices=("srgb", "linear"), required=True)
    probing.add_argument("--checkpoint", type=Path, required=True)
    probing.add_argument("--output", type=Path, required=True)
    probing.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    probing.add_argument("--warmup-steps", type=int, default=1)
    probing.add_argument("--steps", type=int, default=5)
    probing.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.command == "probe":
        if args.steps < 1 or args.warmup_steps < 0 or any(size < 16 for size in args.sizes):
            parser.error("Probe sizes must be >=16; measured steps positive and warmup nonnegative")
        probe_native_sizes(args)
    elif args.command == "train":
        if args.epochs < 1 or args.max_steps < 1 or args.max_minutes < 0 or args.learning_rate <= 0 or args.gradient_weight < 0 or args.highpass_weight < 0 or args.export_limit < 0 or args.validation_every_epochs < 1:
            parser.error("Training limits and learning rate must be positive; gradient weight/time limit cannot be negative")
        train(args)
    elif args.command == "evaluate":
        args.output.mkdir(parents=True, exist_ok=False)
        device = choose_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        if args.export_limit < 0:
            parser.error("Export limit cannot be negative")
        samples = [sample for sample in find_samples(args.dataset, args.allow_unreviewed) if sample["metadata"]["split"] == args.split]
        evaluate(model, samples, device, args.output, "training-fit" if args.split == "train" else "best", args.baseline_directory, args.export_limit, args.mask_input_alpha, args.objective or checkpoint.get("objective", "current-l1"))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        device = choose_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        raw, _ = read_native_png(args.input)
        rgb = linear_rgb(raw, args.input_encoding)
        source = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(source)
        origin_offset = float(0.5 - prediction.mean()) if args.neutral_origin else 0.0
        if args.neutral_origin:
            prediction = prediction + origin_offset
        save_prediction(args.output, prediction, source, None, "predicted")
        write_json(args.output / "inference.json", {"schema": SCHEMA, "checkpoint_sha256": digest(args.checkpoint), "input_sha256": digest(args.input), "input_encoding": args.input_encoding, "native_dimensions": [rgb.shape[1], rgb.shape[0]], "trained_steps": checkpoint["step"], "constant_height_origin_offset_added": origin_offset, "height_amplitudes_rescaled": False, "memory": memory(device)})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
