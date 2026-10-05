"""Inspect and curate datasets without changing their precision-preserved arrays.

Review PNGs are disposable visualizations. They never become training labels:
the original scientific arrays, including NaNs and floating-point bits, stay intact.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .dataset import DatasetError, load_dataset
from .formats import sha256_array, sha256_file, write_png
from .array_storage import ARRAY_SUFFIXES, read_array
from .concurrency import ordered_map, resolve_workers


LABEL_TITLES = {
    "training": "Selected training target",
    "teacher": "Teacher on stereo left image",
    "anchored_teacher": "Teacher with estimated meter scale",
    "metric_anchor": "DepthPro scale estimate on stereo left image",
    "display_teacher": "Teacher on full display image",
    "registered_display_teacher": "Display teacher aligned to stereo image",
    "anchored_display_teacher": "Display teacher with estimated meter scale",
    "display_metric_anchor": "DepthPro scale estimate on full display image",
    "reference": "Supplied measured reference",
}
PREVIEW_LABELS = tuple(LABEL_TITLES)
_TRAINING_LABELS = {"teacher", "anchored_teacher", "registered_display_teacher", "reference", "display_teacher", "anchored_display_teacher"}
_DEPTH_UNITS = {"meters", "relative_depth", "relative_inverse_depth"}
_ARRAY_FIELDS = {"path", "shape", "dtype", "array_sha256", "file_sha256"}
_TEACHER_PAYLOAD_FIELDS = frozenset({"teacher", "raft_target", "display_teacher", "registered_display_teacher",
                                    "metric_anchor", "display_metric_anchor", "anchored_teacher", "anchored_display_teacher"})


def _label_title(key: str, label: dict[str, Any]) -> str:
    if key == "teacher" and str(label.get("coordinate_reference", "")).startswith("display"):
        return LABEL_TITLES["display_teacher"]
    return LABEL_TITLES[key]


def _array_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if _ARRAY_FIELDS.issubset(value):
            yield value
        else:
            for key, child in value.items():
                if value.get("teacher_payload_removed") is True and value.get("excluded") is True and key in _TEACHER_PAYLOAD_FIELDS:
                    continue
                yield from _array_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _array_records(child)


def _array_path(root: Path, record: dict[str, Any], *, validate_file: bool = True) -> Path:
    name = record.get("path")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise DatasetError("Dataset array path escapes the dataset directory or is not relative")
    path = root / name
    if path.suffix.lower() not in ARRAY_SUFFIXES:
        raise DatasetError("Dataset array path is not an NPY/NPZ/EXR/PNG array")
    if not validate_file:
        return path
    path = path.resolve()
    if not path.is_relative_to(root) or path == root or path.suffix.lower() not in ARRAY_SUFFIXES:
        raise DatasetError("Dataset array path escapes the dataset directory or is not an NPY/NPZ/EXR/PNG array")
    if not path.is_file():
        raise DatasetError(f"Dataset array is missing: {name}")
    return path


def _label(sample: dict[str, Any], key: str) -> tuple[str, dict[str, Any]]:
    resolved = sample.get("training_target_choice", "teacher") if key == "training" else key
    if resolved not in LABEL_TITLES or resolved == "training" or (key == "training" and resolved not in _TRAINING_LABELS):
        raise DatasetError(f"Unsupported training target choice: {resolved!r}")
    label = sample.get(resolved)
    if not isinstance(label, dict) or not isinstance(label.get("target"), dict) or not _ARRAY_FIELDS.issubset(label["target"]):
        raise DatasetError(f"Sample {sample['id']} has no {resolved} depth label")
    if label.get("units") not in _DEPTH_UNITS:
        raise DatasetError(f"Sample {sample['id']} has unsupported {resolved} units")
    return resolved, label


def _read_manifest_snapshot(directory: Path | str, *, validate_files: bool = True) -> tuple[Path, dict[str, Any], str]:
    """Read one manifest snapshot; optional file checks never decode payloads."""
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "dataset.json"
    if not manifest_path.resolve().is_relative_to(root):
        raise DatasetError("Dataset manifest path escapes the dataset directory")
    try:
        serialized = manifest_path.read_bytes()
        manifest = json.loads(serialized)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read dataset manifest: {exc}") from exc
    _validate_manifest_metadata(root, manifest, validate_files=validate_files)
    return root, manifest, hashlib.sha256(serialized).hexdigest()


def _validate_manifest_metadata(root: Path, manifest: Any, *, validate_files: bool = True) -> None:
    """Validate a captured snapshot without rereading its mutable manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != "ipde-depth-dataset-v1":
        raise DatasetError("Not a supported IPDE depth dataset")
    pending = manifest.get("pending_photos", [])
    if not isinstance(pending, list) or any(not isinstance(item, str) or not item or not Path(item).is_absolute() for item in pending) or len(pending) != len(set(pending)):
        raise DatasetError("Pending photos must be unique absolute source paths")
    revision = manifest.get("edit_revision", 0)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise DatasetError("Dataset edit revision must be a nonnegative integer")
    fraction = manifest.get("validation_fraction")
    if fraction is not None and (not isinstance(fraction, (float, int)) or isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise DatasetError("Validation fraction must be between zero and one, inclusive")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DatasetError("Dataset must contain at least one sample")
    ids: set[str] = set()
    splits: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not sample["id"]:
            raise DatasetError("Dataset sample ID is missing or malformed")
        if sample["id"] in ids:
            raise DatasetError(f"Duplicate dataset sample ID: {sample['id']}")
        ids.add(sample["id"])
        if "excluded" in sample and not isinstance(sample["excluded"], bool):
            raise DatasetError(f"Sample {sample['id']} has malformed excluded membership")
        if "teacher_payload_removed" in sample and (not isinstance(sample["teacher_payload_removed"], bool)
                or sample["teacher_payload_removed"] and sample.get("excluded") is not True):
            raise DatasetError(f"Sample {sample['id']} has malformed discarded teacher membership; regenerate it before enabling")
        group, split = sample.get("group_id"), sample.get("split")
        if not isinstance(group, str) or not group or split not in {"train", "validation"}:
            raise DatasetError(f"Sample {sample['id']} has invalid group/split metadata")
        if group in splits and splits[group] != split:
            raise DatasetError("Training/validation leakage: an independent group crosses the split")
        splits[group] = split
        if not isinstance(sample.get("source_path"), str):
            raise DatasetError(f"Sample {sample['id']} has no source path")
        for rgb_key in ("rgb", "right_rgb"):
            if not isinstance(sample.get(rgb_key), dict) or not _ARRAY_FIELDS.issubset(sample[rgb_key]):
                raise DatasetError(f"Sample {sample['id']} has no {rgb_key} array record")
        _label(sample, "teacher")
        _label(sample, "training")
        for key in LABEL_TITLES:
            if key != "training" and key in sample:
                _label(sample, key)
        if sample.get("teacher_payload_removed"):
            for key in _TEACHER_PAYLOAD_FIELDS:
                for record in _array_records(sample.get(key)):
                    _array_path(root, record, validate_file=False)
        for record in _array_records(sample):
            _array_path(root, record, validate_file=validate_files)


def _read_manifest(directory: Path | str, *, validate_files: bool = True) -> tuple[Path, dict[str, Any]]:
    root, manifest, _ = _read_manifest_snapshot(directory, validate_files=validate_files)
    return root, manifest


def _summary(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
    samples = [sample for sample in samples if not sample.get("excluded", False)]
    return {
        "samples": len(samples), "groups": len({sample["group_id"] for sample in samples}),
        "train_samples": sum(sample["split"] == "train" for sample in samples),
        "validation_samples": sum(sample["split"] == "validation" for sample in samples),
    }


def _warnings(manifest: dict[str, Any]) -> list[str]:
    warnings = [str(value) for value in manifest.get("warnings", [])]
    summary = _summary(manifest["samples"])
    if summary["groups"] < 2:
        warnings.append("Fewer than two independent photo groups are included; restore or add a separate group before training.")
    if not summary["train_samples"] or not summary["validation_samples"]:
        warnings.append("Training requires both training and validation samples; change the validation percentage or restore an independent photo group.")
    return list(dict.fromkeys(warnings))


def _registration_summary(sample: dict[str, Any]) -> dict[str, Any] | None:
    registration = sample.get("display_registration")
    if not isinstance(registration, dict):
        return None
    return {key: registration[key] for key in (
        "accepted", "reference_role", "reason", "display_shape", "stereo_shape",
        "heldout_median_error_pixels", "heldout_p90_error_pixels", "feature_match_count",
        "unique_feature_match_count", "calibration_recovered",
    ) if key in registration}


def review_dataset(directory: Path | str) -> dict[str, Any]:
    root, manifest, digest = _read_manifest_snapshot(directory, validate_files=False)
    from .dataset_edit import review_split_components
    from .training import training_target_eligibility
    from .display_training import _display_exclusion, display_target_eligibility
    from .dataset_recovery import generation_status
    management_groups = review_split_components(root, manifest)
    samples = []
    for sample in manifest["samples"]:
        chosen, training_label = _label(sample, "training")
        labels = [{"key": "training", "title": f"Selected training target: {_label_title(chosen, training_label)}", "units": training_label["units"]}]
        # The training choice is an alias, not another prediction. Identical
        # target/mask records do not deserve indistinguishable menu entries.
        seen = {(training_label["target"]["array_sha256"], json.dumps(training_label.get("valid_mask"), sort_keys=True), training_label["units"])}
        for key, title in LABEL_TITLES.items():
            if key == "training" or key not in sample:
                continue
            record = sample[key]
            signature = (record["target"]["array_sha256"], json.dumps(record.get("valid_mask"), sort_keys=True), record["units"])
            if signature in seen:
                continue
            seen.add(signature)
            labels.append({"key": key, "title": _label_title(key, record), "units": record["units"]})
        warnings = []
        if training_label["units"] != "meters":
            warnings.append("This target keeps its relative units; a display student trained on it cannot claim meter depth. Metric RAFT flow needs a separately accepted meter scale.")
        calibration = sample.get("pseudo_calibration")
        if isinstance(calibration, dict) and not calibration.get("accepted"):
            warnings.append(f"Meter-scale anchor was rejected: {calibration.get('reason', 'fit was not accepted')}")
        if chosen == "registered_display_teacher":
            warnings.append("The selected target uses approximate display-image registration; excluded regions remain invalid.")
            if training_label.get("reference_role") != "left":
                warnings.append("The selected display target is not aligned to the stereo left image required for RAFT training.")
        display_registration = _registration_summary(sample)
        if display_registration is not None and not display_registration.get("accepted"):
            warnings.append(f"Display alignment was rejected: {display_registration.get('reason', 'registration was not accepted')}. "
                            "The display image stays on its separate grid.")
        rejection = _display_exclusion(sample, "mixed")
        if rejection is not None:
            warnings.append(rejection["reason"])
        if not sample.get("calibration", {}).get("raft_stereo_ready"):
            warnings.append("Stereo calibration is not ready for RAFT training.")
        samples.append({
            "id": sample["id"], "source_path": sample["source_path"], "split": sample["split"],
            "excluded": sample.get("excluded", False), "included": not sample.get("excluded", False),
            "group_id": sample["group_id"], "requested_group": sample.get("requested_group"),
            "management_group_id": management_groups[sample["id"]],
            "training_target_choice": chosen, "labels": [] if sample.get("teacher_payload_removed") else labels, "warnings": warnings,
            "teacher_payload_removed": sample.get("teacher_payload_removed", False),
            "can_generate_display_teacher": isinstance(sample.get("display_rgb"), dict),
            "source_id": sample.get("source_id", sample.get("source_sha256", sample["source_path"])),
            "teacher_id": sample.get("teacher_id", sample.get("teacher_model", sample.get("teacher", {}).get("metadata", {}).get("model_id", "Teacher"))),
            "teacher_model": sample.get("teacher_model", sample.get("teacher", {}).get("metadata", {}).get("model_id", sample.get("teacher_id", ""))),
            "photo_metadata": sample.get("photo_metadata", {}),
            "rgb_reference": "display" if str(training_label.get("coordinate_reference", "")).startswith("display") else "spatial_left", "display_registration": display_registration,
            "training_ready": not sample.get("teacher_payload_removed", False) and rejection is None and bool(sample.get("calibration", {}).get("raft_stereo_ready")),
        })
    return {"dataset_path": str(root), "samples": samples, "summary": _summary(manifest["samples"]), "warnings": _warnings(manifest),
            "max_native_stereo_pixels": max((math.prod(sample["rgb"]["shape"][:2]) for sample in manifest["samples"]), default=0),
            "excluded_samples": sum(sample.get("excluded", False) for sample in manifest["samples"]),
            "manifest_sha256": digest, "edit_revision": manifest.get("edit_revision", 0),
            "pending_photos": manifest.get("pending_photos", []),
            "validation_fraction": manifest.get("validation_fraction"),
            "training_eligibility": display_target_eligibility(manifest),
            "raft_training_eligibility": training_target_eligibility(manifest),
            "training_eligibility_by_mode": {mode: display_target_eligibility(manifest, mode) for mode in ("auto", "distillation", "supervised", "mixed")},
            "raft_training_eligibility_by_mode": {mode: training_target_eligibility(manifest, mode) for mode in ("auto", "distillation", "supervised", "mixed")},
            "generation_state": manifest.get("generation_state", "complete"), "splits_provisional": bool(manifest.get("splits_provisional", False)),
            "generation_status": generation_status(root, manifest)}


def _verified_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _array_path(root, record)
    try:
        value = read_array(path, mmap_mode="r")
        if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
            raise DatasetError(f"Dataset array shape/dtype mismatch: {record['path']}")
        if sha256_file(path) != record["file_sha256"] or sha256_array(value) != record["array_sha256"]:
            raise DatasetError(f"Dataset checksum mismatch: {record['path']}")
    except (ValueError, OSError) as exc:
        raise DatasetError(f"Cannot read dataset array {record['path']}: {exc}") from exc
    return value


def _rgb_preview(rgb: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise DatasetError("Preview RGB must have three or four channels")
    value = rgb[:, :, :3]
    if value.dtype.kind == "u":
        bits = record.get("source_bit_depth")
        if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= value.dtype.itemsize * 8:
            bits = value.dtype.itemsize * 8
        limit = float((1 << bits) - 1)
    elif value.dtype.kind == "f":
        limit = 1.0
    else:
        raise DatasetError(f"Unsupported preview RGB dtype: {value.dtype}")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > limit):
        raise DatasetError("Preview RGB exceeds its declared nominal code-value range")
    # A fixed encoded bit range, never the image's observed minimum/maximum.
    return np.rint(value.astype(np.float64) * (255.0 / limit)).astype(np.uint8)


def preview_sample(
    directory: Path | str, sample_id: str, label: str, output_dir: Path | str, *, max_dimension: int = 1600,
    _display_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if label not in PREVIEW_LABELS:
        raise DatasetError(f"Unknown preview label: {label}")
    if not isinstance(max_dimension, int) or isinstance(max_dimension, bool) or max_dimension < 0:
        raise DatasetError("Preview maximum dimension must be a nonnegative integer; zero means full resolution")
    root, manifest = _read_manifest(directory, validate_files=False)
    sample = next((sample for sample in manifest["samples"] if sample["id"] == sample_id), None)
    if sample is None:
        raise DatasetError(f"Unknown dataset sample ID: {sample_id}")
    if sample.get("teacher_payload_removed"):
        raise DatasetError("This teacher map was discarded. Enable its teacher to regenerate the result.")
    resolved, target = _label(sample, label)
    rgb_key = "display_rgb" if resolved in {"display_teacher", "anchored_display_teacher", "display_metric_anchor"} or str(target.get("coordinate_reference", "")).startswith("display") else "right_rgb" if resolved == "registered_display_teacher" and target.get("reference_role") == "right" else "rgb"
    if rgb_key not in sample:
        raise DatasetError(f"Sample {sample_id} has no RGB image for {resolved}")
    depth = _verified_array(root, target["target"])
    rgb = _verified_array(root, sample[rgb_key])
    if depth.ndim != 2 or depth.dtype.kind != "f" or not depth.size or rgb.shape[:2] != depth.shape:
        raise DatasetError("Depth preview and RGB must have the exact same nonempty grid")
    valid = np.isfinite(depth) & (depth > 0)
    if "valid_mask" in target:
        mask = _verified_array(root, target["valid_mask"])
        if mask.shape != depth.shape or mask.dtype.kind != "b":
            raise DatasetError("Depth valid mask must be boolean on the exact target grid")
        valid &= mask
    minimum = float(np.min(depth[valid])) if valid.any() else None
    maximum = float(np.max(depth[valid])) if valid.any() else None
    height, width = depth.shape
    scale = min(1.0, max_dimension / max(height, width)) if max_dimension else 1.0
    preview_height, preview_width = max(1, round(height * scale)), max(1, round(width * scale))
    yy = np.minimum(height - 1, np.floor((np.arange(preview_height) + .5) * height / preview_height).astype(np.intp))
    xx = np.minimum(width - 1, np.floor((np.arange(preview_width) + .5) * width / preview_width).astype(np.intp))
    # Select the same exact pixels in both images; no interpolation through NaNs.
    small_depth = depth[np.ix_(yy, xx)].astype(np.float64)
    small_valid = valid[np.ix_(yy, xx)]
    small_rgb = _rgb_preview(rgb[np.ix_(yy, xx)], sample[rgb_key])
    gray = np.full(small_depth.shape, 127, dtype=np.uint8)
    contrast_min, contrast_max = _display_range if _display_range is not None else (minimum, maximum)
    if contrast_min is not None and contrast_max > contrast_min:
        fraction = (small_depth[small_valid] - contrast_min) / (contrast_max - contrast_min)
        if target["units"] != "relative_inverse_depth":
            fraction = 1.0 - fraction
        gray[small_valid] = np.rint(np.clip(fraction, 0.0, 1.0) * 255).astype(np.uint8)
    depth_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    depth_rgb[~small_valid] = [255, 0, 255]
    destination = Path(output_dir).expanduser().resolve()
    if destination.is_relative_to(root):
        raise DatasetError("Preview output must be outside the precision-preserved dataset")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="sample-preview-", dir=destination))
    try:
        rgb_path, depth_path = temporary / "rgb.png", temporary / "depth.png"
        attributes = {"ipdePreviewOnly": "true; original NPY arrays untouched", "ipdeDepthLegend": "near white; far black; invalid magenta"}
        write_png(rgb_path, small_rgb, attributes={"ipdePreviewOnly": attributes["ipdePreviewOnly"]})
        write_png(depth_path, depth_rgb, attributes=attributes)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "sample_id": sample_id, "label": label, "resolved_label": resolved,
        "label_title": f"Selected training target: {_label_title(resolved, target)}" if label == "training" else _label_title(resolved, target),
        "units": target["units"], "rgb_preview_path": str(rgb_path), "depth_preview_path": str(depth_path),
        "width": width, "height": height, "preview_width": preview_width, "preview_height": preview_height,
        "valid_fraction": float(valid.mean()), "min": minimum, "max": maximum,
        "training_target_choice": sample.get("training_target_choice", "teacher"),
        "rgb_reference": {"display_rgb": "display", "right_rgb": "spatial_right", "rgb": "spatial_left"}[rgb_key],
        "display_registration": _registration_summary(sample),
        "display_range": [contrast_min, contrast_max],
        "surface": _surface_preview(depth, valid, target["units"]),
        "legend": "Near white; far black; invalid magenta. Contrast is for viewing only and is scaled separately for each label.",
        "preview_only": True,
    }


def _surface_preview(depth: np.ndarray, valid: np.ndarray, units: str) -> dict[str, Any]:
    """Small display-only height field for interactive lighting, never a label."""
    rows, columns = min(64, depth.shape[0]), min(64, depth.shape[1])
    yy = np.linspace(0, depth.shape[0] - 1, rows).astype(np.intp)
    xx = np.linspace(0, depth.shape[1] - 1, columns).astype(np.intp)
    small = depth[np.ix_(yy, xx)].astype(np.float64)
    mask = valid[np.ix_(yy, xx)]
    heights = np.zeros(small.shape, np.float64)
    if mask.any():
        values = small[mask]
        low, high = float(values.min()), float(values.max())
        if high > low:
            heights[mask] = (values - low) / (high - low)
            if units != "relative_inverse_depth":
                heights[mask] = 1 - heights[mask]
    return {"rows": rows, "columns": columns, "heights": heights.ravel().tolist(), "valid": mask.ravel().tolist(),
            "legend": "Display-only normalized relief. Drag to rotate; lighting shows local slope. This is not reconstructed geometry."}


def compare_samples(
    directory: Path | str, sample_ids: Sequence[str], output_dir: Path | str, *, max_dimension: int = 0,
) -> dict[str, Any]:
    """Compare teacher variants for one photo on a shared native stereo grid."""
    if not 2 <= len(sample_ids) <= 3 or len(set(sample_ids)) != len(sample_ids):
        raise DatasetError("Select two or three distinct teacher entries for comparison")
    root, manifest = _read_manifest(directory, validate_files=False)
    by_id = {sample["id"]: sample for sample in manifest["samples"]}
    if any(identifier not in by_id for identifier in sample_ids):
        raise DatasetError("Unknown sample in teacher comparison")
    selected = [by_id[identifier] for identifier in sample_ids]
    if any(sample.get("teacher_payload_removed") for sample in selected):
        raise DatasetError("A compared teacher map was discarded. Enable that teacher to regenerate it first.")
    source_ids = {sample.get("source_id", sample.get("source_sha256", sample["source_path"])) for sample in selected}
    if len(source_ids) != 1:
        raise DatasetError("Teacher comparison requires entries from the same source photo")
    targets = [_label(sample, "training")[1] for sample in selected]
    arrays = [_verified_array(root, target["target"]) for target in targets]
    if any(array.ndim != 2 or array.shape != arrays[0].shape for array in arrays):
        raise DatasetError("Teacher comparison requires the same two-dimensional stereo grid")
    masks = []
    for array, target in zip(arrays, targets):
        valid = np.isfinite(array) & (array > 0)
        if "valid_mask" in target:
            mask = _verified_array(root, target["valid_mask"])
            if mask.shape != array.shape or mask.dtype.kind != "b":
                raise DatasetError("Teacher comparison valid mask must be boolean on the exact target grid")
            valid &= mask
        masks.append(valid)
    metric = all(target["units"] == "meters" for target in targets)
    display_range = None
    if metric and any(mask.any() for mask in masks):
        display_range = (min(float(array[mask].min()) for array, mask in zip(arrays, masks) if mask.any()),
                         max(float(array[mask].max()) for array, mask in zip(arrays, masks) if mask.any()))
    previews = [preview_sample(root, sample["id"], "training", output_dir, max_dimension=max_dimension,
                               _display_range=display_range) for sample in selected]
    for preview, sample in zip(previews, selected):
        preview["teacher_id"] = sample.get("teacher_id", sample.get("teacher", {}).get("metadata", {}).get("model_id", "Teacher"))
    both = masks[0] & masks[1]
    if metric:
        difference = np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))
        legend = "Absolute difference between the first two teachers in estimated meters. Black means agreement; bright red means disagreement; magenta means invalid."
        mode = "estimated-meters"
    else:
        relief = []
        for array, mask, target in zip(arrays[:2], masks[:2], targets[:2]):
            mapped = np.zeros(array.shape, np.float64)
            if mask.any():
                low, high = float(array[mask].min()), float(array[mask].max())
                if high > low:
                    mapped[mask] = (array[mask] - low) / (high - low)
                    if target["units"] != "relative_inverse_depth":
                        mapped[mask] = 1 - mapped[mask]
            relief.append(mapped)
        difference = np.abs(relief[0] - relief[1])
        legend = "Display-only difference of separately normalized relative relief for the first two teachers. It compares shapes, not metric distance. Magenta means invalid."
        mode = "relative-display-relief"
    maximum = float(difference[both].max()) if both.any() else 0.0
    difference_rgb = np.zeros((*difference.shape, 3), np.uint8)
    if maximum > 0:
        difference_rgb[:, :, 0][both] = np.rint(difference[both] / maximum * 255).astype(np.uint8)
    difference_rgb[~both] = [255, 0, 255]
    if max_dimension:
        step = max(1, int(np.ceil(max(difference.shape) / max_dimension)))
        difference_rgb = difference_rgb[::step, ::step]
    destination = Path(previews[0]["depth_preview_path"]).parent / "difference.png"
    write_png(destination, difference_rgb, attributes={"ipdePreviewOnly": "true; original NPY arrays untouched"})
    return {"samples": previews, "difference_preview_path": str(destination),
            "comparison": {"mode": mode, "legend": legend, "shared_metric_range": display_range,
                           "valid_fraction": float(both.mean()),
                           "mean_absolute_difference": float(difference[both].mean()) if both.any() else None,
                           "maximum_difference": maximum}}


def _publish_new_directory(temporary: Path, destination: Path) -> None:
    """Atomically publish, with an OS-level refusal to replace a raced-in path."""
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(temporary), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise DatasetError("This platform does not support atomic dataset publication without replacement")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif os.name == "nt":
        os.rename(temporary, destination)  # Windows rename refuses existing destinations.
        return
    else:
        raise DatasetError("This platform does not support atomic dataset publication without replacement")
    if result:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
        raise OSError(error, os.strerror(error), str(destination))


def curate_dataset(directory: Path | str, keep_ids: Sequence[str], output_dir: Path | str, *, workers: int | None = None) -> dict[str, Any]:
    try:
        worker_count = resolve_workers(workers)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    root, manifest = _read_manifest(directory, validate_files=False)
    if manifest.get("generation_state") == "generating" or manifest.get("splits_provisional"):
        raise DatasetError("Wait for dataset generation to finish before saving a reviewed copy; current split assignments are provisional")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if destination.is_relative_to(root):
        raise DatasetError("Curated dataset must be outside the original dataset")
    if not keep_ids:
        raise DatasetError("Keep at least one sample; all samples cannot be excluded")
    if len(set(keep_ids)) != len(keep_ids):
        raise DatasetError("Duplicate kept sample IDs are not allowed")
    all_ids = {sample["id"] for sample in manifest["samples"]}
    unknown = set(keep_ids) - all_ids
    if unknown:
        raise DatasetError(f"Unknown dataset sample IDs: {', '.join(sorted(unknown))}")
    # Curation is the durable action: validate all source arrays/provenance first.
    if load_dataset(root, verify=True, workers=worker_count) != manifest:
        raise DatasetError("Source dataset manifest changed before curation")
    source_manifest_hash = sha256_file(root / "dataset.json")
    kept = set(keep_ids)
    curated = copy.deepcopy(manifest)
    curated["samples"] = [sample for sample in curated["samples"] if sample["id"] in kept]
    curated["summary"] = _summary(curated["samples"])
    curated["group_ids"] = sorted({sample["group_id"] for sample in curated["samples"]})
    curated["explicit_scene_groups"] = bool(manifest.get("explicit_scene_groups") and manifest.get("grouping_semantics") == "scene"
                                              and all(sample.get("requested_group") for sample in curated["samples"]))
    excluded = [sample["id"] for sample in manifest["samples"] if sample["id"] not in kept]
    curated["curation"] = {
        "source_dataset_path": str(root), "source_manifest_sha256": source_manifest_hash,
        "kept_sample_ids": [sample["id"] for sample in curated["samples"]], "excluded_sample_ids": excluded,
        "excluded_samples": [{"id": sample["id"], "source_path": sample["source_path"]} for sample in manifest["samples"] if sample["id"] not in kept],
        "split_assignments_preserved": True, "array_bytes_preserved": True,
    }
    curated["warnings"] = _warnings(curated)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        records: dict[str, dict[str, Any]] = {}
        for sample in curated["samples"]:
            for record in _array_records(sample):
                records.setdefault(record["path"], record)
        def copy_record(record: dict[str, Any]) -> None:
            source = _array_path(root, record)
            target = temporary / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != record["file_sha256"]:
                raise DatasetError(f"Dataset checksum mismatch while copying: {record['path']}")
        for _ in ordered_map(copy_record, records.values(), workers=worker_count):
            pass
        if sha256_file(root / "dataset.json") != source_manifest_hash:
            raise DatasetError("Source dataset manifest changed during curation")
        (temporary / "dataset.json").write_text(json.dumps(curated, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _publish_new_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"dataset_path": str(destination), "summary": curated["summary"], "warnings": curated["warnings"], "curation": curated["curation"]}
