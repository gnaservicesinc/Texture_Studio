"""Local experimental RAFT-Stereo fine-tuning with metric teacher-derived flow."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dataset import DatasetError, load_dataset, teacher_depth_to_flow
from .array_storage import read_array
from .concurrency import memory_limited_workers, ordered_map, resolve_workers
from .formats import sha256_array, sha256_file
from .spatial import (
    RaftStereoError, RaftStereoOptions, _checkpoint_bytes, _model_configuration,
    _select_device, correspondence_validity, register_stereo_rows,
    resolve_raft_resources, run_raft_stereo, stereo_photometric_support,
)


class TrainingError(RuntimeError):
    """A local RAFT experiment could not be run or verified."""


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 10
    steps_per_epoch: int = 16
    patch_size: int = 128
    learning_rate: float = 1e-5
    device: str = "auto"
    seed: int = 0
    mode: str = "distillation"
    raft_root: Path | None = None
    raft_model: Path | None = None
    raft_model_member: str | None = None
    iterations: int = 4
    train_scope: str = "update"
    require_photometric_support: bool = False
    cache_samples: int = 2
    skip_incompatible_targets: bool = True
    workers: int | None = None


def _selected_target(sample: Mapping[str, Any], mode: str) -> str:
    return "reference" if mode == "supervised" else sample.get("training_target_choice", "teacher")


def _target_exclusion(sample: Mapping[str, Any], mode: str) -> dict[str, Any] | None:
    choice = _selected_target(sample, mode)
    allowed = {"reference"} if mode == "supervised" else {"teacher", "registered_display_teacher", "anchored_teacher"}
    if mode == "mixed":
        allowed.add("reference")
    label = sample.get(choice) if isinstance(choice, str) else None
    reason = None
    if not isinstance(choice, str) or choice not in allowed or not isinstance(label, Mapping):
        reason = "Selected training target is unavailable or incompatible with the training mode"
    elif choice == "registered_display_teacher" and label.get("reference_role") != "left":
        reason = "Selected display teacher is not registered to the LEFT stereo grid"
    elif label.get("units") != "meters":
        reason = "Selected target has no accepted meter scale; relative values remain unchanged"
    elif choice != "reference":
        metadata = label.get("metadata", {})
        checkpoint = metadata.get("checkpoint_sha256") if isinstance(metadata, Mapping) else None
        if not isinstance(checkpoint, str) or not checkpoint:
            reason = "Selected teacher target does not identify its checkpoint"
    if reason is None:
        return None
    return {"sample_id": sample.get("id", ""), "source_path": sample.get("source_path", ""),
        "target_choice": choice, "units": label.get("units") if isinstance(label, Mapping) else None, "reason": reason}


def training_target_eligibility(manifest: Mapping[str, Any], mode: str = "auto") -> dict[str, Any]:
    """Inspect selected target metadata without reading or changing any arrays.

    A rejected relative teacher is never replaced by another stored label.
    Geometry/support and dataset integrity are checked by training preflight.
    """
    if mode not in {"auto", "distillation", "supervised", "mixed"}:
        raise TrainingError("Use auto/distillation/supervised/mixed training mode")
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or any(not isinstance(sample, Mapping) for sample in samples):
        raise TrainingError("Dataset samples must be a list of sample records")
    if any("excluded" in sample and not isinstance(sample["excluded"], bool) for sample in samples):
        raise TrainingError("Dataset excluded membership must be boolean")
    removed_count = sum(sample.get("excluded", False) for sample in samples)
    samples = [sample for sample in samples if not sample.get("excluded", False)]
    if mode == "auto":
        choices = {choice for sample in samples if isinstance(choice := sample.get("training_target_choice", "teacher"), str)}
        mode = "supervised" if choices == {"reference"} else "mixed" if "reference" in choices else "distillation"
    excluded, eligible = [], []
    for sample in samples:
        rejection = _target_exclusion(sample, mode)
        if rejection is None:
            eligible.append(sample)
        else:
            excluded.append(rejection)
    train_count = sum(sample.get("split") == "train" for sample in eligible)
    validation_count = sum(sample.get("split") == "validation" for sample in eligible)
    trainable = bool(train_count and validation_count) and manifest.get("generation_state", "complete") == "complete" and not manifest.get("splits_provisional", False)
    reason = ""
    if manifest.get("generation_state", "complete") != "complete" or manifest.get("splits_provisional", False):
        reason = "Finish dataset generation before training"
    elif not samples:
        reason = "No photos are included; restore or add photos before training"
    elif not eligible:
        reason = "No usable meter-scale training targets are included; review the depth targets or generate them with a metric teacher"
    elif not train_count:
        reason = "No training targets are included; reduce the validation percentage or move an independent photo group to training"
    elif not validation_count:
        reason = "No validation targets are included; increase the validation percentage or set aside an independent photo group"
    return {"mode": mode, "sample_count": len(samples), "removed_count": removed_count, "eligible_count": len(eligible),
        "excluded_count": len(excluded), "train_count": sum(sample.get("split") == "train" for sample in eligible),
        "validation_count": validation_count, "excluded": excluded, "trainable": trainable, "reason": reason}


def _check_splits(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = [sample for sample in samples if not sample.get("excluded", False)]
    train = [s for s in samples if s.get("split") == "train"]
    validation = [s for s in samples if s.get("split") == "validation"]
    if not train or not validation or len(train) + len(validation) != len(samples):
        raise TrainingError("Training requires train and held-out validation groups; add another independent scene")
    seen: dict[str, str] = {}
    for sample in samples:
        tokens = [f"group:{sample['group_id']}", f"source:{sample['source_sha256']}", f"rgb:{sample['rgb']['array_sha256']}"]
        if sample.get("requested_group"):
            tokens.append(f"scene:{sample['requested_group']}")
        tokens.extend(f"burst:{burst}" for burst in sample.get("burst_ids", []))
        for token in tokens:
            if token in seen and seen[token] != sample["split"]:
                raise TrainingError("Training/validation leakage: scene, burst, or duplicate photo occurs in both splits")
            seen[token] = sample["split"]
    return train, validation


def sequence_loss(torch: Any, predictions: list[Any], target: Any, valid: Any) -> Any:
    """Keep the imported validity mask, including one-iteration training."""
    if not predictions or target.ndim != 4 or target.shape[1] != 1 or valid.shape != target.shape:
        raise TrainingError("RAFT loss requires Bx1xHxW target/mask and at least one prediction")
    accepted = valid & torch.isfinite(target) & (target.abs() < 700)
    if not bool(accepted.any()):
        raise TrainingError("Training patch has no supported teacher correspondence pixels")
    loss = None
    gamma = 0.9 ** (15 / (len(predictions) - 1)) if len(predictions) > 1 else 1.0
    for index, prediction in enumerate(predictions):
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
            raise TrainingError("RAFT training prediction has incorrect shape or nonfinite values")
        term = gamma ** (len(predictions) - index - 1) * (prediction[accepted] - target[accepted]).abs().mean()
        loss = term if loss is None else loss + term
    return loss


def _visible_teacher_pixels(depth: np.ndarray, flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Teacher-derived forward z-buffer check; this is estimated visibility."""
    h, w = depth.shape
    xr = np.arange(w, dtype=np.float32)[None, :] + flow
    nearest = np.clip(np.rint(np.where(valid, xr, 0)).astype(np.intp), 0, w - 1)
    rows = np.broadcast_to(np.arange(h)[:, None], depth.shape)
    buffer = np.full(depth.shape, np.inf, dtype=np.float32)
    np.minimum.at(buffer, (rows[valid], nearest[valid]), depth[valid])
    return valid & (depth <= buffer[rows, nearest] * np.float32(1.01) + np.float32(1e-3))


def _load_sample(root: Path, sample: dict[str, Any], options: TrainingOptions) -> dict[str, Any]:
    label_key = "reference" if options.mode == "supervised" else sample.get("training_target_choice", "teacher")
    if label_key not in {"reference", "teacher", "registered_display_teacher", "anchored_teacher"} or label_key not in sample:
        raise TrainingError("Dataset has an unsupported training target choice")
    label = sample[label_key]
    if label_key == "registered_display_teacher" and label.get("reference_role") != "left":
        raise TrainingError("A RIGHT display teacher requires explicit camera reprojection; it cannot supervise the LEFT pair directly")
    if label["units"] != "meters":
        raise TrainingError("RAFT pseudo-labels require meters; relative outputs need an explicit accepted metric-anchor estimate or independently measured calibration")
    left = read_array(root / sample["rgb"]["path"])
    raw_right = read_array(root / sample["right_rgb"]["path"])
    depth = read_array(root / label["target"]["path"])
    if left.dtype != np.uint8 or raw_right.dtype != np.uint8 or left.shape != raw_right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise TrainingError("Pretrained RAFT requires matching native uint8 RGB views; higher-bit raw arrays remain untouched")
    try:
        flow, valid, geometry = teacher_depth_to_flow(depth, sample["calibration"])
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    if label_key == "reference":
        geometry = {**geometry, "label_kind": "user_declared_measured_reference", "precision_note": "Reference accuracy and calibration require independent verification"}
    right, right_valid, registration = register_stereo_rows(left, raw_right)
    valid &= correspondence_validity(flow, right_valid)
    recorded_valid = read_array(root / label["valid_mask"]["path"])
    if recorded_valid.shape != valid.shape or recorded_valid.dtype != np.bool_:
        raise TrainingError("Dataset target validity mask has the wrong shape or dtype")
    valid &= recorded_valid
    valid = _visible_teacher_pixels(depth, flow, valid)
    photometric, evidence = stereo_photometric_support(left, right, -flow)
    photometric &= valid
    if options.require_photometric_support:
        valid &= photometric
    if not valid.any():
        raise TrainingError(f"{sample['id']}: no visible in-image teacher correspondences remain")
    return {"left": left, "right": right, "flow": flow, "valid": valid, "details": {
        "sample_id": sample["id"], "geometry": geometry,
        "target_choice": label_key,
        "right_inference_sha256": sha256_array(right), "right_registration": registration,
        "accepted_teacher_pixels": int(valid.sum()), "photometrically_supported_teacher_pixels": int(photometric.sum()),
        "photometric_filter_required": options.require_photometric_support, "photometric_support": evidence,
        "visibility_note": "Reference-derived visibility depends on the supplied measurement and calibration" if label_key == "reference" else "Teacher-derived z-buffer visibility is estimated, not independently measured",
    }}


class _SamplePool(Sequence):
    """Keep only a bounded number of decoded training targets in memory."""
    def __init__(self, root: Path, samples: list[dict[str, Any]], options: TrainingOptions):
        self.root, self.samples, self.options = root, samples, options
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.details: dict[int, dict[str, Any]] = {}
        self.crop_origins: dict[int, tuple[int, int]] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < 0: index += len(self.samples)
        if not 0 <= index < len(self.samples): raise IndexError(index)
        if index not in self.cache:
            value = _load_sample(self.root, self.samples[index], self.options)
            self.details[index] = value["details"]
            self.cache[index] = value
            while len(self.cache) > self.options.cache_samples:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def prepare(self, role: str, progress_callback: Callable[[dict[str, Any]], None] | None = None,
                skipped_callback: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
        retained, excluded = [], []
        # Native row registration and scientific decoding release the GIL.
        # Keep worker-owned decoded values separate from the caller-owned LRU,
        # details, exclusions and progress callbacks.
        largest_pixels = max((math.prod(sample["rgb"].get("shape", [1])[:2]) for sample in self.samples
                              if isinstance(sample.get("rgb"), Mapping)), default=1)
        count = memory_limited_workers(self.options.workers, largest_pixels * 64,
                                       budget_bytes=1024 * 1024**2)

        def prepare_one(index: int) -> Any:
            origin = None
            try:
                value = _load_sample(self.root, self.samples[index], self.options)
                if role == "training":
                    origin = _training_crop_origin(value, self.options.patch_size)
                    if origin is None:
                        raise TrainingError(f"No supported teacher correspondence fits the requested {self.options.patch_size}-pixel training crop")
                elif not any(_patch(value, y, x, self.options.patch_size)[3].any()
                             for y, x in _validation_crop_origins(value, self.options.patch_size)):
                    raise TrainingError("Fixed validation crops contain no supported teacher correspondences")
            except TrainingError as exc:
                return index, None, None, exc
            return index, value, origin, None

        if progress_callback is not None:
            progress_callback({"role": role, "processed": 0, "total": len(self), "status": "running", "workers": count})
        for index, value, origin, error in ordered_map(prepare_one, range(len(self)), workers=count):
            if error is not None:
                if not self.options.skip_incompatible_targets:
                    raise error
                sample = self.samples[index]
                choice = _selected_target(sample, self.options.mode)
                rejection = {"sample_id": sample["id"], "source_path": sample.get("source_path", ""),
                    "target_choice": choice, "units": sample[choice].get("units"), "reason": str(error)}
                excluded.append(rejection)
                if skipped_callback is not None:
                    skipped_callback(rejection)
            else:
                retained.append(index)
                self.details[index] = value["details"]
                self.cache[index] = value
                if origin is not None:
                    self.crop_origins[index] = origin
                while len(self.cache) > self.options.cache_samples:
                    self.cache.popitem(last=False)
            print(f"Preparing {role} target {index + 1}/{len(self)}", file=sys.stderr, flush=True)
            if progress_callback is not None:
                progress_callback({"role": role, "processed": index + 1, "total": len(self),
                    "sample_id": self.samples[index]["id"], "status": "running"})
        # Keep already decoded values under their new indices without another
        # decode or an unbounded second collection of scientific arrays.
        remap = {previous: current for current, previous in enumerate(retained)}
        self.samples = [self.samples[index] for index in retained]
        self.cache = OrderedDict((remap[index], value) for index, value in self.cache.items() if index in remap)
        self.details = {remap[index]: value for index, value in self.details.items() if index in remap}
        self.crop_origins = {remap[index]: value for index, value in self.crop_origins.items() if index in remap}
        return excluded

    def preprocessing(self):
        return [self.details[index] for index in sorted(self.details)]


def _patch(sample: Mapping[str, Any], y: int, x: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = sample["flow"].shape
    ph, pw = min(size, h), min(size, w)
    left = np.array(sample["left"][y:y + ph, x:x + pw], copy=True)
    right = np.array(sample["right"][y:y + ph, x:x + pw], copy=True)
    flow = np.array(sample["flow"][y:y + ph, x:x + pw], copy=True)
    valid = np.array(sample["valid"][y:y + ph, x:x + pw], copy=True)
    # Same crop origins preserve signed flow; out-of-crop matches are excluded.
    xr = np.arange(pw, dtype=np.float32)[None, :] + flow
    valid &= np.isfinite(xr) & (xr >= 0) & (xr <= pw - 1)
    flow[~valid] = 0  # A new loss tensor only; saved scientific targets retain NaN.
    return left, right, flow, valid


def _training_crop_origin(sample: Mapping[str, Any], size: int) -> tuple[int, int] | None:
    """Find one supported native crop without random draws or modifying arrays.

    A full-image match fits a crop only when both corresponding x positions
    lie inside its width. Scan one row at a time to keep temporary storage
    bounded, then check the exact patch/loss validity predicate.
    """
    h, w = sample["flow"].shape
    ph, pw = min(size, h), min(size, w)
    for y in range(h):
        flow = sample["flow"][y]
        supported = sample["valid"][y] & np.isfinite(flow) & (np.abs(flow) < 700) & (np.abs(flow) <= pw - 1)
        for x in np.flatnonzero(supported):
            origin_y = min(y, h - ph)
            origin_x = int(np.clip(math.floor(min(float(x), float(x) + float(flow[x]))), 0, w - pw))
            patch = _patch(sample, origin_y, origin_x, size)
            accepted = patch[3] & np.isfinite(patch[2]) & (np.abs(patch[2]) < 700)
            if accepted.any():
                return origin_y, origin_x
    return None


def _validation_crop_origins(sample: Mapping[str, Any], size: int) -> list[tuple[int, int]]:
    h, w = sample["flow"].shape
    my, mx = max(0, h - size), max(0, w - size)
    return sorted({(0, 0), (my, mx), (my // 2, mx // 2)})


def _tensors(torch: Any, device: str, padder_class: Any, patch: Any) -> tuple[Any, Any, Any, Any, Any]:
    left, right, flow, valid = patch
    lt = torch.from_numpy(left).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    rt = torch.from_numpy(right).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    ft = torch.from_numpy(flow[None, None]).to(device)
    vt = torch.from_numpy(valid[None, None]).to(device)
    padder = padder_class(lt.shape, divis_by=32)
    lt, rt = padder.pad(lt, rt)
    return lt, rt, ft, vt, padder


def _evaluate(torch: Any, model: Any, device: str, padder_class: Any, samples: Sequence[dict[str, Any]], options: TrainingOptions, *,
              progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    errors, patches = [], []
    model.eval()
    with torch.inference_mode():
        for index in range(len(samples)):
            if progress_callback is not None:
                progress_callback({"processed": index, "total": len(samples), "sample_index": index,
                    "evaluated_patches": len(patches), "status": "running"})
            sample = samples[index]
            crop_origins = _validation_crop_origins(sample, options.patch_size)
            for crop_index, (y, x) in enumerate(crop_origins):
                patch = _patch(sample, y, x, options.patch_size)
                if not patch[3].any():
                    continue
                if progress_callback is not None:
                    progress_callback({"processed": index, "total": len(samples), "sample_index": index,
                        "sample_patch": crop_index + 1, "sample_patches": len(crop_origins),
                        "evaluated_patches": len(patches), "status": "running"})
                lt, rt, ft, vt, padder = _tensors(torch, device, padder_class, patch)
                _, prediction = model(lt, rt, iters=options.iterations, test_mode=True)
                difference = (padder.unpad(prediction) - ft).abs()[vt].cpu().numpy().astype(np.float64)
                if not np.isfinite(difference).all():
                    raise TrainingError("RAFT produced nonfinite validation predictions")
                errors.append(difference)
                patches.append({"sample_index": index, "x": x, "y": y, "width": patch[0].shape[1], "height": patch[0].shape[0]})
                if progress_callback is not None:
                    progress_callback({"processed": index, "total": len(samples), "sample_index": index,
                        "sample_patch": crop_index + 1, "sample_patches": len(crop_origins),
                        "evaluated_patches": len(patches), "status": "running"})
            if progress_callback is not None:
                progress_callback({"processed": index + 1, "total": len(samples), "sample_index": index,
                    "evaluated_patches": len(patches), "status": "running"})
    if not errors:
        raise TrainingError("Held-out crops lack supported correspondences; enlarge patch_size or add a scene")
    values = np.concatenate(errors)
    result = {"mean_absolute_flow_error_pixels": float(values.mean()), "within_one_pixel_fraction": float(np.mean(values < 1)),
        "within_three_pixels_fraction": float(np.mean(values < 3)), "evaluated_pixels_including_overlapping_patches": int(values.size),
        "scope": "Fixed native-resolution stereo crops; overlapping pixels may count more than once", "patches": patches,
        "reference": "teacher-derived flow pseudo-label agreement" if options.mode == "distillation" else "user-supplied measured meter-depth references" if options.mode == "supervised" else "mixed measured-reference and teacher pseudo-label agreement"}
    if progress_callback is not None:
        progress_callback({"processed": len(samples), "total": len(samples), "evaluated_patches": len(patches),
            "mean_absolute_flow_error_pixels": result["mean_absolute_flow_error_pixels"], "status": "finished"})
    return result


def train_dataset(dataset_dir: Path | str, checkpoint_path: Path | str, options: TrainingOptions | None = None, *,
                  progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Fine-tune real RAFT weights and report stages and completed optimizer steps.

    Progress contains no timing estimates. Validation counters count samples;
    completed_steps counts only successful optimizer updates, independently of
    setup, validation, and checkpoint publication.
    """
    options = options or TrainingOptions()
    try:
        worker_count = resolve_workers(options.workers)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    if options.mode not in {"auto", "distillation", "supervised", "mixed"} or options.train_scope not in {"update", "full"}:
        raise TrainingError("Use auto/distillation/supervised/mixed mode and update/full train_scope")
    if min(options.epochs, options.steps_per_epoch) < 1 or options.patch_size < 64 or options.patch_size % 32:
        raise TrainingError("Epochs/steps must be positive; native patch_size must be a multiple of 32 and at least 64")
    if not isinstance(options.cache_samples, int) or isinstance(options.cache_samples, bool) or not 1 <= options.cache_samples <= 64:
        raise TrainingError("cache_samples must be an integer from 1 to 64")
    if not isinstance(options.skip_incompatible_targets, bool):
        raise TrainingError("skip_incompatible_targets must be boolean")
    if not 1 <= options.iterations <= 256 or not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise TrainingError("RAFT iterations must be 1..256 and learning_rate finite and positive")
    progress_state = {"epoch": 0, "epochs": options.epochs, "step": 0,
        "steps_per_epoch": options.steps_per_epoch, "completed_steps": 0,
        "total_steps": options.epochs * options.steps_per_epoch}

    def progress(stage: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({"phase": "training_progress", "stage": stage, **progress_state, **details})

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    report_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if checkpoint.exists() or report_path.exists():
        raise TrainingError("Checkpoint/report already exists; choose a new checkpoint path")
    root = Path(dataset_dir).expanduser().resolve()
    if checkpoint.is_relative_to(root):
        raise TrainingError("Checkpoint output must be outside the input dataset directory")
    progress("checking_dataset", status="started", dataset_path=str(root))
    try:
        from .dataset_review import _read_manifest_snapshot
        _, manifest, manifest_digest = _read_manifest_snapshot(root, validate_files=False)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    if manifest.get("generation_state", "complete") != "complete" or manifest.get("splits_provisional"):
        raise TrainingError("Finish dataset generation before training; streamed split assignments are provisional")
    # Verify every original split before filtering. An incompatible label may
    # not conceal a duplicate, burst, or scene crossing the held-out boundary.
    _check_splits(manifest["samples"])
    eligibility = training_target_eligibility(manifest, options.mode)
    options = replace(options, mode=eligibility["mode"])
    excluded = list(eligibility["excluded"])
    progress_state.update(sample_count=eligibility["sample_count"], eligible_count=eligibility["eligible_count"],
        excluded_count=eligibility["excluded_count"], train_count=eligibility["train_count"],
        validation_count=eligibility["validation_count"])
    if excluded and not options.skip_incompatible_targets:
        raise TrainingError(f"{excluded[0]['sample_id']}: {excluded[0]['reason']}")
    try:
        load_dataset(root, include_excluded=False, workers=worker_count, snapshot=manifest)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    included = [sample for sample in manifest["samples"] if not sample.get("excluded", False)]
    progress("filtering_targets", status="started", processed=0, total=len(included))
    for rejection in excluded:
        progress("skipped_sample", status="finished", **rejection)
    eligible = [sample for sample in included if _target_exclusion(sample, options.mode) is None]
    progress("filtering_targets", status="finished", processed=len(included),
        total=len(included), eligible_count=len(eligible), excluded_count=len(excluded))
    if not any(sample.get("split") == "train" for sample in eligible) or not any(sample.get("split") == "validation" for sample in eligible):
        raise TrainingError("No usable training and held-out validation split remains after incompatible targets were skipped")
    train, validation = _check_splits(eligible)
    # Each reviewed teacher variant is a separate target on the same source
    # grid. Splits keep the source together; mixed teachers need not share weights.
    train_arrays = _SamplePool(root, train, options)
    validation_arrays = _SamplePool(root, validation, options)
    total_targets = len(train) + len(validation)
    progress("preparing_targets", status="started", processed=0, total=total_targets)
    excluded.extend(train_arrays.prepare("training", lambda event: progress("preparing_targets", **{**event, "total": total_targets}),
        lambda rejection: progress("skipped_sample", status="finished", **rejection)))
    excluded.extend(validation_arrays.prepare("validation", lambda event: progress("preparing_targets",
        **{**event, "processed": len(train) + event["processed"], "total": total_targets}),
        lambda rejection: progress("skipped_sample", status="finished", **rejection)))
    progress("preparing_targets", status="finished", processed=total_targets, total=total_targets)
    train, validation = train_arrays.samples, validation_arrays.samples
    if not train or not validation:
        raise TrainingError("No usable training and held-out validation split remains after targets without valid geometry/support were skipped")
    _check_splits(train + validation)
    usable_records = {id(sample) for sample in train + validation}
    eligible = [sample for sample in eligible if id(sample) in usable_records]
    progress_state.update(eligible_count=len(eligible), excluded_count=len(excluded),
        train_count=len(train), validation_count=len(validation))
    chosen = [_selected_target(sample, options.mode) for sample in eligible]
    teachers = {s[key].get("metadata", {}).get("checkpoint_sha256") for s, key in zip(eligible, chosen) if key != "reference"}
    teachers.discard(None)
    metric_anchors = {s[key].get("metadata", {}).get("metric_anchor_checkpoint_sha256") for s, key in zip(eligible, chosen) if key != "reference"}
    intentional_mixture = bool(manifest.get("collection")) or len({s.get("teacher_id") for s in eligible if s.get("teacher_id")}) > 1
    if options.mode == "distillation" and not intentional_mixture:
        if len(teachers) != 1:
            raise TrainingError("Distillation requires one consistent teacher checkpoint, or an explicitly assembled multi-teacher collection")
        if len(metric_anchors) != 1:
            raise TrainingError("Distillation requires one consistent explicit metric-anchor checkpoint, or an assembled collection")
    if options.mode == "supervised":
        teachers.clear()
        metric_anchors.clear()
    print(f"RAFT experiment: {len(train)} training capture(s), {len(validation)} held-out capture(s)", file=sys.stderr, flush=True)
    progress("model_setup", status="started")
    raft_options = RaftStereoOptions(root=options.raft_root, model=options.raft_model, model_member=options.raft_model_member, device=options.device, iterations=options.iterations)
    try:
        raft_root, original_model, member = resolve_raft_resources(raft_options)
        checkpoint_bytes, checkpoint_name = _checkpoint_bytes(original_model, member)
        import torch
        device = _select_device(torch, options.device)
    except (RaftStereoError, ImportError) as exc:
        raise TrainingError(str(exc)) from exc
    root_text = str(raft_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder
    except Exception as exc:
        raise TrainingError(f"Cannot import RAFT-Stereo training model: {exc}") from exc
    finally:
        if inserted:
            sys.path.remove(root_text)
    torch.manual_seed(options.seed)
    rng = np.random.default_rng(options.seed)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    from .spatial import checkpoint_model_configuration
    configuration = checkpoint_model_configuration(state, checkpoint_name)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TrainingError("Original RAFT checkpoint lacks a weights dictionary")
    try:
        model = RAFTStereo(configuration)
        model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()}, strict=True)
        model.to(device)
    except Exception as exc:
        raise TrainingError(f"Cannot load selected RAFT architecture/checkpoint: {exc}") from exc
    if options.train_scope == "update":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("update_block."))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise TrainingError("Selected train_scope contains no trainable RAFT parameters")
    optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=1e-5, eps=1e-8)
    progress("model_setup", status="finished", device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)
        progress("baseline_validation", status="started", processed=0, total=len(validation))
        baseline = _evaluate(torch, model, device, InputPadder, validation_arrays, options,
            progress_callback=lambda event: progress("baseline_validation", **event))
        print(f"Baseline held-out teacher flow error: {baseline['mean_absolute_flow_error_pixels']:.4f} px", file=sys.stderr, flush=True)
        best_metric = baseline["mean_absolute_flow_error_pixels"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch, history, fallback_training_crops = 0, [], 0
        for epoch in range(1, options.epochs + 1):
            progress_state.update(epoch=epoch, step=0)
            model.train()
            model.freeze_bn()
            losses = []
            for step in range(1, options.steps_per_epoch + 1):
                progress_state["step"] = step
                progress("epoch_step", status="started")
                sample_index = int(rng.integers(len(train_arrays)))
                sample = train_arrays[sample_index]
                h, w = sample["flow"].shape
                for attempt in range(128):
                    y = int(rng.integers(max(0, h - options.patch_size) + 1))
                    x = int(rng.integers(max(0, w - options.patch_size) + 1))
                    patch = _patch(sample, y, x, options.patch_size)
                    if (patch[3] & np.isfinite(patch[2]) & (np.abs(patch[2]) < 700)).any():
                        break
                else:
                    y, x = train_arrays.crop_origins[sample_index]
                    patch = _patch(sample, y, x, options.patch_size)
                    fallback_training_crops += 1
                lt, rt, ft, vt, padder = _tensors(torch, device, InputPadder, patch)
                optimizer.zero_grad(set_to_none=True)
                predictions = [padder.unpad(p) for p in model(lt, rt, iters=options.iterations, test_mode=False)]
                loss = sequence_loss(torch, predictions, ft, vt)
                if not bool(torch.isfinite(loss)):
                    raise TrainingError("RAFT loss became nonfinite; checkpoint was not saved")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                progress_state["completed_steps"] += 1
                progress("epoch_step", status="finished", loss=losses[-1])
            progress("epoch_validation", status="started", processed=0, total=len(validation))
            evaluation = _evaluate(torch, model, device, InputPadder, validation_arrays, options,
                progress_callback=lambda event: progress("epoch_validation", **event))
            history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": evaluation})
            print(f"Epoch {epoch}/{options.epochs}: loss {float(np.mean(losses)):.4f}; held-out teacher flow error {evaluation['mean_absolute_flow_error_pixels']:.4f} px", file=sys.stderr, flush=True)
            if evaluation["mean_absolute_flow_error_pixels"] < best_metric:
                best_metric = evaluation["mean_absolute_flow_error_pixels"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        progress("final_validation", status="started", processed=0, total=len(validation), best_epoch=best_epoch)
        final = _evaluate(torch, model, device, InputPadder, validation_arrays, options,
            progress_callback=lambda event: progress("final_validation", **event, best_epoch=best_epoch))
    mode_notes = (["Experimental supervised RAFT training against user-declared measured references.",
        "Reference measurement accuracy and camera registration require independent verification."] if options.mode == "supervised" else
        ["Experimental mixed RAFT training uses measured references and model pseudo-labels, with separate sample provenance.",
         "Pseudo-label agreement does not establish absolute accuracy; verify reference accuracy independently."] if options.mode == "mixed" else
        ["Experimental RAFT teacher distillation; targets are pseudo-labels, not measured ground truth.",
         "Teacher agreement does not prove improved accuracy or eliminate teacher hallucinations."])
    notes = [*mode_notes,
        "Results retain spatial_left coordinates. The display image has separate camera/framing.",
        "Trained weights are never selected automatically. Validate independent geometry before trusting displacement."]
    if len({sample["group_id"] for sample in eligible}) < 10:
        notes.append("Fewer than ten independent groups: pipeline smoke experiment, not a validated iPhone-specific model")
    if not manifest.get("explicit_scene_groups"):
        notes.append("Scene groups were not fully supplied; undetected related captures may leak across splits")
    if not options.require_photometric_support:
        notes.append("Photometric support is recorded but not required; some teacher labels lack independent stereo evidence")
    report = {"schema": "ipde-raft-training-report-v1", "status": "experimental", "model": "RAFT-Stereo", "mode": options.mode,
        "device": device, "torch_version": str(torch.__version__),
        "file_workers": worker_count,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(options).items()},
        "dataset_manifest_sha256": manifest_digest, "dataset_edit_revision": manifest.get("edit_revision", 0),
        "teacher_checkpoint_sha256": sorted(teachers),
        "metric_anchor_checkpoint_sha256": sorted(value for value in metric_anchors if value is not None),
        "label_provenance": [{"sample_id": s["id"], "target_choice": key, "label_kind": s[key].get("label_kind"),
            "target_array_sha256": s[key]["target"]["array_sha256"], "source_sha256": s["source_sha256"]} for s, key in zip(eligible, chosen)],
        "original_raft_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "raft_configuration": vars(configuration),
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "train_sample_ids": [s["id"] for s in train], "validation_sample_ids": [s["id"] for s in validation],
        "sample_count": len(included), "removed_count": eligibility["removed_count"], "eligible_sample_ids": [sample["id"] for sample in eligible],
        "eligible_count": len(eligible), "excluded_count": len(excluded), "excluded_samples": excluded,
        "train_count": len(train), "validation_count": len(validation),
        "train_group_ids": sorted({s["group_id"] for s in train}), "validation_group_ids": sorted({s["group_id"] for s in validation}),
        "sample_preprocessing": train_arrays.preprocessing() + validation_arrays.preprocessing(),
        "input_preprocessing": "Native stereo crops without resizing; upstream RAFT transforms new float32 RGB model tensors from 0..255 to -1..1",
        "target_preprocessing": "Metric depth converted to signed flow; invalid/occluded/out-of-crop labels masked; stored targets untouched",
        "baseline_validation": baseline, "best_epoch": best_epoch, "validation": final, "history": history,
        "total_steps": progress_state["completed_steps"], "epochs_completed": options.epochs,
        "fallback_training_crops": fallback_training_crops, "warnings": notes}
    payload = {"state_dict": best_state, "ipde_configuration": vars(configuration), "ipde_training": report}
    progress("writing_checkpoint", status="started", checkpoint_path=str(checkpoint), best_epoch=best_epoch)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{checkpoint.name}-", dir=checkpoint.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if "state_dict" not in verified or "ipde_configuration" not in verified:
            raise TrainingError("Saved RAFT checkpoint failed verification")
        report["checkpoint_sha256"] = sha256_file(temporary)
        report["checkpoint_path"] = str(checkpoint)
        if checkpoint.exists() or report_path.exists():
            raise TrainingError("Checkpoint destination appeared during training")
        temporary.rename(checkpoint)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
        del model, optimizer, trainable, state, best_state
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    progress("completed", status="finished", checkpoint_path=str(checkpoint), best_epoch=best_epoch,
        mean_absolute_flow_error_pixels=final["mean_absolute_flow_error_pixels"])
    return report


def infer_trained_depth(left: np.ndarray, right: np.ndarray, calibration: Mapping[str, Any], checkpoint_path: Path | str, *, raft_root: Path | None = None, device: str = "auto", iterations: int = 32) -> Any:
    """Run explicitly selected trained weights through normal stereo checks."""
    return run_raft_stereo(left, right, calibration, RaftStereoOptions(root=raft_root, model=Path(checkpoint_path), device=device, iterations=iterations))


def export_raft_checkpoint(checkpoint_path: Path | str, destination: Path | str, *, raft_root: Path | None = None) -> dict[str, Any]:
    """Export verified local trained weights/configuration without source photos.

    The destination is a new directory containing raft-model.pth and a JSON
    manifest. It is never uploaded or installed as the application's default.
    """
    import torch
    from .spatial import checkpoint_model_configuration

    source = Path(checkpoint_path).expanduser().resolve()
    folder = Path(destination).expanduser().resolve()
    if folder.exists():
        raise TrainingError("Export destination already exists; choose a new directory")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or "state_dict" not in payload or "ipde_training" not in payload:
            raise TrainingError("Export requires an IPDE-trained RAFT checkpoint with provenance")
        configuration = checkpoint_model_configuration(payload, source.name)
        state = payload["state_dict"]
        if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
            raise TrainingError("Checkpoint has an invalid RAFT weights dictionary")
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise TrainingError("Checkpoint contains nonfinite RAFT weights")
        root, _, _ = resolve_raft_resources(RaftStereoOptions(root=raft_root, model=source))
        root_text = str(root)
        inserted = root_text not in sys.path
        if inserted:
            sys.path.insert(0, root_text)
        try:
            from core.raft_stereo import RAFTStereo
            model = RAFTStereo(configuration)
            model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
            del model
        finally:
            if inserted:
                sys.path.remove(root_text)
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError(f"Cannot verify RAFT export checkpoint/architecture: {exc}") from exc
    provenance_keys = ("schema", "status", "model", "mode", "dataset_manifest_sha256", "teacher_checkpoint_sha256", "metric_anchor_checkpoint_sha256",
                       "original_raft_checkpoint_sha256", "train_group_ids", "validation_group_ids", "best_epoch", "validation", "warnings")
    training = {key: payload["ipde_training"][key] for key in provenance_keys if key in payload["ipde_training"]}
    portable = {"state_dict": dict(state), "ipde_configuration": vars(configuration), "ipde_training": training}
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    try:
        output = temporary / "raft-model.pth"
        torch.save(portable, output)
        checked = torch.load(output, map_location="cpu", weights_only=True)
        if set(checked["state_dict"]) != set(state) or any(not torch.equal(state[key], checked["state_dict"][key]) for key in state):
            raise TrainingError("Export changed the stored RAFT weights")
        manifest = {"schema": "ipde-raft-model-export-v1", "model": "RAFT-Stereo", "status": "experimental",
            "checkpoint": "raft-model.pth", "checkpoint_sha256": sha256_file(output),
            "source_checkpoint_sha256": sha256_file(source), "architecture": vars(configuration), "training": training,
            "weights_verified": "Strict architecture load plus bit-exact tensor round trip",
            "use": "Select raft-model.pth in IPDE's RAFT model field, with the compatible upstream RAFT-Stereo source folder",
            "source_photos_included": False, "installed_as_default": False}
        (temporary / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if folder.exists():
            raise TrainingError("Export destination appeared during verification")
        temporary.rename(folder)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
