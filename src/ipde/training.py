"""Local experimental RAFT-Stereo fine-tuning with metric teacher-derived flow."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import warnings
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
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


class TrainingIntegrityError(TrainingError):
    """Consumed scientific data changed or failed its recorded checksums."""


class _TrainingStopped(TrainingError):
    """Cooperative interruption at a validation crop boundary."""


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 10
    # Kept only so old callers fail gracefully across the development transition.
    # An epoch now always visits all eligible training images once.
    steps_per_epoch: int | None = None
    limit_mode: str = "epochs"
    total_steps: int = 1000
    steps_per_update: int = 1
    patch_size: int = 512
    learning_rate: float = 1e-5
    device: str = "auto"
    seed: int = 0
    mode: str = "distillation"
    raft_root: Path | None = None
    raft_model: Path | None = None
    raft_model_member: str | None = None
    iterations: int = 12
    train_scope: str = "update"
    require_photometric_support: bool = False
    require_display_teacher: bool = False
    cache_samples: int = 4
    prefetch_samples: int = 2
    skip_incompatible_targets: bool = True
    workers: int | None = None
    validation_schedule: str = "epoch"
    validation_samples: int = 0
    checkpoint_schedule: str = "epoch"
    checkpoint_every: int = 1
    early_stop_error: float | None = None
    max_loss: float = 1000.0
    control_file: Path | None = None
    resume_from: Path | None = None


def training_step_plan(train_count: int, options: TrainingOptions) -> dict[str, int]:
    """Optimizer updates, with a short accumulation group at every epoch end."""
    if train_count < 1 or options.steps_per_update < 1:
        raise TrainingError("Training images and Steps Per Update must be positive")
    updates = math.ceil(train_count / options.steps_per_update)
    total = options.epochs * updates if options.limit_mode == "epochs" else options.total_steps
    return {"steps_per_epoch": train_count, "epoch_updates": updates,
            "total_steps": total, "epochs": options.epochs if options.limit_mode == "epochs" else math.ceil(total / updates)}


def _native_training_dataset(manifest: Mapping[str, Any]) -> bool:
    """Recognize native Apple data retained by known app transformations.

    This selects when to check arrays, never whether to check consumed arrays.
    Legacy compositions retain per-photo ImageIO provenance and the original
    left/right raw-asset hashes even when their source dataset was removed.
    Matching that evidence avoids requiring a new marker or another full
    decode/hash pass of unrelated auxiliary and duplicate teacher planes.
    """
    if manifest.get("external_import") or manifest.get("hf_import"):
        return False
    generated = bool(manifest.get("generation_output_dir")) or str(manifest.get("precision_policy", "")).startswith(
        "Raw samples/auxiliaries preserved bit-for-bit")
    transformed = any(isinstance(record := manifest.get(key), Mapping) and record.get(flag) is True for key, flag in (
        ("collection", "array_bytes_preserved"), ("curation", "array_bytes_preserved"),
        ("storage_compaction", "array_values_preserved"), ("dataset_edit", "array_bytes_preserved")))
    if not generated and not transformed:
        return False
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        return False
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("external_import") or sample.get("source_identity_kind"):
            return False
        if sample.get("coordinate_reference") != "spatial_left: exact decoded sample coordinates, no EXIF rotation":
            return False
        calibration = sample.get("calibration")
        if not isinstance(calibration, Mapping) or calibration.get("is_spatial_photo") is not True or calibration.get(
                "metadata_source") != "macOS ImageIO container properties":
            return False
        for label in (sample.get(key) for key in ("teacher", "reference", "anchored_teacher", "registered_display_teacher")):
            if isinstance(label, Mapping) and isinstance(metadata := label.get("metadata"), Mapping) and metadata.get("import_provenance"):
                return False
        raw = sample.get("raw_assets")
        if not isinstance(raw, list):
            return False
        for role, key in (("spatial_left", "rgb"), ("spatial_right", "right_rgb")):
            view = sample.get(key)
            if not isinstance(view, Mapping) or not isinstance(view.get("array_sha256"), str):
                return False
            if not any(isinstance(asset, Mapping) and asset.get("kind") == "spatial_view" and asset.get("semantic_name") == role
                    and isinstance(storage := asset.get("storage"), Mapping) and all(storage.get(field) == view.get(field)
                        for field in ("array_sha256", "shape", "dtype")) for asset in raw):
                return False
    return True


def _selected_target(sample: Mapping[str, Any], mode: str) -> str:
    return "reference" if mode == "supervised" else sample.get("training_target_choice", "teacher")


def _target_exclusion(sample: Mapping[str, Any], mode: str, *, require_display_teacher: bool = False) -> dict[str, Any] | None:
    choice = _selected_target(sample, mode)
    allowed = {"reference"} if mode == "supervised" else {"teacher", "registered_display_teacher", "anchored_teacher"}
    if mode == "mixed":
        allowed.add("reference")
    label = sample.get(choice) if isinstance(choice, str) else None
    reason = None
    if choice in {"display_teacher", "anchored_display_teacher"} or isinstance(label, Mapping) and str(label.get("coordinate_reference", "")).startswith("display"):
        reason = sample.get("training_target_choice_note") or "Full display-grid labels require a display-grid student; stock RAFT predicts native LEFT-grid flow. Native-stereo teacher fallback is disabled."
    elif not isinstance(choice, str) or choice not in allowed or not isinstance(label, Mapping):
        reason = "Selected training target is unavailable or incompatible with the training mode"
    elif choice == "registered_display_teacher" and label.get("reference_role") != "left":
        reason = "Selected display teacher is not registered to the LEFT stereo grid"
    elif label.get("units") != "meters":
        reason = "Selected target has no accepted meter scale; relative values remain unchanged"
    elif require_display_teacher and choice != "reference":
        metadata = label.get("metadata", {})
        display = sample.get("display_rgb", {})
        registration = label.get("registration", sample.get("display_registration", {}))
        if (choice != "registered_display_teacher" or not isinstance(metadata, Mapping) or not isinstance(display, Mapping)
                or not isinstance(registration, Mapping) or registration.get("accepted") is not True
                or metadata.get("reference_label") != "display" or not display.get("array_sha256")
                or metadata.get("input_rgb_sha256") != display.get("array_sha256")):
            reason = "Regenerate display-only teacher labels with accepted LEFT-camera registration; this production training run cannot use native-stereo or unknown teacher inputs."
    if reason is None and isinstance(label, Mapping) and choice != "reference":
        metadata = label.get("metadata", {})
        checkpoint = metadata.get("checkpoint_sha256") if isinstance(metadata, Mapping) else None
        if not isinstance(checkpoint, str) or not checkpoint:
            reason = "Selected teacher target does not identify its checkpoint"
    if reason is None:
        return None
    return {"sample_id": sample.get("id", ""), "source_path": sample.get("source_path", ""),
        "target_choice": choice, "units": label.get("units") if isinstance(label, Mapping) else None, "reason": reason}


def training_target_eligibility(manifest: Mapping[str, Any], mode: str = "auto", *, require_display_teacher: bool = False) -> dict[str, Any]:
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
        rejection = _target_exclusion(sample, mode, require_display_teacher=require_display_teacher)
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


def _verified_array(root: Path, record: Mapping[str, Any]) -> np.ndarray:
    """Decode once and verify exactly the scientific array being consumed.

    Lazy native-dataset preflight never grants a checksum exemption. Checking
    unused portrait mattes/native teacher planes can instead remain an explicit
    dataset inspection operation.
    """
    from .dataset_review import _array_path
    try:
        root = root.expanduser().resolve()
        path = _array_path(root, record)
        before = path.stat()
        array = read_array(path)
        if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
            raise TrainingIntegrityError(f"Dataset array shape/dtype mismatch: {record['path']}")
        if sha256_file(path) != record["file_sha256"] or sha256_array(array) != record["array_sha256"]:
            raise TrainingIntegrityError(f"Dataset checksum mismatch: {record['path']}")
        # Own NPY memory too; a later write to a mapped source cannot change
        # values already accepted into the model-input cache.
        if isinstance(array, np.memmap):
            array = np.array(array, copy=True)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise TrainingIntegrityError(f"Dataset array changed during decoding: {record['path']}")
        return array
    except TrainingIntegrityError:
        raise
    except (DatasetError, ValueError, OSError, KeyError) as exc:
        raise TrainingIntegrityError(f"Cannot read dataset array {record.get('path', '')}: {exc}") from exc


def _load_sample(root: Path, sample: dict[str, Any], options: TrainingOptions) -> dict[str, Any]:
    rejection = _target_exclusion(sample, options.mode, require_display_teacher=options.require_display_teacher)
    if rejection is not None:
        raise TrainingError(rejection["reason"])
    label_key = "reference" if options.mode == "supervised" else sample.get("training_target_choice", "teacher")
    if label_key not in {"reference", "teacher", "registered_display_teacher", "anchored_teacher"} or label_key not in sample:
        raise TrainingError("Dataset has an unsupported training target choice")
    label = sample[label_key]
    if label_key == "registered_display_teacher" and label.get("reference_role") != "left":
        raise TrainingError("A RIGHT display teacher requires explicit camera reprojection; it cannot supervise the LEFT pair directly")
    if label["units"] != "meters":
        raise TrainingError("RAFT pseudo-labels require meters; relative outputs need an explicit accepted metric-anchor estimate or independently measured calibration")
    left = _verified_array(root, sample["rgb"])
    raw_right = _verified_array(root, sample["right_rgb"])
    depth = _verified_array(root, label["target"])
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
    if "valid_mask" in label:
        recorded_valid = _verified_array(root, label["valid_mask"])
        if recorded_valid.shape != valid.shape or recorded_valid.dtype != np.bool_:
            raise TrainingError("Dataset target validity mask has the wrong shape or dtype")
        valid &= recorded_valid
    elif label.get("validity_policy") != "positive_finite":
        raise TrainingError("Dataset target is missing its recorded validity mask or derived validity policy")
    valid = _visible_teacher_pixels(depth, flow, valid)
    if options.require_photometric_support:
        photometric, evidence = stereo_photometric_support(left, right, -flow)
        photometric &= valid
        valid &= photometric
        supported_pixels = int(photometric.sum())
    else:
        supported_pixels = None
        evidence = {"status": "not_requested", "reason": "Photometric filtering disabled; no redundant full-image support pass"}
    if not valid.any():
        raise TrainingError(f"{sample['id']}: no visible in-image teacher correspondences remain")
    return {"left": left, "right": right, "flow": flow, "valid": valid, "details": {
        "sample_id": sample["id"], "geometry": geometry,
        "target_choice": label_key,
        "right_inference_sha256": sha256_array(right), "right_registration": registration,
        "accepted_teacher_pixels": int(valid.sum()), "photometrically_supported_teacher_pixels": supported_pixels,
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
        self.pending: dict[int, Future] = {}
        self.executor: ThreadPoolExecutor | None = None
        largest = max((math.prod(sample.get("rgb", {}).get("shape", [1])[:2]) for sample in samples), default=1)
        self.prefetch_count = min(options.prefetch_samples,
            memory_limited_workers(options.workers, largest * 64, budget_bytes=1024 * 1024**2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < 0: index += len(self.samples)
        if not 0 <= index < len(self.samples): raise IndexError(index)
        if index not in self.cache:
            value = self.pending.pop(index).result() if index in self.pending else _load_sample(self.root, self.samples[index], self.options)
            self.details[index] = value["details"]
            self.cache[index] = value
            while len(self.cache) > self.options.cache_samples:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def prefetch(self, indices: Sequence[int]) -> None:
        """Overlap bounded CPU decoding/geometry with the accelerator update.

        Worker threads return values; only the training thread publishes cache
        entries or changes scientific membership.
        """
        if not self.prefetch_count:
            return
        wanted = set(indices[:self.prefetch_count])
        for index in list(self.pending):
            if index not in wanted and (self.pending[index].done() or self.pending[index].cancel()):
                self.pending.pop(index)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.prefetch_count, thread_name_prefix="ipde-train-input")
        for index in indices[:self.prefetch_count]:
            if index not in self.cache and index not in self.pending and len(self.pending) < self.prefetch_count:
                self.pending[index] = self.executor.submit(_load_sample, self.root, self.samples[index], self.options)

    def close(self) -> None:
        for future in self.pending.values():
            future.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.pending.clear()

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
            except TrainingIntegrityError:
                raise
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
              progress_callback: Callable[[dict[str, Any]], None] | None = None,
              sample_indices: Sequence[int] | None = None,
              stopped_callback: Callable[[], bool] | None = None) -> dict[str, Any]:
    indices = list(range(len(samples))) if sample_indices is None else list(sample_indices)
    total_error, total_pixels, within_one, within_three = 0.0, 0, 0, 0
    patches = []
    model.eval()
    with torch.inference_mode():
        for position, index in enumerate(indices):
            if stopped_callback is not None and stopped_callback():
                raise _TrainingStopped("Training stopped during validation")
            if isinstance(samples, _SamplePool):
                samples.prefetch(indices[position + 1:position + 1 + samples.prefetch_count])
            if progress_callback is not None:
                progress_callback({"processed": position, "total": len(indices), "sample_index": index,
                    "evaluated_patches": len(patches), "status": "running"})
            sample = samples[index]
            crop_origins = _validation_crop_origins(sample, options.patch_size)
            for crop_index, (y, x) in enumerate(crop_origins):
                if stopped_callback is not None and stopped_callback():
                    raise _TrainingStopped("Training stopped during validation")
                patch = _patch(sample, y, x, options.patch_size)
                if not patch[3].any():
                    continue
                if progress_callback is not None:
                    progress_callback({"processed": position, "total": len(indices), "sample_index": index,
                        "sample_patch": crop_index + 1, "sample_patches": len(crop_origins),
                        "evaluated_patches": len(patches), "status": "running"})
                lt, rt, ft, vt, padder = _tensors(torch, device, padder_class, patch)
                _, prediction = model(lt, rt, iters=options.iterations, test_mode=True)
                difference = (padder.unpad(prediction) - ft).abs()[vt].cpu().numpy().astype(np.float64)
                if not np.isfinite(difference).all():
                    raise TrainingError("RAFT produced nonfinite validation predictions")
                total_error += float(difference.sum(dtype=np.float64))
                total_pixels += difference.size
                within_one += int(np.count_nonzero(difference < 1))
                within_three += int(np.count_nonzero(difference < 3))
                patches.append({"sample_index": index, "x": x, "y": y, "width": patch[0].shape[1], "height": patch[0].shape[0]})
                if progress_callback is not None:
                    progress_callback({"processed": position, "total": len(indices), "sample_index": index,
                        "sample_patch": crop_index + 1, "sample_patches": len(crop_origins),
                        "evaluated_patches": len(patches), "status": "running"})
            if progress_callback is not None:
                progress_callback({"processed": position + 1, "total": len(indices), "sample_index": index,
                    "evaluated_patches": len(patches), "status": "running"})
    if not total_pixels:
        raise TrainingError("Held-out crops lack supported correspondences; enlarge patch_size or add a scene")
    result = {"mean_absolute_flow_error_pixels": total_error / total_pixels, "within_one_pixel_fraction": within_one / total_pixels,
        "within_three_pixels_fraction": within_three / total_pixels, "evaluated_pixels_including_overlapping_patches": total_pixels,
        "sample_indices": indices, "sample_count": len(indices), "full_validation": len(indices) == len(samples),
        "scope": "Fixed native-resolution stereo crops; overlapping pixels may count more than once", "patches": patches,
        "reference": "teacher-derived flow pseudo-label agreement" if options.mode == "distillation" else "user-supplied measured meter-depth references" if options.mode == "supervised" else "mixed measured-reference and teacher pseudo-label agreement"}
    if progress_callback is not None:
        progress_callback({"processed": len(indices), "total": len(indices), "evaluated_patches": len(patches),
            "mean_absolute_flow_error_pixels": result["mean_absolute_flow_error_pixels"], "status": "finished"})
    return result


def _atomic_training_checkpoint(torch: Any, destination: Path, payload: dict[str, Any], report: dict[str, Any]) -> None:
    """Publish new verified weights, resumable state, and their matching report."""
    companion = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or companion.exists():
        raise TrainingError(f"Checkpoint/report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    json_temporary = temporary.with_suffix(temporary.suffix + ".json")
    try:
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(verified, Mapping) or not verified.get("state_dict") or "ipde_configuration" not in verified:
            raise TrainingError("Saved RAFT checkpoint failed verification")
        if any(not bool(torch.isfinite(value).all()) for value in verified["state_dict"].values()):
            raise TrainingError("Checkpoint contains nonfinite RAFT weights")
        report.update(checkpoint_sha256=sha256_file(temporary), checkpoint_path=str(destination))
        json_temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Hard-link publication is atomic and cannot overwrite a destination
        # which appeared during training, unlike a rename on POSIX.
        os.link(temporary, destination)
        try:
            os.link(json_temporary, companion)
        except Exception:
            destination.unlink()
            raise
    finally:
        temporary.unlink(missing_ok=True)
        json_temporary.unlink(missing_ok=True)


def _cpu_tree(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(torch, child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(torch, child) for child in value)
    return value


def _validation_indices(count: int, requested: int, rng: Any) -> list[int]:
    if requested == 0 or requested >= count:
        return list(range(count))
    return [int(index) for index in rng.choice(count, requested, replace=False)]


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
    if min(options.epochs, options.total_steps, options.steps_per_update) < 1 or options.patch_size < 64 or options.patch_size % 32:
        raise TrainingError("Epochs/steps must be positive; native patch_size must be a multiple of 32 and at least 64")
    if not isinstance(options.cache_samples, int) or isinstance(options.cache_samples, bool) or not 1 <= options.cache_samples <= 64:
        raise TrainingError("cache_samples must be an integer from 1 to 64")
    if not isinstance(options.skip_incompatible_targets, bool):
        raise TrainingError("skip_incompatible_targets must be boolean")
    if not isinstance(options.require_display_teacher, bool):
        raise TrainingError("require_display_teacher must be boolean")
    if options.limit_mode not in {"epochs", "steps"} or options.validation_schedule not in {"epoch", "checkpoint"}:
        raise TrainingError("Use epochs/steps limit mode and epoch/checkpoint validation schedule")
    if options.checkpoint_schedule not in {"epoch", "epochs", "steps", "end"} or options.checkpoint_every < 1:
        raise TrainingError("Use epoch/epochs/steps/end checkpoint schedule and a positive checkpoint interval")
    if options.validation_samples < 0 or not 0 <= options.prefetch_samples <= 8:
        raise TrainingError("Validation sample count must be nonnegative and prefetch_samples must be 0..8")
    if options.early_stop_error is not None and (not math.isfinite(options.early_stop_error) or options.early_stop_error <= 0):
        raise TrainingError("Early-stop flow error must be finite and positive")
    if not math.isfinite(options.max_loss) or options.max_loss <= 0:
        raise TrainingError("Maximum loss must be finite and positive")
    if not 1 <= options.iterations <= 256 or not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise TrainingError("RAFT iterations must be 1..256 and learning_rate finite and positive")
    progress_state = {"epoch": 0, "epochs": options.epochs, "step": 0,
        "steps_per_epoch": 0, "completed_steps": 0, "completed_images": 0,
        "total_steps": options.total_steps if options.limit_mode == "steps" else 0}

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
    if options.control_file and Path(options.control_file).expanduser().resolve().is_relative_to(root):
        raise TrainingError("Training control file must be outside the input dataset directory")
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
    eligibility = training_target_eligibility(manifest, options.mode, require_display_teacher=options.require_display_teacher)
    options = replace(options, mode=eligibility["mode"])
    excluded = list(eligibility["excluded"])
    progress_state.update(sample_count=eligibility["sample_count"], eligible_count=eligibility["eligible_count"],
        excluded_count=eligibility["excluded_count"], train_count=eligibility["train_count"],
        validation_count=eligibility["validation_count"])
    if excluded and not options.skip_incompatible_targets:
        raise TrainingError(f"{excluded[0]['sample_id']}: {excluded[0]['reason']}")
    lazy_native = _native_training_dataset(manifest)
    try:
        load_dataset(root, include_excluded=False, workers=worker_count, snapshot=manifest, metadata_only=lazy_native)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    included = [sample for sample in manifest["samples"] if not sample.get("excluded", False)]
    progress("filtering_targets", status="started", processed=0, total=len(included))
    for rejection in excluded:
        progress("skipped_sample", status="finished", **rejection)
    eligible = [sample for sample in included if _target_exclusion(sample, options.mode, require_display_teacher=options.require_display_teacher) is None]
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
    if not lazy_native:
        excluded.extend(train_arrays.prepare("training", lambda event: progress("preparing_targets", **{**event, "total": total_targets}),
        lambda rejection: progress("skipped_sample", status="finished", **rejection)))
        excluded.extend(validation_arrays.prepare("validation", lambda event: progress("preparing_targets",
        **{**event, "processed": len(train) + event["processed"], "total": total_targets}),
        lambda rejection: progress("skipped_sample", status="finished", **rejection)))
    progress("preparing_targets", status="finished", processed=0 if lazy_native else total_targets, total=total_targets,
        integrity_policy="Consumed arrays are checksum-verified on decoding" if lazy_native else "Full dataset verification plus consumed-array checks",
        preparation_policy="Native targets prepared when used" if lazy_native else "Eager geometry eligibility")
    train, validation = train_arrays.samples, validation_arrays.samples
    if not train or not validation:
        raise TrainingError("No usable training and held-out validation split remains after targets without valid geometry/support were skipped")
    _check_splits(train + validation)
    usable_records = {id(sample) for sample in train + validation}
    eligible = [sample for sample in eligible if id(sample) in usable_records]
    progress_state.update(eligible_count=len(eligible), excluded_count=len(excluded),
        train_count=len(train), validation_count=len(validation), **training_step_plan(len(train), options))
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
    raft_options = RaftStereoOptions(root=options.raft_root, model=options.resume_from or options.raft_model,
        model_member=None if options.resume_from else options.raft_model_member, device=options.device, iterations=options.iterations)
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
    validation_rng = np.random.default_rng(options.seed + 1)
    loaded_payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    state = loaded_payload
    from .spatial import checkpoint_model_configuration
    configuration = checkpoint_model_configuration(state, checkpoint_name)
    # Upstream CUDA autocast is inappropriate for the FP32 MPS training path.
    configuration.mixed_precision = False
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
    progress("model_setup", status="finished", device=device, precision="float32", prefetch_samples=train_arrays.prefetch_count)
    history: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}
    final: dict[str, Any] = {}
    best_metric, best_epoch = math.inf, 0
    fallback_training_crops = 0
    epochs_completed, epoch, cursor = 0, 1, 0
    epoch_order: list[int] = []
    epoch_losses: list[float] = []
    active_train = list(range(len(train_arrays)))
    checkpoints: list[str] = []
    stop_reason = "completed"
    stop_requested = False
    pending_manual_save = False
    last_control_request: str | None = None
    last_safe_payload: dict[str, Any] | None = None
    last_evaluation_step = -1
    update_in_progress = False
    optimizer_step_in_progress = False
    cursor_before = cursor
    group_rng_state = None
    group_torch_rng_state = None
    original_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    options_json = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(options).items()}

    if options.resume_from:
        resume = loaded_payload.get("ipde_resume") if isinstance(loaded_payload, Mapping) else None
        previous_report = loaded_payload.get("ipde_training", {}) if isinstance(loaded_payload, Mapping) else {}
        if not isinstance(resume, Mapping) or resume.get("schema") != "ipde-raft-resume-v1":
            raise TrainingError("Resume requires an IPDE v0.9 resumable training checkpoint")
        if resume.get("dataset_manifest_sha256") != manifest_digest:
            raise TrainingError("Resume dataset has changed; use the identical dataset snapshot")
        previous_options = previous_report.get("options", {})
        for key in ("mode", "train_scope", "patch_size", "iterations", "steps_per_update", "learning_rate", "seed", "require_photometric_support", "require_display_teacher"):
            if previous_options.get(key) != options_json[key]:
                raise TrainingError(f"Resume must preserve {key}; start a new fine-tuning run to change it")
        ids = {sample["id"]: index for index, sample in enumerate(train_arrays.samples)}
        try:
            active_train = [ids[sample_id] for sample_id in resume["active_train_ids"]]
            epoch_order = [ids[sample_id] for sample_id in resume["epoch_order_ids"]]
            optimizer.load_state_dict(resume["optimizer_state"])
            epoch, cursor = int(resume["epoch"]), int(resume["cursor"])
            epochs_completed = int(resume["epochs_completed"])
            epoch_losses = list(resume.get("epoch_losses", []))
            progress_state.update(completed_steps=int(resume["completed_steps"]), completed_images=int(resume["completed_images"]),
                **training_step_plan(len(active_train), options))
            excluded_ids = {record.get("sample_id") for record in previous_report.get("excluded_samples", [])}
            inactive_order_ids = {train_arrays.samples[index]["id"] for index in epoch_order if index not in active_train}
            if (not 0 <= cursor <= len(epoch_order) or len(epoch_order) != len(set(epoch_order))
                    or inactive_order_ids - excluded_ids):
                raise ValueError("invalid saved epoch cursor")
            rng.bit_generator.state = json.loads(resume["numpy_rng_state"])
            validation_rng.bit_generator.state = json.loads(resume["validation_rng_state"])
            torch.set_rng_state(resume["torch_rng_state"])
            if device == "mps" and resume.get("accelerator_rng_state") is not None:
                torch.mps.set_rng_state(resume["accelerator_rng_state"])
            elif device == "cuda" and resume.get("accelerator_rng_state") is not None:
                torch.cuda.set_rng_state_all(resume["accelerator_rng_state"])
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            raise TrainingError(f"Resume state is incompatible: {exc}") from exc
        baseline = previous_report.get("baseline_validation", {})
        final = previous_report.get("validation", {})
        history = list(previous_report.get("history", []))
        best_epoch = int(previous_report.get("best_epoch", 0))
        if final:
            best_metric = float(final["mean_absolute_flow_error_pixels"])
        excluded = list(previous_report.get("excluded_samples", excluded))
        progress_state.update(train_count=len(active_train), eligible_count=len(active_train) + len(validation),
            excluded_count=len(excluded))
        original_digest = previous_report.get("original_raft_checkpoint_sha256", original_digest)
        fallback_training_crops = int(previous_report.get("fallback_training_crops", 0))
        progress("resumed", status="finished", source_checkpoint=str(options.resume_from), epoch=epoch, step=cursor)

    mode_notes = (["Experimental supervised RAFT training against user-declared measured references.",
        "Reference measurement accuracy and camera registration require independent verification."] if options.mode == "supervised" else
        ["Experimental mixed RAFT training uses measured references and model pseudo-labels, with separate sample provenance.",
         "Pseudo-label agreement does not establish absolute accuracy; verify reference accuracy independently."] if options.mode == "mixed" else
        ["Experimental RAFT teacher distillation; targets are pseudo-labels, not measured ground truth.",
         "Teacher agreement does not prove improved accuracy or eliminate teacher hallucinations."])
    notes = [*mode_notes, "Results retain spatial_left coordinates. The display image has separate camera/framing.",
        "Trained weights are never selected automatically. Validate independent geometry before trusting displacement."]
    if len({sample["group_id"] for sample in eligible}) < 10:
        notes.append("Fewer than ten independent groups: pipeline smoke experiment, not a validated iPhone-specific model")
    if not manifest.get("explicit_scene_groups"):
        notes.append("Scene groups were not fully supplied; undetected related captures may leak across splits")
    if not options.require_photometric_support:
        notes.append("Photometric filtering is disabled; teacher labels do not constitute independent stereo evidence")

    def make_report(reason: str, *, intermediate: bool = False) -> dict[str, Any]:
        actual_train = [train_arrays.samples[index] for index in active_train]
        retained_ids = {sample["id"] for sample in actual_train + validation}
        actual_eligible = [sample for sample in eligible if sample["id"] in retained_ids]
        selected = [_selected_target(sample, options.mode) for sample in actual_eligible]
        return {"schema": "ipde-raft-training-report-v2", "status": "experimental", "model": "RAFT-Stereo", "mode": options.mode,
            "device": device, "torch_version": str(torch.__version__), "file_workers": worker_count, "options": options_json,
            "dataset_manifest_sha256": manifest_digest, "dataset_edit_revision": manifest.get("edit_revision", 0),
            "teacher_checkpoint_sha256": sorted(teachers),
            "metric_anchor_checkpoint_sha256": sorted(value for value in metric_anchors if value is not None),
            "label_provenance": [{"sample_id": sample["id"], "target_choice": key, "label_kind": sample[key].get("label_kind"),
                "target_array_sha256": sample[key]["target"]["array_sha256"], "source_sha256": sample["source_sha256"]}
                for sample, key in zip(actual_eligible, selected)],
            "original_raft_checkpoint_sha256": original_digest, "raft_configuration": vars(configuration),
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "train_sample_ids": [sample["id"] for sample in actual_train], "validation_sample_ids": [sample["id"] for sample in validation],
            "sample_count": len(included), "removed_count": eligibility["removed_count"],
            "eligible_sample_ids": [sample["id"] for sample in actual_eligible], "eligible_count": len(actual_eligible),
            "excluded_count": len(excluded), "excluded_samples": excluded, "train_count": len(actual_train), "validation_count": len(validation),
            "train_group_ids": sorted({sample["group_id"] for sample in actual_train}),
            "validation_group_ids": sorted({sample["group_id"] for sample in validation}),
            "sample_preprocessing": train_arrays.preprocessing() + validation_arrays.preprocessing(),
            "input_preprocessing": "Native stereo crops without resizing; upstream RAFT transforms new float32 RGB tensors from 0..255 to -1..1",
            "target_preprocessing": "Metric depth converted to signed flow; invalid/occluded/out-of-crop labels masked; stored targets untouched",
            "divergence_guard": "max_loss compares the sequence-weighted mean per-iteration absolute flow error in pixels; optimization keeps the weighted sum",
            "integrity_policy": "Consumed-array path, shape, dtype, file and payload checksums; native startup defers unused arrays" if lazy_native else "Full dataset and consumed-array verification",
            "baseline_validation": baseline, "best_epoch": best_epoch, "validation": final, "history": list(history),
            "validation_completed_steps": last_evaluation_step,
            "total_steps": progress_state["completed_steps"], "planned_total_steps": progress_state["total_steps"],
            "completed_images": progress_state["completed_images"], "epochs_completed": epochs_completed,
            "current_epoch": epoch, "epoch_image_cursor": cursor, "stop_reason": reason, "intermediate": intermediate,
            "checkpoint_paths": list(checkpoints), "resumable": True,
            "fallback_training_crops": fallback_training_crops, "warnings": list(notes)}

    def resume_state() -> dict[str, Any]:
        accelerator_rng = torch.mps.get_rng_state() if device == "mps" else torch.cuda.get_rng_state_all() if device == "cuda" else None
        return {"schema": "ipde-raft-resume-v1", "dataset_manifest_sha256": manifest_digest,
            "optimizer_state": _cpu_tree(torch, optimizer.state_dict()), "epoch": epoch, "cursor": cursor,
            "epoch_order_ids": [train_arrays.samples[index]["id"] for index in epoch_order],
            "active_train_ids": [train_arrays.samples[index]["id"] for index in active_train],
            "epochs_completed": epochs_completed, "epoch_losses": list(epoch_losses),
            "completed_steps": progress_state["completed_steps"], "completed_images": progress_state["completed_images"],
            "numpy_rng_state": json.dumps(rng.bit_generator.state), "validation_rng_state": json.dumps(validation_rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(), "accelerator_rng_state": accelerator_rng}

    def epoch_image_step() -> int:
        retained = set(active_train)
        return sum(index in retained for index in epoch_order[:cursor])

    def snapshot(reason: str, *, intermediate: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        report = make_report(reason, intermediate=intermediate)
        weights = _cpu_tree(torch, model.state_dict())
        if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
            raise TrainingError("RAFT weights became nonfinite")
        return {"state_dict": weights, "ipde_configuration": vars(configuration),
            "ipde_training": report, "ipde_resume": resume_state()}, report

    def save_checkpoint(reason: str, *, final_checkpoint: bool = False) -> dict[str, Any]:
        nonlocal last_safe_payload
        destination = checkpoint
        if not final_checkpoint:
            stem = f"{checkpoint.stem}-step-{progress_state['completed_steps']:08d}"
            destination = checkpoint.with_name(stem + checkpoint.suffix)
            serial = 1
            while destination.exists() or destination.with_suffix(destination.suffix + ".json").exists():
                serial += 1
                destination = checkpoint.with_name(f"{stem}-{serial}{checkpoint.suffix}")
        progress("writing_checkpoint", status="started", checkpoint_path=str(destination), checkpoint_reason=reason)
        payload, report = snapshot(reason, intermediate=not final_checkpoint)
        checkpoints.append(str(destination))
        report["checkpoint_paths"] = list(checkpoints)
        _atomic_training_checkpoint(torch, destination, payload, report)
        last_safe_payload = payload
        progress("checkpoint_saved", status="finished", checkpoint_path=str(destination), checkpoint_reason=reason)
        return report

    def evaluate(stage: str, *, full: bool = False) -> dict[str, Any]:
        nonlocal final, best_epoch, best_metric, last_evaluation_step
        indices = _validation_indices(len(validation_arrays), 0 if full else options.validation_samples, validation_rng)
        progress(stage, status="started", processed=0, total=len(indices), full_validation=len(indices) == len(validation_arrays))
        evaluation = _evaluate(torch, model, device, InputPadder, validation_arrays, options, sample_indices=indices,
            progress_callback=lambda event: progress(stage, **event), stopped_callback=validation_stopped)
        last_evaluation_step = progress_state["completed_steps"]
        final = evaluation
        metric = evaluation["mean_absolute_flow_error_pixels"]
        if metric < best_metric:
            best_metric, best_epoch = metric, progress_state["epoch"]
        return evaluation

    def threshold_met(evaluation: dict[str, Any]) -> bool:
        if options.early_stop_error is None or evaluation["mean_absolute_flow_error_pixels"] > options.early_stop_error:
            return False
        if not evaluation["full_validation"]:
            evaluation = evaluate("early_stop_confirmation", full=True)
        return evaluation["mean_absolute_flow_error_pixels"] <= options.early_stop_error

    def control_command() -> str | None:
        nonlocal last_control_request
        if not options.control_file:
            return None
        path = Path(options.control_file)
        try:
            if path.stat().st_size > 8192:
                raise TrainingError("Training control file is too large")
            message = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            # A caller may be replacing its tiny JSON request atomically.
            return None
        if not isinstance(message, dict) or message.get("command") not in {"save", "stop"}:
            return None
        request = str(message.get("request_id", json.dumps(message, sort_keys=True)))
        if request == last_control_request:
            return None
        last_control_request = request
        return message["command"]

    def validation_stopped() -> bool:
        nonlocal stop_requested, pending_manual_save
        command = control_command()
        if command == "stop":
            stop_requested = True
        elif command == "save":
            pending_manual_save = True
        return stop_requested

    old_signals = {}
    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, request_stop)
    except ValueError:
        # Library callers may run in a background thread; the control-file
        # protocol remains available there.
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)
            last_safe_payload, _ = snapshot("initial_state", intermediate=True)
            if not options.resume_from:
                baseline = evaluate("baseline_validation")
            last_safe_payload, _ = snapshot("initial_state", intermediate=True)
            while progress_state["completed_steps"] < progress_state["total_steps"]:
                if options.limit_mode == "epochs" and epoch > options.epochs:
                    break
                if not epoch_order:
                    epoch_order = [int(index) for index in rng.permutation(active_train)]
                    cursor, epoch_losses = 0, []
                progress_state.update(epoch=epoch, step=epoch_image_step())
                command = control_command()
                if stop_requested or command == "stop":
                    stop_reason = "user_stopped"
                    break
                if command == "save" or pending_manual_save:
                    pending_manual_save = False
                    if options.validation_schedule == "checkpoint":
                        evaluation = evaluate("checkpoint_validation")
                        if threshold_met(evaluation):
                            stop_reason = "early_stop_error_reached"
                            break
                    save_checkpoint("manual")
                model.train()
                model.freeze_bn()
                cursor_before = cursor
                group_rng_state = json.loads(json.dumps(rng.bit_generator.state))
                group_torch_rng_state = torch.get_rng_state()
                update_in_progress = True
                optimizer.zero_grad(set_to_none=True)
                losses = []
                guard_errors = []
                while cursor < len(epoch_order) and len(losses) < options.steps_per_update:
                    sample_index = epoch_order[cursor]
                    if sample_index not in active_train:
                        cursor += 1
                        continue
                    progress_state["step"] = epoch_image_step() + 1
                    progress("epoch_step", status="started", sample_id=train_arrays.samples[sample_index]["id"])
                    train_arrays.prefetch(epoch_order[cursor + 1:cursor + 1 + train_arrays.prefetch_count])
                    try:
                        sample = train_arrays[sample_index]
                        if sample_index not in train_arrays.crop_origins:
                            origin = _training_crop_origin(sample, options.patch_size)
                            if origin is None:
                                raise TrainingError(f"No supported teacher correspondence fits the requested {options.patch_size}-pixel training crop")
                            train_arrays.crop_origins[sample_index] = origin
                    except TrainingIntegrityError:
                        raise
                    except TrainingError as exc:
                        if not options.skip_incompatible_targets:
                            raise
                        record = train_arrays.samples[sample_index]
                        rejection = {"sample_id": record["id"], "source_path": record.get("source_path", ""),
                            "target_choice": _selected_target(record, options.mode), "reason": str(exc)}
                        excluded.append(rejection)
                        active_train.remove(sample_index)
                        if active_train:
                            progress_state.update(**training_step_plan(len(active_train), options), train_count=len(active_train),
                                excluded_count=len(excluded), eligible_count=len(active_train) + len(validation))
                        progress("skipped_sample", status="finished", **rejection)
                        cursor += 1
                        if not active_train:
                            raise TrainingError("No usable training targets remain")
                        continue
                    h, w = sample["flow"].shape
                    for _ in range(128):
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
                    try:
                        predictions = [padder.unpad(prediction) for prediction in model(lt, rt, iters=options.iterations, test_mode=False)]
                        loss = sequence_loss(torch, predictions, ft, vt)
                    except TrainingError as exc:
                        if "nonfinite" not in str(exc):
                            raise
                        stop_reason = "nonfinite_prediction"
                        break
                    value = float(loss.detach().cpu())
                    gamma = 0.9 ** (15 / (len(predictions) - 1)) if len(predictions) > 1 else 1.0
                    sequence_weight = sum(gamma ** (len(predictions) - index - 1) for index in range(len(predictions)))
                    guard_error = value / sequence_weight
                    if not math.isfinite(value) or value <= 0 or guard_error >= options.max_loss:
                        stop_reason = "nonfinite_loss" if not math.isfinite(value) else "zero_loss" if value <= 0 else "loss_limit_exceeded"
                        diagnostic = guard_error if math.isfinite(guard_error) else None
                        notes.append(f"Stopped for {stop_reason}; weighted mean flow error {diagnostic} pixels, limit {options.max_loss}")
                        progress("training_stopped", status="finished", stop_reason=stop_reason, loss=value if math.isfinite(value) else None,
                            weighted_mean_flow_error_pixels=diagnostic, guard_error_pixels=diagnostic)
                        break
                    loss.backward()
                    losses.append(value)
                    guard_errors.append(guard_error)
                    cursor += 1
                if stop_reason != "completed":
                    # The incomplete accumulation group was never applied.
                    optimizer.zero_grad(set_to_none=True)
                    cursor = cursor_before
                    rng.bit_generator.state = group_rng_state
                    torch.set_rng_state(group_torch_rng_state)
                    update_in_progress = False
                    break
                if losses:
                    # Normalize by the actual group size, including a short
                    # final group or a target skipped by lazy geometry checks.
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.div_(len(losses))
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not bool(torch.isfinite(norm)):
                        optimizer.zero_grad(set_to_none=True)
                        cursor = cursor_before
                        rng.bit_generator.state = group_rng_state
                        torch.set_rng_state(group_torch_rng_state)
                        update_in_progress = False
                        stop_reason = "nonfinite_gradients"
                        break
                    optimizer_step_in_progress = True
                    optimizer.step()
                    optimizer_step_in_progress = False
                    epoch_losses.extend(losses)
                    progress_state["completed_steps"] += 1
                    progress_state["completed_images"] += len(losses)
                    progress_state["step"] = epoch_image_step()
                    progress("epoch_step", status="finished", loss=float(np.mean(losses)), images_in_update=len(losses),
                        guard_error_pixels=float(np.mean(guard_errors)))
                update_in_progress = False
                epoch_ended = cursor >= len(epoch_order)
                if epoch_ended:
                    epochs_completed = epoch
                due = (options.checkpoint_schedule == "steps" and progress_state["completed_steps"] % options.checkpoint_every == 0
                    or epoch_ended and (options.checkpoint_schedule == "epoch"
                        or options.checkpoint_schedule == "epochs" and epoch % options.checkpoint_every == 0))
                at_limit = progress_state["completed_steps"] >= progress_state["total_steps"] or options.limit_mode == "epochs" and epoch_ended and epoch >= options.epochs
                evaluation = None
                if epoch_ended and options.validation_schedule == "epoch" or due and options.validation_schedule == "checkpoint":
                    evaluation = evaluate("epoch_validation" if epoch_ended else "checkpoint_validation")
                    if threshold_met(evaluation):
                        stop_reason = "early_stop_error_reached"
                if epoch_ended:
                    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else None
                    history.append({"epoch": epoch, "training_loss": mean_loss, "validation": evaluation,
                        "completed_steps": progress_state["completed_steps"]})
                    print(f"Epoch {epoch}/{progress_state['epochs']}: loss {mean_loss}; total steps {progress_state['completed_steps']}/{progress_state['total_steps']}", file=sys.stderr, flush=True)
                    epoch += 1
                    epoch_order, cursor, epoch_losses = [], 0, []
                    progress_state.update(**training_step_plan(len(active_train), options), train_count=len(active_train))
                if due and not at_limit and stop_reason == "completed":
                    save_checkpoint("scheduled")
                if stop_reason != "completed" or at_limit:
                    break
            if stop_reason == "completed" and last_evaluation_step != progress_state["completed_steps"]:
                try:
                    evaluation = evaluate("final_validation")
                    if threshold_met(evaluation) and stop_reason == "completed":
                        stop_reason = "early_stop_error_reached"
                except TrainingError as exc:
                    if "nonfinite" not in str(exc):
                        raise
                    stop_reason = "nonfinite_validation"
                    notes.append(str(exc))
            try:
                report = save_checkpoint(stop_reason, final_checkpoint=True)
            except TrainingError as exc:
                if "nonfinite" not in str(exc) or last_safe_payload is None:
                    raise
                # Preserve the most recent completely finite, resumable state.
                # Its cursor and RNG accompany its weights; never pair old
                # weights with a newer optimizer/cursor.
                saved = last_safe_payload
                report = dict(saved["ipde_training"])
                report.update(stop_reason="nonfinite_weights_restored_last_safe_state", intermediate=False)
                report["warnings"] = [*report["warnings"], str(exc)]
                saved = {**saved, "ipde_training": report}
                _atomic_training_checkpoint(torch, checkpoint, saved, report)
        progress("completed", status="finished", checkpoint_path=str(checkpoint), best_epoch=best_epoch,
            stop_reason=report["stop_reason"], mean_absolute_flow_error_pixels=report.get("validation", {}).get("mean_absolute_flow_error_pixels"))
        return report
    except _TrainingStopped:
        report = save_checkpoint("user_stopped", final_checkpoint=True)
        progress("completed", status="finished", checkpoint_path=str(checkpoint), stop_reason="user_stopped")
        return report
    except TrainingError as exc:
        if last_safe_payload is not None and not checkpoint.exists() and not report_path.exists():
            notes.append(str(exc))
            if update_in_progress and not optimizer_step_in_progress:
                optimizer.zero_grad(set_to_none=True)
                cursor = cursor_before
                if group_rng_state is not None:
                    rng.bit_generator.state = group_rng_state
                    torch.set_rng_state(group_torch_rng_state)
            try:
                save_checkpoint("training_error", final_checkpoint=True)
            except (TrainingError, RuntimeError, OSError):
                pass
        raise
    except RuntimeError as exc:
        if last_safe_payload is None:
            raise
        notes.append(str(exc))
        if update_in_progress and not optimizer_step_in_progress:
            optimizer.zero_grad(set_to_none=True)
            cursor = cursor_before
            if group_rng_state is not None:
                rng.bit_generator.state = group_rng_state
                torch.set_rng_state(group_torch_rng_state)
        try:
            if optimizer_step_in_progress:
                raise TrainingError("Optimizer update failed; restoring last consistent resumable state")
            report = save_checkpoint("runtime_error", final_checkpoint=True)
        except (TrainingError, RuntimeError):
            report = dict(last_safe_payload["ipde_training"])
            report.update(stop_reason="runtime_error_restored_last_safe_state", intermediate=False)
            report["warnings"] = [*report["warnings"], str(exc)]
            payload = {**last_safe_payload, "ipde_training": report}
            _atomic_training_checkpoint(torch, checkpoint, payload, report)
        progress("completed", status="finished", checkpoint_path=str(checkpoint), stop_reason=report["stop_reason"],
            completed_steps=report["total_steps"])
        return report
    except OSError as exc:
        if last_safe_payload is not None and not checkpoint.exists() and not report_path.exists():
            notes.append(str(exc))
            if update_in_progress and not optimizer_step_in_progress:
                optimizer.zero_grad(set_to_none=True)
                cursor = cursor_before
                if group_rng_state is not None:
                    rng.bit_generator.state = group_rng_state
                    torch.set_rng_state(group_torch_rng_state)
            try:
                save_checkpoint("training_error", final_checkpoint=True)
            except (TrainingError, RuntimeError, OSError):
                # Keep the original failure visible; a previous verified
                # intermediate checkpoint remains available if publication
                # itself failed (for example because the disk is full).
                pass
        raise
    finally:
        for signum, old_handler in old_signals.items():
            signal.signal(signum, old_handler)
        train_arrays.close()
        validation_arrays.close()
        del model, optimizer, trainable, state
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()


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
