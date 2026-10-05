"""Full-display-grid supervision from native stereo inputs.

Teacher coordinates are never registered or resized. Every valid display pixel
participates in each image loss; bounded decoder tiles limit temporary memory.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import shutil
import tempfile
import threading
from typing import Any, Callable, Mapping

import numpy as np

from .dataset import load_dataset
from .concurrency import memory_limited_workers, resolve_workers
from .formats import sha256_file
from .spatial import _select_device
from .training import (TrainingError, TrainingIntegrityError, TrainingOptions,
    _TrainingStopped, _atomic_training_checkpoint, _check_splits, _cpu_tree,
    _native_training_dataset, _validation_indices, _verified_array, training_step_plan)


def _display_label(sample: Mapping[str, Any], mode: str) -> tuple[str, Mapping[str, Any] | None]:
    choice = "reference" if mode == "supervised" else sample.get("training_target_choice", "display_teacher" if "display_teacher" in sample else "teacher")
    return choice, sample.get(choice) if isinstance(choice, str) else None


def _display_exclusion(sample: Mapping[str, Any], mode: str) -> dict[str, Any] | None:
    choice, label = _display_label(sample, mode)
    display = sample.get("display_rgb")
    metadata = label.get("metadata", {}) if isinstance(label, Mapping) else {}
    reason = None
    allowed = {"reference"} if mode == "supervised" else {"display_teacher", "anchored_display_teacher", "teacher"}
    if mode == "mixed":
        allowed.add("reference")
    if (choice not in allowed
            or not isinstance(label, Mapping) or not isinstance(display, Mapping)):
        reason = "Select a full display-grid teacher label; registered/native stereo targets cannot supervise the display-depth model"
    elif choice == "reference" and not str(label.get("coordinate_reference", "")).startswith("display"):
        reason = "The supplied measured reference is on the LEFT stereo grid; display training requires an independently measured display-grid reference"
    elif choice != "reference" and (not str(label.get("coordinate_reference", "")).startswith("display")
            or not isinstance(metadata, Mapping) or metadata.get("reference_label") != "display"
            or metadata.get("input_rgb_sha256") != display.get("array_sha256") or not display.get("array_sha256")):
        reason = "Regenerate full display-image teacher labels; native-stereo or unknown teacher inputs cannot train this display-depth model"
    elif not isinstance(label.get("target"), Mapping) or label["target"].get("shape") != display.get("shape", [])[:2]:
        reason = "Display teacher target must retain the exact full display H×W grid"
    elif label.get("units") not in {"meters", "relative_depth", "relative_inverse_depth"}:
        reason = "Display target has unsupported units; no automatic scale or inverse-depth conversion is applied"
    elif choice != "reference" and not metadata.get("checkpoint_sha256"):
        reason = "Display teacher does not identify its checkpoint"
    if reason is None:
        return None
    return {"sample_id": sample.get("id", ""), "source_path": sample.get("source_path", ""),
            "target_choice": choice, "units": label.get("units") if isinstance(label, Mapping) else None, "reason": reason}


def display_target_eligibility(manifest: Mapping[str, Any], mode: str = "auto") -> dict[str, Any]:
    """Metadata-only production target checks; numerical files are verified on use."""
    if mode not in {"auto", "distillation", "supervised", "mixed"}:
        raise TrainingError("Use auto/distillation/supervised/mixed training mode")
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or any(not isinstance(sample, Mapping) for sample in samples):
        raise TrainingError("Dataset samples must be a list of sample records")
    if any("excluded" in sample and not isinstance(sample["excluded"], bool) for sample in samples):
        raise TrainingError("Dataset excluded membership must be boolean")
    removed = sum(sample.get("excluded", False) for sample in samples)
    included = [sample for sample in samples if not sample.get("excluded", False)]
    if mode == "auto":
        choices = {sample.get("training_target_choice") for sample in included}
        mode = "supervised" if choices == {"reference"} else "mixed" if "reference" in choices else "distillation"
    eligible, excluded = [], []
    for sample in included:
        rejection = _display_exclusion(sample, mode)
        (excluded if rejection else eligible).append(rejection if rejection else sample)
    units = {sample[_display_label(sample, mode)[0]]["units"] for sample in eligible}
    train_count = sum(sample.get("split") == "train" for sample in eligible)
    validation_count = sum(sample.get("split") == "validation" for sample in eligible)
    reason = ""
    if manifest.get("generation_state", "complete") != "complete" or manifest.get("splits_provisional", False):
        reason = "Finish dataset generation before training"
    elif not eligible:
        reason = "No full display-grid labels are selected; generate display-only teachers or review the selected targets"
    elif len(units) != 1:
        reason = "Display training requires one target unit convention; separate meters, relative depth, and relative inverse-depth runs"
    elif not train_count or not validation_count:
        reason = "Display training requires independent training and held-out validation groups"
    return {"mode": mode, "sample_count": len(included), "removed_count": removed, "eligible_count": len(eligible),
        "excluded_count": len(excluded), "excluded": excluded, "train_count": train_count, "validation_count": validation_count,
        "units": next(iter(units)) if len(units) == 1 else None, "trainable": not reason, "reason": reason,
        "student_architecture": "ipde-display-depth-v1"}


class _DisplayPool:
    """Small verified LRU of original stereo and full display target arrays."""
    def __init__(self, root: Path, samples: list[dict[str, Any]], options: TrainingOptions):
        self.root, self.samples, self.options = root, samples, options
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        estimate = max((sum(math.prod(sample[key]["shape"]) * np.dtype(sample[key]["dtype"]).itemsize for key in ("rgb", "right_rgb"))
            + math.prod(_display_label(sample, options.mode)[1]["target"]["shape"]) * 5 for sample in samples), default=1)
        self.prefetch_count = min(options.prefetch_samples, memory_limited_workers(resolve_workers(options.workers), estimate * 2))
        self.executor = ThreadPoolExecutor(max_workers=self.prefetch_count) if self.prefetch_count else None
        self.pending: dict[int, Any] = {}

    def get(self, index: int) -> dict[str, Any]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        value = self.pending.pop(index).result() if index in self.pending else self._read(index)
        self.cache[index] = value
        while len(self.cache) > max(1, self.options.cache_samples):
            self.cache.popitem(last=False)
        return value

    def prefetch(self, indices):
        if self.executor:
            for index in indices:
                if len(self.pending) >= self.prefetch_count:
                    break
                if index not in self.cache and index not in self.pending:
                    self.pending[index] = self.executor.submit(self._read, index)

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.cache.clear()

    def _read(self, index: int):
        sample = self.samples[index]
        rejection = _display_exclusion(sample, self.options.mode)
        if rejection:
            raise TrainingError(rejection["reason"])
        key, label = _display_label(sample, self.options.mode)
        left, right = (_verified_array(self.root, sample[name]) for name in ("rgb", "right_rgb"))
        target = _verified_array(self.root, label["target"])
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise TrainingError("Display-depth model requires matching full native RGB views")
        if target.ndim != 2 or target.dtype.kind != "f" or list(target.shape) != sample["display_rgb"]["shape"][:2]:
            raise TrainingIntegrityError("Display target grid/dtype differs from its full display reference")
        valid = np.isfinite(target) & (target > 0)
        if "valid_mask" in label:
            recorded = _verified_array(self.root, label["valid_mask"])
            if recorded.shape != target.shape or recorded.dtype != np.bool_:
                raise TrainingIntegrityError("Display target validity mask has the wrong grid/dtype")
            valid &= recorded
        if not valid.any():
            raise TrainingError("Display target contains no positive finite supported pixels")
        value = {"left": left, "right": right, "target": target, "valid": valid,
                 "valid_count": int(valid.sum()), "target_choice": key, "record": sample}
        return value


def _tiles(shape: tuple[int, int], side: int):
    for y in range(0, shape[0], side):
        for x in range(0, shape[1], side):
            yield y, min(y + side, shape[0]), x, min(x + side, shape[1])


def train_display_dataset(dataset_dir: Path | str, checkpoint_path: Path | str,
                          options: TrainingOptions | None = None, *,
                          progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train exact output grids with full-image traversal and resumable updates."""
    from .display_student import DisplayStudentError, create_student, load_student_checkpoint, rgb_tensor
    import torch
    options = options or TrainingOptions()
    if (options.limit_mode not in {"epochs", "steps"} or options.epochs < 1 or options.total_steps < 1
            or options.steps_per_update < 1 or not 32 <= options.patch_size <= 2048):
        raise TrainingError("Use a positive epoch/step limit, Steps Per Update, and decoder tile size 32..2048")
    if (options.validation_schedule not in {"epoch", "checkpoint"} or options.validation_samples < 0
            or options.checkpoint_schedule not in {"epoch", "epochs", "steps", "end"} or options.checkpoint_every < 1):
        raise TrainingError("Invalid display validation/checkpoint schedule")
    if options.train_scope not in {"update", "full"} or options.cache_samples < 1 or not 0 <= options.prefetch_samples <= 8:
        raise TrainingError("Use update/full train scope and at least one cached image")
    if (not math.isfinite(options.learning_rate) or options.learning_rate <= 0
            or not math.isfinite(options.max_loss) or options.max_loss <= 0
            or options.early_stop_error is not None and (not math.isfinite(options.early_stop_error) or options.early_stop_error <= 0)):
        raise TrainingError("Learning rate, safe fractional error, and optional early-stop error must be finite and positive")
    root = Path(dataset_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.is_relative_to(root) or options.control_file and Path(options.control_file).expanduser().resolve().is_relative_to(root):
        raise TrainingError("Checkpoints and training controls must be outside the read-only dataset")
    if checkpoint.exists() or checkpoint.with_suffix(checkpoint.suffix + ".json").exists():
        raise TrainingError("Checkpoint/report already exists; choose a new checkpoint path")
    progress_state: dict[str, Any] = {"epoch": 0, "epochs": options.epochs, "step": 0, "completed_steps": 0,
        "completed_images": 0, "total_steps": 0, "steps_per_epoch": 0}
    def progress(stage: str, **details: Any):
        if progress_callback:
            progress_callback({"phase": "training_progress", "stage": stage, **progress_state, **details})
    progress("dataset_preflight", status="started")
    manifest_digest = sha256_file(root / "dataset.json")
    manifest = load_dataset(root, metadata_only=True)
    lazy = _native_training_dataset(manifest)
    if not lazy:
        manifest = load_dataset(root, workers=options.workers, include_excluded=False, snapshot=manifest)
    if sha256_file(root / "dataset.json") != manifest_digest:
        raise TrainingIntegrityError("Dataset manifest changed during display training preflight")
    eligibility = display_target_eligibility(manifest, options.mode)
    if not eligibility["trainable"]:
        raise TrainingError(eligibility["reason"])
    if eligibility["excluded"] and not options.skip_incompatible_targets:
        raise TrainingError(eligibility["excluded"][0]["reason"])
    from dataclasses import replace
    options = replace(options, mode=eligibility["mode"])
    included = [sample for sample in manifest["samples"] if not sample.get("excluded", False)]
    _check_splits(included)
    eligible = [sample for sample in included if _display_exclusion(sample, options.mode) is None]
    train, validation = _check_splits(eligible)
    progress_state.update(**training_step_plan(len(train), options), train_count=len(train),
        validation_count=len(validation), eligible_count=len(eligible), excluded_count=len(eligibility["excluded"]))
    progress("dataset_preflight", status="finished", integrity_policy="consumed arrays" if lazy else "full verification", units=eligibility["units"])
    torch.manual_seed(options.seed)
    rng, validation_rng = np.random.default_rng(options.seed), np.random.default_rng(options.seed + 1)
    device = _select_device(torch, options.device)
    progress("model_setup", status="started", device=device)
    if options.resume_from:
        model, architecture, device = load_student_checkpoint(options.resume_from, raft_root=options.raft_root,
            device=device, train_scope=options.train_scope, allow_legacy=False)
        if architecture.get("units") != eligibility["units"] or architecture.get("iterations") != options.iterations:
            raise TrainingError("Resume model units/iterations differ from the selected training contract")
    else:
        model, architecture, device = create_student(raft_root=options.raft_root, raft_model=options.raft_model,
            raft_model_member=options.raft_model_member, device=device, train_scope=options.train_scope,
            iterations=options.iterations, seed=options.seed, units=eligibility["units"],
            quality=0 if options.patch_size <= 256 else 1 if options.patch_size <= 512 else 2)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise TrainingError("Display-depth model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=1e-5, eps=1e-8)
    pools = _DisplayPool(root, train, options), _DisplayPool(root, validation, options)
    progress("model_setup", status="finished", device=device, precision="float32", student_architecture="ipde-display-depth-v1")
    options_json = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(options).items()}
    epoch, cursor, epochs_completed = 1, 0, 0
    order: list[int] = []
    epoch_losses, history, checkpoints, notes = [], [], [], [
        "Stereo correspondence aligns right features to one left-reference field; the loss compares one predicted map with the untouched depth target.",
        "Teacher agreement is not independent accuracy and cannot guarantee detail absent from native stereo inputs."]
    if len({sample["group_id"] for sample in eligible}) < 10:
        notes.append("Fewer than ten independent groups; validation coverage is limited")
    if not manifest.get("explicit_scene_groups"):
        notes.append("Explicit scene groups were not fully supplied; undetected related captures may leak across validation splits")
    baseline, final = {}, {}
    best_error, best_epoch, evaluated_step = math.inf, 0, -1
    stop_requested, pending_save, last_request = False, False, None
    stop_reason = "completed"
    last_safe: dict[str, Any] | None = None

    def accelerator_rng():
        return torch.mps.get_rng_state() if device == "mps" else torch.cuda.get_rng_state_all() if device == "cuda" else None

    def restore_rng(saved):
        rng.bit_generator.state = json.loads(saved["numpy_rng_state"])
        validation_rng.bit_generator.state = json.loads(saved["validation_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device == "mps" and saved.get("accelerator_rng_state") is not None:
            torch.mps.set_rng_state(saved["accelerator_rng_state"])
        elif device == "cuda" and saved.get("accelerator_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["accelerator_rng_state"])

    if options.resume_from:
        loaded = torch.load(options.resume_from, map_location="cpu", weights_only=True)
        saved, previous = loaded.get("ipde_resume", {}), loaded.get("ipde_training", {})
        if loaded.get("schema") != "ipde-display-depth-v1" or saved.get("schema") != "ipde-display-resume-v1":
            raise TrainingError("Resume requires a display-depth training checkpoint")
        if saved.get("dataset_manifest_sha256") != manifest_digest:
            raise TrainingError("Resume dataset has changed; use its identical snapshot")
        for key in ("mode", "train_scope", "patch_size", "iterations", "steps_per_update", "learning_rate", "seed"):
            if previous.get("options", {}).get(key) != options_json[key]:
                raise TrainingError(f"Resume must preserve {key}")
        if loaded.get("architecture") != architecture:
            raise TrainingError("Resume display-depth model architecture/units differ")
        model.load_state_dict(loaded["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        ids = {sample["id"]: index for index, sample in enumerate(train)}
        try:
            order = [ids[value] for value in saved["epoch_order_ids"]]
            epoch, cursor, epochs_completed = int(saved["epoch"]), int(saved["cursor"]), int(saved["epochs_completed"])
            if not 0 <= cursor <= len(order) or len(order) != len(set(order)) or order and set(order) != set(range(len(train))):
                raise ValueError("invalid display epoch order/cursor")
            epoch_losses = list(saved["epoch_losses"])
            progress_state.update(completed_steps=int(saved["completed_steps"]), completed_images=int(saved["completed_images"]))
            restore_rng(saved)
        except (KeyError, ValueError, TypeError) as exc:
            raise TrainingError(f"Invalid display resume state: {exc}") from exc
        baseline, final, history = previous.get("baseline_validation", {}), previous.get("validation", {}), list(previous.get("history", []))
        best_error, best_epoch = float(previous.get("best_error") or math.inf), int(previous.get("best_epoch", 0))
        evaluated_step = int(previous.get("validation_completed_steps", -1))
        progress("resumed", status="finished", source_checkpoint=str(options.resume_from))

    def report(reason: str, intermediate=False):
        return {"schema": "ipde-display-training-report-v1", "status": "trained", "model": "RAFT stereo depth model",
            "mode": options.mode, "units": eligibility["units"], "device": device, "torch_version": str(torch.__version__),
            "options": options_json, "architecture": architecture, "dataset_manifest_sha256": manifest_digest,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "train_sample_ids": [sample["id"] for sample in train], "validation_sample_ids": [sample["id"] for sample in validation],
            "train_count": len(train), "validation_count": len(validation), "eligible_count": len(eligible),
            "excluded_samples": eligibility["excluded"], "excluded_count": len(eligibility["excluded"]),
            "label_provenance": [{"sample_id": sample["id"], "target_choice": _display_label(sample, options.mode)[0],
                "target_array_sha256": _display_label(sample, options.mode)[1]["target"]["array_sha256"],
                "source_sha256": sample["source_sha256"]} for sample in eligible],
            "input_preprocessing": "Full native LEFT/RIGHT RGB; right-to-left correspondence alignment and masked feature fusion into one reference; no input resizing",
            "target_preprocessing": "Full unregistered display grid; each supported positive finite pixel visited once per image in bounded decoder tiles",
            "divergence_guard": "Mean absolute fractional depth error abs(prediction-target)/target; dimensionless",
            "integrity_policy": "Consumed-array checksums; native datasets defer unused arrays" if lazy else "Full dataset plus consumed-array verification",
            "baseline_validation": baseline, "validation": final, "history": list(history), "best_epoch": best_epoch,
            "best_error": best_error if math.isfinite(best_error) else None, "validation_completed_steps": evaluated_step,
            "total_steps": progress_state["completed_steps"], "planned_total_steps": progress_state["total_steps"],
            "completed_images": progress_state["completed_images"], "current_epoch": epoch,
            "epochs_completed": epochs_completed, "epoch_image_cursor": cursor,
            "stop_reason": reason, "checkpoint_paths": list(checkpoints), "intermediate": intermediate, "resumable": True, "warnings": list(notes)}

    def snapshot(reason: str, intermediate=False):
        weights = _cpu_tree(torch, model.state_dict())
        if any(not bool(torch.isfinite(value).all()) for value in weights.values()):
            raise TrainingError("Display-depth model weights became nonfinite")
        metadata = report(reason, intermediate)
        resume = {"schema": "ipde-display-resume-v1", "dataset_manifest_sha256": manifest_digest,
            "optimizer_state": _cpu_tree(torch, optimizer.state_dict()), "epoch": epoch, "cursor": cursor,
            "epoch_order_ids": [train[index]["id"] for index in order], "epochs_completed": epochs_completed,
            "epoch_losses": list(epoch_losses), "completed_steps": progress_state["completed_steps"],
            "completed_images": progress_state["completed_images"], "numpy_rng_state": json.dumps(rng.bit_generator.state),
            "validation_rng_state": json.dumps(validation_rng.bit_generator.state), "torch_rng_state": torch.get_rng_state(),
            "accelerator_rng_state": accelerator_rng()}
        return {"schema": "ipde-display-depth-v1", "state_dict": weights, "architecture": architecture,
            "ipde_configuration": architecture, "ipde_training": metadata, "ipde_resume": resume}

    def save(reason: str, final_checkpoint=False):
        nonlocal last_safe
        destination = checkpoint
        if not final_checkpoint:
            destination = checkpoint.with_name(f"{checkpoint.stem}-step-{progress_state['completed_steps']:08d}{checkpoint.suffix}")
            serial = 1
            while destination.exists() or destination.with_suffix(destination.suffix + ".json").exists():
                serial += 1
                destination = checkpoint.with_name(f"{checkpoint.stem}-step-{progress_state['completed_steps']:08d}-{serial}{checkpoint.suffix}")
        progress("writing_checkpoint", status="started", checkpoint_path=str(destination), checkpoint_reason=reason)
        payload = snapshot(reason, not final_checkpoint)
        checkpoints.append(str(destination))
        payload["ipde_training"]["checkpoint_paths"] = list(checkpoints)
        _atomic_training_checkpoint(torch, destination, payload, payload["ipde_training"])
        last_safe = payload
        progress("checkpoint_saved", status="finished", checkpoint_path=str(destination), checkpoint_reason=reason)
        return payload["ipde_training"]

    def controls():
        nonlocal stop_requested, pending_save, last_request
        if options.control_file:
            try:
                path = Path(options.control_file)
                message = json.loads(path.read_text()) if path.stat().st_size <= 8192 else {}
                request = str(message.get("request_id", json.dumps(message, sort_keys=True))) if isinstance(message, dict) else None
                if request != last_request and isinstance(message, dict) and message.get("command") in {"save", "stop"}:
                    last_request = request
                    stop_requested |= message["command"] == "stop"
                    pending_save |= message["command"] == "save"
            except (FileNotFoundError, ValueError, OSError):
                pass
        return stop_requested

    def tensors(sample):
        return tuple(rgb_tensor(sample[name], sample["record"][key], torch, device)
                     for name, key in (("left", "rgb"), ("right", "right_rgb")))

    def image_error(sample, *, backward: bool):
        if controls():
            raise _TrainingStopped("User stopped display training")
        left, right = tensors(sample)
        context = model.encode(left, right)
        original_context = context
        if backward:
            # Tile losses accumulate decoder/leaf gradients independently.
            # Traverse the expensive full native encoder graph once per image.
            context = {key: value.detach().requires_grad_(value.requires_grad) if isinstance(value, torch.Tensor) else value
                       for key, value in original_context.items()}
        shape = sample["target"].shape
        tiles = list(_tiles(shape, options.patch_size))
        relative_sum = torch.zeros((), device=device)
        absolute_sum = torch.zeros((), device=device)
        finite = torch.ones((), device=device, dtype=torch.bool)
        for ordinal, bounds in enumerate(tiles):
            if controls():
                raise _TrainingStopped("User stopped display training")
            y0, y1, x0, x1 = bounds
            valid = sample["valid"][y0:y1, x0:x1]
            if not valid.any():
                continue
            target = torch.from_numpy(np.array(sample["target"][y0:y1, x0:x1], dtype=np.float32, copy=True)).to(device)
            mask = torch.from_numpy(np.array(valid, copy=True)).to(device)
            prediction = model.render(context, shape, bounds)
            if prediction.shape != (1, 1, y1 - y0, x1 - x0):
                raise TrainingError("Display decoder returned the wrong tile grid")
            difference = (prediction[0, 0][mask] - target[mask]).abs()
            fraction = difference / target[mask]
            finite &= torch.isfinite(fraction).all() & (prediction[0, 0][mask] > 0).all()
            absolute_sum += difference.detach().sum()
            relative_sum += fraction.detach().sum()
            if backward:
                (fraction.sum() / sample["valid_count"]).backward()
            progress("display_tile", status="finished", sample_id=sample["record"]["id"], tile=ordinal + 1,
                tiles=len(tiles), output_shape=list(shape), validation=not backward)
        relative, absolute, accepted = torch.stack((relative_sum, absolute_sum, finite.float())).cpu().tolist()
        if not accepted or not math.isfinite(relative) or not math.isfinite(absolute):
            raise TrainingError("nonfinite or nonpositive display prediction")
        if backward:
            originals, gradients = [], []
            for key, value in original_context.items():
                if isinstance(value, torch.Tensor) and value.requires_grad and context[key].grad is not None:
                    originals.append(value)
                    gradients.append(context[key].grad)
            if originals:
                torch.autograd.backward(originals, gradients)
        return relative / sample["valid_count"], absolute / sample["valid_count"], sample["valid_count"]

    def evaluate(stage, full=False):
        nonlocal final, best_error, best_epoch, evaluated_step
        indices = _validation_indices(len(validation), 0 if full else options.validation_samples, validation_rng)
        progress(stage, status="started", processed=0, total=len(indices), full_validation=len(indices) == len(validation))
        model.eval()
        relative_sum, absolute_sum, pixels = 0., 0., 0
        with torch.no_grad():
            pools[1].prefetch(indices)
            for ordinal, index in enumerate(indices):
                sample = pools[1].get(index)
                pools[1].prefetch(indices[ordinal + 1:])
                relative, absolute, count = image_error(sample, backward=False)
                relative_sum += relative * count
                absolute_sum += absolute * count
                pixels += count
                progress(stage, status="running", processed=ordinal + 1, total=len(indices))
        final = {"mean_absolute_fractional_depth_error": relative_sum / pixels, "mean_absolute_depth_error": absolute_sum / pixels,
            "units": eligibility["units"], "evaluated_pixels": pixels, "sample_indices": indices, "sample_count": len(indices),
            "full_validation": len(indices) == len(validation), "scope": "All supported pixels on each selected full display grid, without overlap",
            "reference": "Display teacher pseudo-label agreement; not independent accuracy"}
        evaluated_step = progress_state["completed_steps"]
        if final["mean_absolute_fractional_depth_error"] < best_error:
            best_error, best_epoch = final["mean_absolute_fractional_depth_error"], progress_state["epoch"]
        progress(stage, status="finished", processed=len(indices), total=len(indices), **final)
        return final

    def threshold(evaluation):
        if options.early_stop_error is None or evaluation["mean_absolute_fractional_depth_error"] > options.early_stop_error:
            return False
        if not evaluation["full_validation"]:
            evaluation = evaluate("early_stop_confirmation", full=True)
        return evaluation["mean_absolute_fractional_depth_error"] <= options.early_stop_error

    old_signals = {}
    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, request_stop)
    try:
        last_safe = snapshot("initial_state")
        if not baseline:
            baseline = evaluate("baseline_validation")
        while progress_state["completed_steps"] < progress_state["total_steps"]:
            if controls():
                stop_reason = "user_stopped"
                break
            if pending_save:
                if options.validation_schedule == "checkpoint":
                    if threshold(evaluate("checkpoint_validation")):
                        stop_reason = "early_stop_error_reached"
                        break
                save("manual")
                pending_save = False
            if order and cursor == len(order):
                if epoch_losses and not any(item.get("epoch") == epoch for item in history):
                    history.append({"epoch": epoch, "training_loss": float(np.mean(epoch_losses)),
                        "validation": final if evaluated_step == progress_state["completed_steps"] else None,
                        "completed_steps": progress_state["completed_steps"]})
                epoch += 1
                order, cursor, epoch_losses = [], 0, []
            if not order:
                order = [int(index) for index in rng.permutation(len(train))]
                cursor, epoch_losses = 0, []
            progress_state.update(epoch=epoch, step=cursor)
            last_safe = snapshot("last_finite_update")
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses = []
            group_size = min(options.steps_per_update, len(order) - cursor)
            for _ in range(group_size):
                sample = pools[0].get(order[cursor])
                pools[0].prefetch(order[cursor + 1:])
                progress("epoch_step", status="started", sample_id=sample["record"]["id"], image_step=cursor + 1)
                relative, absolute, _ = image_error(sample, backward=True)
                if not math.isfinite(relative) or relative <= 0 or relative >= options.max_loss:
                    stop_reason = "nonfinite_loss" if not math.isfinite(relative) else "zero_loss" if relative <= 0 else "loss_limit_exceeded"
                    notes.append(f"Stopped for {stop_reason}; fractional display error {relative}, limit {options.max_loss}")
                    break
                losses.append(relative)
                cursor += 1
            if stop_reason != "completed":
                break
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(len(losses))
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not bool(torch.isfinite(norm)):
                stop_reason = "nonfinite_gradients"
                break
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
                stop_reason = "nonfinite_weights"
                break
            progress_state["completed_steps"] += 1
            progress_state["completed_images"] += len(losses)
            progress_state["step"] = cursor
            epoch_losses.extend(losses)
            ended = cursor == len(order)
            if ended:
                epochs_completed = epoch
            progress("epoch_step", status="finished", loss=float(np.mean(losses)), images_in_update=len(losses),
                mean_absolute_fractional_depth_error=float(np.mean(losses)))
            due = (options.checkpoint_schedule == "steps" and progress_state["completed_steps"] % options.checkpoint_every == 0
                or ended and (options.checkpoint_schedule == "epoch" or options.checkpoint_schedule == "epochs" and epoch % options.checkpoint_every == 0))
            last_safe = snapshot("last_finite_update")
            evaluation = None
            if ended and options.validation_schedule == "epoch" or due and options.validation_schedule == "checkpoint":
                evaluation = evaluate("epoch_validation" if ended else "checkpoint_validation")
                if threshold(evaluation):
                    stop_reason = "early_stop_error_reached"
            if ended:
                history.append({"epoch": epoch, "training_loss": float(np.mean(epoch_losses)), "validation": evaluation,
                    "completed_steps": progress_state["completed_steps"]})
                epoch += 1
                order, cursor, epoch_losses = [], 0, []
            if due and stop_reason == "completed" and progress_state["completed_steps"] < progress_state["total_steps"]:
                save("scheduled")
            if stop_reason != "completed":
                break
        if stop_reason == "completed" and evaluated_step != progress_state["completed_steps"]:
            if threshold(evaluate("final_validation")):
                stop_reason = "early_stop_error_reached"
        if stop_reason in {"nonfinite_loss", "zero_loss", "loss_limit_exceeded", "nonfinite_gradients", "nonfinite_weights"}:
            raise TrainingError(stop_reason)
        result = save(stop_reason, final_checkpoint=True)
    except (TrainingError, RuntimeError, DisplayStudentError) as exc:
        if isinstance(exc, TrainingIntegrityError) or last_safe is None:
            raise
        reason = "user_stopped" if isinstance(exc, _TrainingStopped) else stop_reason if stop_reason != "completed" else "nonfinite_prediction" if "nonfinite" in str(exc) else "runtime_error"
        notes.append(str(exc))
        model.load_state_dict(last_safe["state_dict"], strict=True)
        optimizer.load_state_dict(last_safe["ipde_resume"]["optimizer_state"])
        saved = last_safe["ipde_resume"]
        epoch, cursor, epochs_completed = saved["epoch"], saved["cursor"], saved["epochs_completed"]
        ids = {sample["id"]: index for index, sample in enumerate(train)}
        order = [ids[value] for value in saved["epoch_order_ids"]]
        epoch_losses = list(saved["epoch_losses"])
        progress_state.update(completed_steps=saved["completed_steps"], completed_images=saved["completed_images"])
        restore_rng(saved)
        progress("training_stopped", status="finished", stop_reason=reason)
        result = save(reason, final_checkpoint=True)
    finally:
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)
        for pool in pools:
            pool.close()
    progress("completed", status="finished", checkpoint_path=str(checkpoint), stop_reason=result["stop_reason"],
        mean_absolute_fractional_depth_error=result.get("validation", {}).get("mean_absolute_fractional_depth_error"))
    return result


def export_display_checkpoint(checkpoint_path: Path | str, destination: Path | str, *,
                              raft_root: Path | None = None) -> dict[str, Any]:
    """Strict-load an explicitly selected model and export self-contained weights."""
    import torch
    from .display_student import SCHEMA, load_student_checkpoint
    source = Path(checkpoint_path).expanduser().resolve()
    source_digest = sha256_file(source)
    source_report = source.with_suffix(source.suffix + ".json")
    report_digest = sha256_file(source_report) if source_report.exists() else None
    if report_digest:
        metadata = json.loads(source_report.read_text())
        if not isinstance(metadata, Mapping) or metadata.get("checkpoint_sha256") != source_digest:
            raise TrainingIntegrityError("Source display checkpoint/report hash verification failed")
    folder = Path(destination).expanduser().resolve()
    if folder.exists():
        raise TrainingError("Export destination already exists; choose a new directory")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA or not isinstance(payload.get("ipde_training"), Mapping):
        raise TrainingError("Export requires a trained display-depth checkpoint with provenance")
    model, architecture, _ = load_student_checkpoint(source, raft_root=raft_root, device="cpu")
    del model
    if sha256_file(source) != source_digest:
        raise TrainingIntegrityError("Source display checkpoint changed during export verification")
    provenance = ("schema", "status", "model", "mode", "units", "dataset_manifest_sha256", "architecture",
        "label_provenance", "best_epoch", "validation", "baseline_validation", "stop_reason", "warnings")
    training = {key: payload["ipde_training"][key] for key in provenance if key in payload["ipde_training"]}
    portable = {"schema": SCHEMA, "state_dict": payload["state_dict"], "architecture": architecture,
                "ipde_configuration": architecture, "ipde_training": training}
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    try:
        output = temporary / "display-model.pth"
        torch.save(portable, output)
        checked = torch.load(output, map_location="cpu", weights_only=True)
        original = payload["state_dict"]
        if set(checked["state_dict"]) != set(original) or any(not torch.equal(original[key], checked["state_dict"][key]) for key in original):
            raise TrainingError("Export changed display-depth model weights")
        reloaded, configuration, _ = load_student_checkpoint(output, raft_root=raft_root, device="cpu")
        if configuration != architecture:
            raise TrainingError("Export changed display-depth model architecture metadata")
        del reloaded
        manifest = {"schema": "ipde-display-model-export-v1", "model": "RAFT stereo depth model", "status": "trained",
            "checkpoint": "display-model.pth", "checkpoint_sha256": sha256_file(output),
            "source_checkpoint_sha256": source_digest, "architecture": architecture, "training": training,
            "weights_verified": "Strict model architecture load plus bit-exact tensor round trip",
            "source_photos_included": False, "installed_as_default": False,
            "use": "Select display-model.pth explicitly in IPDE with its compatible RAFT-Stereo source folder"}
        sidecar = {**training, "checkpoint_sha256": manifest["checkpoint_sha256"],
                   "checkpoint_path": str(folder / "display-model.pth"), "resumable": False, "exported": True}
        output.with_suffix(output.suffix + ".json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
        (temporary / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if sha256_file(source) != source_digest or report_digest and sha256_file(source_report) != report_digest:
            raise TrainingIntegrityError("Source display checkpoint changed during export")
        from .dataset_review import _publish_new_directory
        _publish_new_directory(temporary, folder)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
