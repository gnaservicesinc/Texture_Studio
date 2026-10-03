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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset import DatasetError, load_dataset, teacher_depth_to_flow
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


def _check_splits(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    left = np.load(root / sample["rgb"]["path"], mmap_mode="r", allow_pickle=False)
    raw_right = np.load(root / sample["right_rgb"]["path"], mmap_mode="r", allow_pickle=False)
    depth = np.load(root / label["target"]["path"], mmap_mode="r", allow_pickle=False)
    if left.dtype != np.uint8 or raw_right.dtype != np.uint8 or left.shape != raw_right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise TrainingError("Pretrained RAFT requires matching native uint8 RGB views; higher-bit raw arrays remain untouched")
    try:
        flow, valid, geometry = teacher_depth_to_flow(depth, sample["calibration"])
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    right, right_valid, registration = register_stereo_rows(left, raw_right)
    valid &= correspondence_validity(flow, right_valid)
    recorded_valid = np.load(root / label["valid_mask"]["path"], mmap_mode="r", allow_pickle=False)
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
        "visibility_note": "Teacher-derived z-buffer visibility is estimated, not independently measured",
    }}


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


def _tensors(torch: Any, device: str, padder_class: Any, patch: Any) -> tuple[Any, Any, Any, Any, Any]:
    left, right, flow, valid = patch
    lt = torch.from_numpy(left).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    rt = torch.from_numpy(right).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
    ft = torch.from_numpy(flow[None, None]).to(device)
    vt = torch.from_numpy(valid[None, None]).to(device)
    padder = padder_class(lt.shape, divis_by=32)
    lt, rt = padder.pad(lt, rt)
    return lt, rt, ft, vt, padder


def _evaluate(torch: Any, model: Any, device: str, padder_class: Any, samples: list[dict[str, Any]], options: TrainingOptions) -> dict[str, Any]:
    errors, patches = [], []
    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            h, w = sample["flow"].shape
            my, mx = max(0, h - options.patch_size), max(0, w - options.patch_size)
            for y, x in sorted({(0, 0), (my, mx), (my // 2, mx // 2)}):
                patch = _patch(sample, y, x, options.patch_size)
                if not patch[3].any():
                    continue
                lt, rt, ft, vt, padder = _tensors(torch, device, padder_class, patch)
                _, prediction = model(lt, rt, iters=options.iterations, test_mode=True)
                difference = (padder.unpad(prediction) - ft).abs()[vt].cpu().numpy().astype(np.float64)
                if not np.isfinite(difference).all():
                    raise TrainingError("RAFT produced nonfinite validation predictions")
                errors.append(difference)
                patches.append({"sample_index": index, "x": x, "y": y, "width": patch[0].shape[1], "height": patch[0].shape[0]})
    if not errors:
        raise TrainingError("Held-out crops lack supported correspondences; enlarge patch_size or add a scene")
    values = np.concatenate(errors)
    return {"mean_absolute_flow_error_pixels": float(values.mean()), "within_one_pixel_fraction": float(np.mean(values < 1)),
        "within_three_pixels_fraction": float(np.mean(values < 3)), "evaluated_pixels_including_overlapping_patches": int(values.size),
        "scope": "Fixed native-resolution stereo crops; overlapping pixels may count more than once", "patches": patches,
        "reference": "teacher-derived flow pseudo-label agreement" if options.mode == "distillation" else "user-supplied measured meter-depth references"}


def train_dataset(dataset_dir: Path | str, checkpoint_path: Path | str, options: TrainingOptions | None = None) -> dict[str, Any]:
    """Fine-tune real RAFT weights locally and save the best held-out checkpoint."""
    options = options or TrainingOptions()
    if options.mode not in {"distillation", "supervised"} or options.train_scope not in {"update", "full"}:
        raise TrainingError("Use distillation/supervised mode and update/full train_scope")
    if min(options.epochs, options.steps_per_epoch) < 1 or options.patch_size < 64 or options.patch_size % 32:
        raise TrainingError("Epochs/steps must be positive; native patch_size must be a multiple of 32 and at least 64")
    if not 1 <= options.iterations <= 256 or not math.isfinite(options.learning_rate) or options.learning_rate <= 0:
        raise TrainingError("RAFT iterations must be 1..256 and learning_rate finite and positive")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    report_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if checkpoint.exists() or report_path.exists():
        raise TrainingError("Checkpoint/report already exists; choose a new checkpoint path")
    root = Path(dataset_dir).expanduser().resolve()
    try:
        manifest = load_dataset(root)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    train, validation = _check_splits(manifest["samples"])
    if options.mode == "supervised" and any("reference" not in s for s in manifest["samples"]):
        raise TrainingError("Supervised training requires independently measured meter-depth references for every sample")
    chosen = [s.get("training_target_choice", "teacher") if options.mode == "distillation" else "teacher" for s in manifest["samples"]]
    if any(key not in {"teacher", "registered_display_teacher", "anchored_teacher"} or key not in sample for sample, key in zip(manifest["samples"], chosen)):
        raise TrainingError("Dataset has an unsupported distillation label choice")
    teachers = {s[key]["metadata"]["checkpoint_sha256"] for s, key in zip(manifest["samples"], chosen)}
    metric_anchors = {s[key]["metadata"].get("metric_anchor_checkpoint_sha256") for s, key in zip(manifest["samples"], chosen)}
    if options.mode == "distillation" and len(teachers) != 1:
        raise TrainingError("Distillation requires one consistent teacher checkpoint")
    if options.mode == "distillation" and len(metric_anchors) != 1:
        raise TrainingError("Distillation requires one consistent explicit metric-anchor checkpoint")
    train_arrays = [_load_sample(root, s, options) for s in train]
    validation_arrays = [_load_sample(root, s, options) for s in validation]
    print(f"RAFT experiment: {len(train)} training capture(s), {len(validation)} held-out capture(s)", file=sys.stderr, flush=True)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)
        baseline = _evaluate(torch, model, device, InputPadder, validation_arrays, options)
        print(f"Baseline held-out teacher flow error: {baseline['mean_absolute_flow_error_pixels']:.4f} px", file=sys.stderr, flush=True)
        best_metric = baseline["mean_absolute_flow_error_pixels"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch, history = 0, []
        for epoch in range(1, options.epochs + 1):
            model.train()
            model.freeze_bn()
            losses = []
            for _ in range(options.steps_per_epoch):
                sample = train_arrays[int(rng.integers(len(train_arrays)))]
                h, w = sample["flow"].shape
                for attempt in range(128):
                    y = int(rng.integers(max(0, h - options.patch_size) + 1))
                    x = int(rng.integers(max(0, w - options.patch_size) + 1))
                    patch = _patch(sample, y, x, options.patch_size)
                    if patch[3].any():
                        break
                else:
                    raise TrainingError("Cannot find valid in-crop correspondences; enlarge patch_size")
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
            evaluation = _evaluate(torch, model, device, InputPadder, validation_arrays, options)
            history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": evaluation})
            print(f"Epoch {epoch}/{options.epochs}: loss {float(np.mean(losses)):.4f}; held-out teacher flow error {evaluation['mean_absolute_flow_error_pixels']:.4f} px", file=sys.stderr, flush=True)
            if evaluation["mean_absolute_flow_error_pixels"] < best_metric:
                best_metric = evaluation["mean_absolute_flow_error_pixels"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        final = _evaluate(torch, model, device, InputPadder, validation_arrays, options)
    notes = ["Experimental RAFT teacher distillation; targets are pseudo-labels, not measured ground truth.",
        "Teacher agreement does not prove improved accuracy or eliminate teacher hallucinations.",
        "Results retain spatial_left coordinates. The display image has separate camera/framing.",
        "Trained weights are never selected automatically. Validate independent geometry before trusting displacement."]
    if len(manifest.get("group_ids", [])) < 10:
        notes.append("Fewer than ten independent groups: pipeline smoke experiment, not a validated iPhone-specific model")
    if not manifest.get("explicit_scene_groups"):
        notes.append("Scene groups were not fully supplied; undetected related captures may leak across splits")
    if not options.require_photometric_support:
        notes.append("Photometric support is recorded but not required; some teacher labels lack independent stereo evidence")
    report = {"schema": "ipde-raft-training-report-v1", "status": "experimental", "model": "RAFT-Stereo", "mode": options.mode,
        "device": device, "torch_version": str(torch.__version__),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(options).items()},
        "dataset_manifest_sha256": sha256_file(root / "dataset.json"), "teacher_checkpoint_sha256": sorted(teachers),
        "metric_anchor_checkpoint_sha256": sorted(value for value in metric_anchors if value is not None),
        "original_raft_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "raft_configuration": vars(configuration),
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "train_sample_ids": [s["id"] for s in train], "validation_sample_ids": [s["id"] for s in validation],
        "train_group_ids": sorted({s["group_id"] for s in train}), "validation_group_ids": sorted({s["group_id"] for s in validation}),
        "sample_preprocessing": [s["details"] for s in train_arrays + validation_arrays],
        "input_preprocessing": "Native stereo crops without resizing; upstream RAFT transforms new float32 RGB model tensors from 0..255 to -1..1",
        "target_preprocessing": "Metric depth converted to signed flow; invalid/occluded/out-of-crop labels masked; stored targets untouched",
        "baseline_validation": baseline, "best_epoch": best_epoch, "validation": final, "history": history, "warnings": notes}
    payload = {"state_dict": best_state, "ipde_configuration": vars(configuration), "ipde_training": report}
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
