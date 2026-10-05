"""Lossless, provenance-tracked spatial-photo datasets for depth experiments.

Teacher predictions are pseudo-labels. Saving a float32 estimate does not make
it a measured reference or restore any detail absent from the model output.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from .extractor import _asset_record, _jsonable, discover_file
from .formats import sha256_array, sha256_file, verify_npy, write_npy
from .array_storage import read_array
from .concurrency import memory_limited_workers, ordered_map, resolve_workers

if TYPE_CHECKING:
    from .learned_depth import LearnedDepthConfig, LearnedDepthResult


class DatasetError(RuntimeError):
    """A dataset could not be built without losing alignment or provenance."""


def require_dataset_teacher(model: str) -> None:
    """Keep expensive Depth Anything inference out of dataset generation."""
    if model != "depthpro":
        raise DatasetError(
            "Depth Anything generation is temporarily disabled for datasets because full-resolution "
            "processing takes many minutes per photo. Use DepthPro for datasets. DA3 and DA2 remain "
            "available for individual exports in Extractor; existing dataset maps remain usable."
        )


class _TeacherModelLoadError(DatasetError):
    """A shared model load failure is handled once for the whole phase."""


@dataclass(frozen=True)
class DatasetOptions:
    teacher: LearnedDepthConfig
    group_ids: Mapping[str, str] | None = None
    validation_fraction: float = 0.2
    split_seed: int = 0
    reference_paths: Mapping[str, Path] | None = None
    include_display_teacher: bool = True
    prefer_registered_display_teacher: bool = False
    teacher_view: str = "display"
    grouping_semantics: str = "capture"
    metric_anchor: LearnedDepthConfig | None = None
    additional_teachers: tuple[LearnedDepthConfig, ...] = ()
    teacher_ids: tuple[str, ...] = ()
    skip_bad_photos: bool = True
    name: str | None = None
    category: str | None = None
    require_apple_camera: bool = False
    compress_arrays: bool = True
    array_format: str = "images"
    retain_intermediates: bool = False
    preserve_auxiliary_assets: bool = False
    workers: int | None = None


def teacher_depth_to_flow(depth: np.ndarray, calibration: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert a metric LEFT-grid pseudo-label to RAFT's signed horizontal flow.

    This is a calibrated representation change, not independent confirmation of
    the teacher's meter scale. The presentation disparity adjustment is excluded.
    """
    target = np.asarray(depth)
    if target.ndim != 2 or target.dtype.kind != "f" or not target.size:
        raise DatasetError("Metric teacher depth must be a nonempty floating-point HxW array")
    if not calibration.get("raft_stereo_ready"):
        raise DatasetError("RAFT targets require validated rectified stereo camera calibration")
    try:
        focal = float(calibration["focal_length_pixels_for_depth"])
        baseline = float(calibration["baseline_meters"])
        offset = float(calibration["principal_point_delta_x_pixels"])
        shape = (int(calibration["left_camera"]["height"]), int(calibration["left_camera"]["width"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError("Incomplete metric stereo calibration") from exc
    if shape != target.shape or not all(math.isfinite(value) for value in (focal, baseline, offset)) or min(focal, baseline) <= 0:
        raise DatasetError("Metric stereo calibration/grid is inconsistent with the teacher target")
    valid = np.isfinite(target) & (target > 0)
    flow = np.full(target.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        flow[valid] = np.float32(offset) - np.float32(focal * baseline) / target[valid]
    xr = np.arange(target.shape[1], dtype=np.float32)[None, :] + flow
    valid &= np.isfinite(flow) & (np.abs(flow) < 700) & (xr >= 0) & (xr <= target.shape[1] - 1)
    flow[~valid] = np.nan
    return flow, valid, {
        "units": "signed horizontal correspondence pixels",
        "label_kind": "metric_teacher_derived_pseudo_label",
        "reference": "spatial_left",
        "formula": "x_right - x_left = (cx_right - cx_left) - focal_pixels * baseline_meters / teacher_depth_meters",
        "presentation_disparity_adjustment_applied": False,
        "focal_length_pixels": focal,
        "baseline_meters": baseline,
        "principal_point_delta_x_pixels": offset,
        "validity": "Positive finite metric estimates; finite |flow|<700; right correspondence inside the image",
        "precision_note": "Correct flow units/sign do not establish the accuracy of the teacher's estimated meter scale",
    }


def _infer_teacher(*args: Any, **kwargs: Any) -> LearnedDepthResult:
    from .learned_depth import infer_learned_depth

    return infer_learned_depth(*args, **kwargs)


def _array_record(root: Path, path: Path, value: np.ndarray) -> dict[str, Any]:
    write_npy(path, value)
    verify_npy(path, value)
    return {
        "path": path.relative_to(root).as_posix(),
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "array_sha256": sha256_array(value),
        "file_sha256": sha256_file(path),
    }


def _lookup(mapping: Mapping[str, Any] | None, source: Path) -> Any:
    if mapping is None:
        return None
    # Basename lookup is deliberately excluded: different cameras/directories
    # can contain the same IMG_1234 filename.
    return mapping.get(str(source))


def _burst_ids(value: Any) -> set[str]:
    """Read explicit BurstUUID fields/XMP, never infer a burst from filenames."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {"burstuuid", "burstidentifier", "burstid"}:
                if isinstance(item, str) and item.strip():
                    found.add(item.strip())
            found.update(_burst_ids(item))
        if value.get("encoding") == "base64" and isinstance(value.get("data"), str):
            try:
                found.update(_burst_ids(base64.b64decode(value["data"], validate=True)))
            except ValueError:
                pass
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_burst_ids(item))
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
        found.update(re.findall(r"(?:BurstUUID|BurstIdentifier)\s*=\s*[\"']([^\"']+)", text))
        found.update(re.findall(r"<(?:\w+:)?(?:BurstUUID|BurstIdentifier)>([^<]+)</", text))
    return found


def assign_grouped_splits(samples: list[dict[str, Any]], fraction: float, seed: int) -> list[str]:
    """Union duplicate photos, duplicate RGB, explicit scenes, and known bursts.

    Deterministic group-level splitting happens before patches are drawn. No
    crop, right-eye image, or duplicate of a photo can cross the split boundary.
    Unknown scene relationships still require the user's explicit group map.
    """
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise DatasetError("validation_fraction must be strictly between zero and one")
    parents = list(range(len(samples)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[str, int] = {}
    for i, sample in enumerate(samples):
        tokens = [f"source:{sample['source_sha256']}", f"rgb:{sample['rgb']['array_sha256']}"]
        if sample.get("requested_group"):
            tokens.append(f"scene:{sample['requested_group']}")
        tokens.extend(f"burst:{item}" for item in sample.get("burst_ids", []))
        for token in tokens:
            if token in seen:
                parents[root(i)] = root(seen[token])
            seen[token] = i
    components: dict[int, list[int]] = {}
    for i in range(len(samples)):
        components.setdefault(root(i), []).append(i)
    groups: list[str] = []
    for indices in components.values():
        # Adding/removing teacher variants must never change a photo's group.
        hashes = sorted({samples[i]["source_sha256"] for i in indices})
        group = hashlib.sha256("\n".join(hashes).encode()).hexdigest()[:24]
        groups.append(group)
        for i in indices:
            samples[i]["group_id"] = group
    order = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    count = min(len(order) - 1, max(1, round(len(order) * fraction))) if len(order) >= 2 else 0
    validation = set(order[:count])
    for sample in samples:
        sample["split"] = "validation" if sample["group_id"] in validation else "train"
    return sorted(groups)


def build_dataset(
    sources: Sequence[Path | str], output_dir: Path | str, options: DatasetOptions,
    **kwargs: Any,
) -> dict[str, Any]:
    """Hold generation ownership for CLI and direct Python callers alike."""
    from .resource_lock import resource_lock
    for teacher in (options.teacher, *options.additional_teachers):
        require_dataset_teacher(teacher.model)
    destination = Path(output_dir).expanduser().resolve()
    with resource_lock(destination / ".ipde-generation-owner"):
        return _build_dataset(sources, destination, options, **kwargs)


def _build_dataset(
    sources: Sequence[Path | str],
    output_dir: Path | str,
    options: DatasetOptions,
    *,
    teacher_results: Mapping[str, LearnedDepthResult] | None = None,
    display_teacher_results: Mapping[str, LearnedDepthResult] | None = None,
    metric_anchor_results: Mapping[str, LearnedDepthResult] | None = None,
    display_metric_anchor_results: Mapping[str, LearnedDepthResult] | None = None,
    teacher_results_by_id: Mapping[str, Mapping[str, LearnedDepthResult]] | None = None,
    display_teacher_results_by_id: Mapping[str, Mapping[str, LearnedDepthResult]] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a new dataset with atomic, progressively readable snapshots.

    All discovered auxiliary/sample arrays retain their original dtype and
    bytes. The left and right views retain their original grids/calibration.
    Only the teacher's documented inference preprocessing modifies model input.
    Optional measured references must be positive depth in meters, stored as an
    HxW float NPY on the exact left-view grid. They are never implicitly resized.
    Completed samples are readable from the staging directory reported through
    progress_callback. Its dataset.json is generation-owned: curation must use
    a separate reviewed copy until dataset_complete reports the final path.
    """
    try:
        worker_count = resolve_workers(options.workers)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    destination = Path(output_dir).expanduser().resolve()
    inputs = [Path(source).expanduser().resolve() for source in sources]
    if not inputs:
        raise DatasetError("Select at least one original spatial HEIC photo")
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if len(set(inputs)) != len(inputs):
        raise DatasetError("A source path was selected more than once")
    if options.grouping_semantics not in {"capture", "scene", "none"}:
        raise DatasetError("grouping_semantics must be capture, scene or none")
    if options.teacher_view not in {"display", "stereo-left"}:
        raise DatasetError("teacher_view must be display or stereo-left")
    if options.array_format not in {"images", "numpy"}:
        raise DatasetError("array_format must be images or numpy")
    teachers = (options.teacher, *options.additional_teachers)
    if len(teachers) > 3:
        raise DatasetError("Choose one, two or three teachers")
    if options.teacher_ids and len(options.teacher_ids) != len(teachers):
        raise DatasetError("teacher_ids must contain one unique ID per selected teacher")
    teacher_ids = list(options.teacher_ids) or [
        f"{getattr(config, 'model', 'teacher')}-{index + 1}" for index, config in enumerate(teachers)
    ]
    if len(set(teacher_ids)) != len(teacher_ids) or any(not isinstance(value, str) or not value.strip() for value in teacher_ids):
        raise DatasetError("Teacher IDs must be unique nonempty strings")
    # Fail before inference for malformed splitting options.
    assign_grouped_splits([], options.validation_fraction, options.split_seed)
    normalized_groups = {
        str(Path(key).expanduser().resolve()): str(value).strip()
        for key, value in (options.group_ids or {}).items()
    }
    if any(not value for value in normalized_groups.values()):
        raise DatasetError("Scene/burst group labels must be nonempty")
    references = {
        str(Path(key).expanduser().resolve()): Path(value).expanduser().resolve()
        for key, value in (options.reference_paths or {}).items()
    }
    teacher_results = {str(Path(key).expanduser().resolve()): value for key, value in (teacher_results or {}).items()}
    display_teacher_results = {str(Path(key).expanduser().resolve()): value for key, value in (display_teacher_results or {}).items()}
    metric_anchor_results = {str(Path(key).expanduser().resolve()): value for key, value in (metric_anchor_results or {}).items()}
    display_metric_anchor_results = {str(Path(key).expanduser().resolve()): value for key, value in (display_metric_anchor_results or {}).items()}
    teacher_results_by_id = {teacher_id: {str(Path(key).expanduser().resolve()): value for key, value in values.items()}
                             for teacher_id, values in (teacher_results_by_id or {}).items()}
    display_teacher_results_by_id = {teacher_id: {str(Path(key).expanduser().resolve()): value for key, value in values.items()}
                                     for teacher_id, values in (display_teacher_results_by_id or {}).items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    predictors: dict[str, Any] = {}
    cached_anchors: dict[tuple[str, bool], dict[str, Any]] = {}
    anchor_errors: dict[tuple[str, bool], str] = {}
    array_sizes: dict[str, tuple[int, int]] = {}
    stored_arrays: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}

    def dataset_array_record(root: Path, path: Path, value: np.ndarray) -> dict[str, Any]:
        from .array_storage import array_record
        identity = (value.dtype.str, value.shape, sha256_array(value))
        if identity in stored_arrays:
            # Byte-identical source copies, duplicate HEIF views and repeated
            # teacher planes share one immutable lossless file.
            return dict(stored_arrays[identity])
        record = array_record(root, path, value, compressed=options.compress_arrays, storage=options.array_format)
        array_sizes[record["path"]] = ((root / record["path"]).stat().st_size, value.nbytes)
        stored_arrays[identity] = record
        return record

    def store_raw_assets(discovery: Any, folder: Path) -> dict[int, dict[str, Any]]:
        """Write unique raw planes concurrently; only this caller owns dedup metadata."""
        from .array_storage import array_record
        selected = [(ordinal, asset) for ordinal, asset in enumerate(discovery.assets)
                    if options.preserve_auxiliary_assets or asset.kind == "display_view"
                    or asset.kind == "spatial_view" and asset.semantic_name in {"spatial_left", "spatial_right"}]
        identities = [(asset.array.dtype.str, asset.array.shape, sha256_array(asset.array))
                      for _, asset in selected]
        planned = {}
        for (ordinal, asset), identity in zip(selected, identities):
            if identity not in stored_arrays and identity not in planned:
                planned[identity] = (folder / f"raw-{ordinal:03d}.npy", asset.array)

        def save(item: Any) -> Any:
            identity, (path, array) = item
            record = array_record(temporary, path, array, compressed=options.compress_arrays, storage=options.array_format)
            return identity, record, (temporary / record["path"]).stat().st_size, array.nbytes

        count = memory_limited_workers(worker_count, max((value[1].nbytes for value in planned.values()), default=1) * 2)
        for identity, record, file_bytes, array_bytes in ordered_map(save, planned.items(), workers=count):
            stored_arrays[identity] = record
            array_sizes[record["path"]] = file_bytes, array_bytes
        raw = {}
        for (ordinal, asset), identity in zip(selected, identities):
            record = _asset_record(asset)
            record["storage"] = dict(stored_arrays[identity])
            record["metadata_blocks"] = _jsonable(asset.metadata_blocks)
            raw[ordinal] = record
        return raw

    def discard(folder: Path) -> None:
        shutil.rmtree(folder, ignore_errors=True)
        prefix = folder.relative_to(temporary).as_posix() + "/"
        for name in list(array_sizes):
            if name.startswith(prefix):
                del array_sizes[name]
        for identity, record in list(stored_arrays.items()):
            if record["path"].startswith(prefix):
                del stored_arrays[identity]

    def predict(array: np.ndarray, *, anchor: bool = False, teacher_index: int = 0, **kwargs: Any) -> LearnedDepthResult:
        name = "metric_anchor" if anchor else teacher_ids[teacher_index]
        from .learned_depth import validate_learned_depth_input
        config = options.metric_anchor if anchor else teachers[teacher_index]
        validate_learned_depth_input(array.shape, config)
        if name not in predictors:
            if predictors:
                raise DatasetError("The previous teacher must be released before loading another model")
            from .learned_depth import LearnedDepthPredictor
            emit("model_loading", dataset_dir=str(temporary), teacher_id=name,
                 model=getattr(config, "model", None), device=getattr(config, "device", None))
            try:
                predictors[name] = LearnedDepthPredictor(config)
            except Exception as exc:
                raise _TeacherModelLoadError(f"Cannot load teacher {name}: {exc}") from exc
            emit("model_loaded", dataset_dir=str(temporary), teacher_id=name,
                 model=getattr(config, "model", None))
        return predictors[name](array, **kwargs)

    def release_predictors(*, report: bool = True) -> None:
        while predictors:
            name, predictor = predictors.popitem()
            close = getattr(predictor, "close", None)
            try:
                if report:
                    emit("model_releasing", dataset_dir=str(temporary), teacher_id=name)
            finally:
                if close is not None:
                    close()
            del predictor
            if report:
                emit("model_released", dataset_dir=str(temporary), teacher_id=name)

    def cached_anchor_result(source: Path, *, display: bool) -> LearnedDepthResult | None:
        cached = cached_anchors.get((str(source), display))
        if cached is None:
            return None
        from .learned_depth import LearnedDepthResult
        return LearnedDepthResult(read_array(temporary / cached["native"]["path"]),
                                  read_array(temporary / cached["target"]["path"]), cached["metadata"],
                                  read_array(temporary / cached["confidence"]["path"]) if "confidence" in cached else None)

    def anchored_label(
        source: Path, rgb: np.ndarray, prediction: LearnedDepthResult, folder: Path,
        *, display: bool, focal_pixels: float | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        if prediction.metadata["units"] == "meters" or options.metric_anchor is None:
            return None, None, None
        print(f"  Applying explicit metric-model anchor on the {'DISPLAY' if display else 'LEFT'} grid", file=sys.stderr, flush=True)
        anchor_prediction = _lookup(display_metric_anchor_results if display else metric_anchor_results, source)
        if anchor_prediction is None:
            key = str(source), display
            if key in anchor_errors:
                raise DatasetError(f"Metric anchor failed for this photo: {anchor_errors[key]}")
            anchor_prediction = cached_anchor_result(source, display=display)
            if anchor_prediction is None:
                raise DatasetError("Metric anchor was not prepared before the teacher phase")
        anchor_depth = np.asarray(anchor_prediction.source_depth)
        anchor_native = np.asarray(anchor_prediction.native_depth)
        if anchor_prediction.metadata.get("units") != "meters" or anchor_depth.shape != rgb.shape[:2] or anchor_depth.dtype != np.float32:
            raise DatasetError("Metric anchor must return float32 meters on the exact same RGB grid")
        if anchor_prediction.metadata.get("input_rgb_sha256") != sha256_array(rgb):
            raise DatasetError("Metric anchor RGB provenance differs from the relative teacher")
        if not anchor_prediction.metadata.get("checkpoint_sha256") or anchor_native.ndim != 2 or anchor_native.dtype != np.float32:
            raise DatasetError("Metric anchor must identify a checkpoint and preserve its float32 native plane")
        prefix = "display-" if display else ""
        anchor_record = {
            "label_kind": "metric_model_anchor_pseudo_label", "units": "meters", "metadata": _jsonable(anchor_prediction.metadata),
            "target": dataset_array_record(temporary, folder / f"{prefix}metric-anchor.npy", anchor_depth),
            "native_prediction_retained": options.retain_intermediates,
        }
        anchor_record["metadata"]["native_prediction_shape"] = list(anchor_native.shape)
        if options.retain_intermediates:
            anchor_record["native_target"] = dataset_array_record(temporary, folder / f"{prefix}metric-anchor-native.npy", anchor_native)
        from .pseudo_calibration import PseudoCalibrationError, anchor_relative_depth
        try:
            anchored, accepted, calibration = anchor_relative_depth(prediction.source_depth, prediction.metadata["units"], anchor_depth)
        except PseudoCalibrationError as exc:
            return anchor_record, None, {"accepted": False, "reason": str(exc), "anchor_is_measured": False}
        calibration = _jsonable(calibration)
        anchored_metadata = {**_jsonable(prediction.metadata), "units": "meters", "original_units": prediction.metadata["units"],
            "metric_anchor_checkpoint_sha256": anchor_prediction.metadata["checkpoint_sha256"], "pseudo_calibration": calibration}
        anchored_record = {
            "label_kind": "model_anchored_pseudo_depth", "units": "meters", "metadata": anchored_metadata,
            "target": dataset_array_record(temporary, folder / f"{prefix}anchored-teacher.npy", anchored),
            "valid_mask": dataset_array_record(temporary, folder / f"{prefix}anchored-valid.npy", accepted),
            "calibration": calibration,
        }
        return anchor_record, anchored_record, calibration

    samples: list[dict[str, Any]] = []
    warnings = [
        "Teacher targets are model-generated pseudo-labels, not measured ground truth. "
        "Distillation agreement cannot establish absolute accuracy or recover missing detail.",
        "Known duplicates and reported burst identifiers stay in one split. Supply group_ids "
        "for related captures of the same scene: scene identity cannot be reliably inferred from HEIC metadata.",
    ]
    skipped: list[dict[str, Any]] = []
    event_sequence = 0

    def emit(event: str, **details: Any) -> None:
        nonlocal event_sequence
        event_sequence += 1
        payload = {"event": event, "sequence": event_sequence, **details}
        journal_root = temporary if temporary.exists() else destination
        with (journal_root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
            stream.flush()
        if progress_callback is not None:
            progress_callback(payload)

    def snapshot(state: str, processed_sources: int) -> dict[str, Any]:
        groups = assign_grouped_splits(samples, options.validation_fraction, options.split_seed)
        current_warnings = list(warnings)
        if len(groups) < 2:
            current_warnings.append("Only one independent group is present; training requires another held-out scene/group")
        manifest = {
            "schema": "ipde-depth-dataset-v1",
            "name": options.name or destination.name,
            "category": options.category,
            "precision_policy": "Raw samples/auxiliaries preserved bit-for-bit for retained assets; no normalization, gamma, or resampling",
            "array_storage": "lossless_exr_png_npz" if options.compress_arrays and options.array_format == "images" else "npz_deflate" if options.compress_arrays else "npy",
            "storage_policy": {"array_format": options.array_format,
                               "retain_intermediates": options.retain_intermediates,
                               "preserve_auxiliary_assets": options.preserve_auxiliary_assets,
                               "computed_positive_finite_masks": "stored" if options.retain_intermediates else "derived_on_read"},
            "file_workers": worker_count,
            "teacher_view": options.teacher_view,
            "generation_state": state,
            "generation_output_dir": str(destination),
            "splits_provisional": state != "complete",
            "split_seed": options.split_seed,
            "validation_fraction": options.validation_fraction,
            "group_ids": groups,
            "explicit_scene_groups": options.grouping_semantics == "scene" and bool(samples) and all(sample["requested_group"] for sample in samples),
            "grouping_semantics": options.grouping_semantics,
            "teachers": [{"id": teacher_id, "model": getattr(config, "model", None)} for teacher_id, config in zip(teacher_ids, teachers)],
            "warnings": current_warnings,
            "skipped_sources": list(skipped),
            "samples": samples,
            "summary": {
                "samples": len(samples), "source_photos": len({sample["source_sha256"] for sample in samples}),
                "groups": len(groups), "processed_sources": processed_sources, "total_sources": len(inputs),
                "skipped_entries": len(skipped),
                "array_storage_bytes": sum(size[0] for size in array_sizes.values()),
                "array_sample_bytes": sum(size[1] for size in array_sizes.values()),
                "train_samples": sum(sample["split"] == "train" for sample in samples),
                "validation_samples": sum(sample["split"] == "validation" for sample in samples),
            },
        }
        manifest_path = temporary / "dataset.json"
        staging = temporary / ".dataset.json-writing"
        with staging.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(manifest_path)
        return manifest

    def skip(source: Path, error: Exception, *, teacher_id: str | None = None) -> None:
        if not options.skip_bad_photos:
            raise error
        record = {"source_path": str(source), "teacher_id": teacher_id, "reason": str(error), "error_type": type(error).__name__}
        skipped.append(record)
        print(f"  Skipped {source.name}: {error}", file=sys.stderr, flush=True)
        emit("photo_skipped", dataset_dir=str(temporary), **record)

    def make_sample(source: Path, discovery: Any, index: int, teacher_index: int, raw: Mapping[int, dict[str, Any]],
                    left: Any, right: Any, left_index: int, right_index: int, source_id: str,
                    photo_metadata: dict[str, Any], folder: Path) -> dict[str, Any]:
        spatial = discovery.spatial_photo
        teacher_id = teacher_ids[teacher_index]
        sample_id = source_id if len(teachers) == 1 else f"{source_id}-teacher-{teacher_index + 1}"
        display_indices = [i for i, asset in enumerate(discovery.assets) if asset.kind == "display_view"]
        display_mode = options.teacher_view == "display"
        if display_mode and len(display_indices) != 1:
            raise DatasetError("Display-teacher datasets require exactly one full display image; stereo-view fallback is disabled")
        teacher_rgb = discovery.assets[display_indices[0]].array if display_mode else left.array
        prediction = _lookup((display_teacher_results_by_id if display_mode else teacher_results_by_id).get(teacher_id), source)
        if prediction is None and teacher_index == 0:
            prediction = _lookup(display_teacher_results if display_mode else teacher_results, source)
        if prediction is None and teachers[teacher_index] == options.metric_anchor:
            prediction = cached_anchor_result(source, display=display_mode)
        if prediction is None:
            print("  Running selected teacher on the full display image" if display_mode else
                  "  Running selected teacher on the calibrated LEFT view (explicit legacy workflow)", file=sys.stderr, flush=True)
            prediction = predict(
                teacher_rgb, teacher_index=teacher_index,
                focal_pixels=None if display_mode else spatial["left_camera"]["focal_length_x_pixels"],
                reference_label="display" if display_mode else "spatial_left",
            )
        target = np.asarray(prediction.source_depth)
        native = np.asarray(prediction.native_depth)
        if target.shape != teacher_rgb.shape[:2] or target.dtype != np.float32:
            raise DatasetError("Teacher output must be float32 on the exact selected RGB HxW grid")
        if native.ndim != 2 or native.dtype != np.float32:
            raise DatasetError("Teacher native output must be a float32 HxW plane")
        metadata = dict(prediction.metadata)
        if metadata.get("input_rgb_sha256") != sha256_array(teacher_rgb):
            raise DatasetError("Teacher RGB provenance does not match the extracted display image" if display_mode else
                               "Teacher RGB provenance does not match the extracted left view")
        metadata.update({"dataset_teacher_view": options.teacher_view,
                         "reference_label": "display" if display_mode else "spatial_left",
                         "input_rgb_shape": list(teacher_rgb.shape),
                         "stored_target_shape": list(target.shape),
                         "native_prediction_shape": list(native.shape)})
        units = metadata.get("units")
        if units not in {"meters", "relative_inverse_depth", "relative_depth"}:
            raise DatasetError(f"Unknown teacher output units: {units!r}")
        if not isinstance(metadata.get("checkpoint_sha256"), str) or not metadata["checkpoint_sha256"]:
            raise DatasetError("Teacher metadata must identify the checkpoint SHA-256")
        valid = np.isfinite(target) & (target > 0)
        if not valid.any():
            raise DatasetError(f"{source.name}: teacher returned no finite positive depth values")
        sample: dict[str, Any] = {
            "id": sample_id,
            "source_id": source_id,
            "source_photo_id": source_id,
            "teacher_id": teacher_id,
            "teacher_model": getattr(teachers[teacher_index], "model", metadata.get("model_id")),
            "teacher_view": options.teacher_view,
            "photo_metadata": photo_metadata,
            "source_path": str(source),
            "source_sha256": discovery.source_sha256,
            "source_bytes": discovery.source_size,
            "requested_group": normalized_groups.get(str(source)) if options.grouping_semantics != "none" else None,
            "burst_ids": sorted(_burst_ids(discovery.top_level_images)),
            "rgb": {**raw[left_index]["storage"], "source_bit_depth": left.source_bit_depth},
            "right_rgb": {**raw[right_index]["storage"], "source_bit_depth": right.source_bit_depth},
            "coordinate_reference": "spatial_left: exact decoded sample coordinates, no EXIF rotation",
            "calibration": _jsonable(spatial),
            "top_level_images": _jsonable(discovery.top_level_images),
            "raw_assets": list(raw.values()),
            "teacher": {
                "label_kind": "pseudo_label",
                "units": units,
                "metadata": _jsonable(metadata),
                "target": dataset_array_record(temporary, folder / "teacher.npy", target),
                "native_prediction_retained": options.retain_intermediates,
                "validity_policy": "stored_mask" if options.retain_intermediates else "positive_finite",
                "valid_pixel_count": int(valid.sum()),
                "coordinate_reference": "display; separate from RAFT's spatial_left reference" if display_mode else "spatial_left",
            },
        }
        if options.retain_intermediates:
            sample["teacher"]["native_target"] = dataset_array_record(temporary, folder / "teacher-native.npy", native)
            sample["teacher"]["valid_mask"] = dataset_array_record(temporary, folder / "teacher-valid.npy", valid)
        anchor_record, anchored_record, pseudo_calibration = (None, None, None) if display_mode else anchored_label(
            source, left.array, prediction, folder, display=False,
            focal_pixels=spatial["left_camera"]["focal_length_x_pixels"],
        )
        if anchor_record is not None:
            sample["metric_anchor"] = anchor_record
            sample["pseudo_calibration"] = pseudo_calibration
        if anchored_record is not None:
            sample["anchored_teacher"] = anchored_record
            sample["training_target_choice"] = "anchored_teacher"
        training_target = None if display_mode else target if units == "meters" else read_array(temporary / anchored_record["target"]["path"]) if anchored_record else None
        if training_target is not None and spatial.get("raft_stereo_ready"):
            flow, flow_valid, flow_details = teacher_depth_to_flow(training_target, spatial)
            sample["raft_target"] = {
                "source_label": "anchored_teacher" if anchored_record else "teacher",
                "target": dataset_array_record(temporary, folder / "raft-teacher-flow.npy", flow),
                "valid_mask": dataset_array_record(temporary, folder / "raft-teacher-valid.npy", flow_valid),
                "metadata": flow_details,
            }
        if display_indices:
            display_index = display_indices[0]
            display = discovery.assets[display_index]
            sample["display_rgb"] = {**raw[display_index]["storage"], "source_bit_depth": display.source_bit_depth}
            sample["display_note"] = "Separate display camera/framing; never resized onto the left/right stereo grid"
            display_prediction = prediction if display_mode else _lookup(display_teacher_results_by_id.get(teacher_id), source)
            if display_prediction is None and teacher_index == 0:
                display_prediction = _lookup(display_teacher_results, source)
            if display_prediction is None and options.include_display_teacher and teachers[teacher_index] == options.metric_anchor:
                display_prediction = cached_anchor_result(source, display=True)
            if display_prediction is None and options.include_display_teacher:
                print("  Running selected teacher on the full display image", file=sys.stderr, flush=True)
                display_prediction = predict(display.array, teacher_index=teacher_index, reference_label="display")
            if display_prediction is not None:
                display_target = np.asarray(display_prediction.source_depth)
                if display_target.shape != display.array.shape[:2] or display_target.dtype != np.float32:
                    raise DatasetError("Display teacher must use the exact display grid")
                if display_prediction.metadata.get("input_rgb_sha256") != sha256_array(display.array):
                    raise DatasetError("Display teacher provenance does not match the display image")
                display_metadata = {**display_prediction.metadata, "dataset_teacher_view": "display",
                    "reference_label": "display", "input_rgb_shape": list(display.array.shape),
                    "stored_target_shape": list(display_target.shape),
                    "native_prediction_shape": list(np.shape(display_prediction.native_depth))}
                if display_metadata.get("units") not in {"meters", "relative_inverse_depth", "relative_depth"}:
                    raise DatasetError("Display teacher has unsupported units")
                if not isinstance(display_metadata.get("checkpoint_sha256"), str) or not display_metadata["checkpoint_sha256"]:
                    raise DatasetError("Display teacher must identify its checkpoint SHA-256")
                if np.asarray(display_prediction.native_depth).ndim != 2 or np.asarray(display_prediction.native_depth).dtype != np.float32:
                    raise DatasetError("Display teacher native output must be a float32 HxW plane")
                if not (np.isfinite(display_target) & (display_target > 0)).any():
                    raise DatasetError("Display teacher returned no finite positive target values")
                sample["display_teacher"] = {
                    "label_kind": "pseudo_label",
                    "units": display_prediction.metadata["units"],
                    "metadata": _jsonable(display_metadata),
                    "target": dataset_array_record(temporary, folder / "display-teacher.npy", display_target),
                    "native_prediction_retained": options.retain_intermediates,
                    "validity_policy": "stored_mask" if options.retain_intermediates else "positive_finite",
                    "coordinate_reference": "display; separate from RAFT's spatial_left reference",
                }
                if options.retain_intermediates:
                    sample["display_teacher"]["native_target"] = dataset_array_record(temporary, folder / "display-teacher-native.npy", display_prediction.native_depth)
                    sample["display_teacher"]["valid_mask"] = dataset_array_record(temporary, folder / "display-teacher-valid.npy", np.isfinite(display_target) & (display_target > 0))
                display_anchor, anchored_display, display_calibration = anchored_label(
                    source, display.array, display_prediction, folder, display=True, focal_pixels=None,
                )
                if display_anchor is not None:
                    display_anchor["coordinate_reference"] = "display"
                    sample["display_metric_anchor"] = display_anchor
                    sample["display_pseudo_calibration"] = display_calibration
                if anchored_display is not None:
                    anchored_display["coordinate_reference"] = "display"
                    sample["anchored_display_teacher"] = anchored_display
                if display_mode:
                    # The display camera/framing is an independent grid. A
                    # distance-dependent stereo projection is not a fixed image
                    # registration, and cannot become a stock LEFT-flow label.
                    sample["teacher"] = copy.deepcopy(sample["display_teacher"])
                    sample["teacher"]["alias_of"] = "display_teacher"
                    sample["teacher"]["valid_pixel_count"] = int(valid.sum())
                    sample["training_target_choice"] = "anchored_display_teacher" if anchored_display else "display_teacher"
                    sample["training_target_choice_note"] = (
                        "Full display-grid labels are preserved without camera registration. Stock RAFT predicts native LEFT-grid flow; "
                        "these targets require a display-grid student. Native-stereo teacher fallback is disabled.")
                    sample["display_grid_policy"] = "Preserve full display coordinates; no automatic fixed registration or stereo-view inference"
                else:
                    display_training_target = display_target if display_metadata["units"] == "meters" else read_array(temporary / anchored_display["target"]["path"]) if anchored_display else None
                    from .registration import RegistrationError, estimate_display_registration, register_display_depth
                    try:
                        registration = estimate_display_registration(discovery)
                    except RegistrationError as exc:
                        registration = {"accepted": False, "reason": str(exc), "source_sha256": discovery.source_sha256}
                    print(f"  Display registration (legacy diagnostic): {registration.get('reason')}", file=sys.stderr, flush=True)
                    sample["display_registration"] = _jsonable(registration)
                    if registration.get("accepted") and display_training_target is not None:
                        try:
                            registered, registered_valid = register_display_depth(display_training_target, discovery, registration)
                        except RegistrationError as exc:
                            registration = {**registration, "accepted": False, "reason": str(exc)}
                            sample["display_registration"] = _jsonable(registration)
                        else:
                            sample["registered_display_teacher"] = {
                                "label_kind": "registered_display_teacher_pseudo_label",
                                "units": "meters",
                                "target": dataset_array_record(temporary, folder / "registered-display-teacher.npy", registered),
                                "valid_mask": dataset_array_record(temporary, folder / "registered-display-valid.npy", registered_valid),
                                "reference_role": registration["reference_role"],
                                "coordinate_reference": "spatial_left" if registration["reference_role"] == "left" else "spatial_right",
                                "metadata": {**(anchored_display["metadata"] if anchored_display else _jsonable(display_metadata)),
                                             "dataset_teacher_view": "display", "reference_label": "display"},
                                "registration": _jsonable(registration),
                                "precision_note": "Approximate same-camera depth transport only inside independently validated registration cells",
                            }
                    use_display = options.prefer_registered_display_teacher and "registered_display_teacher" in sample and registration.get("reference_role") == "left"
                    sample["training_target_choice"] = "registered_display_teacher" if use_display else "anchored_teacher" if anchored_record else "teacher"
                    sample["training_target_choice_note"] = (
                        "Explicit legacy supervision uses empirically registered LEFT-camera display labels; holes stay excluded"
                        if use_display else "Explicit legacy supervision uses the separate LEFT-grid teacher; display diagnostics are not selected")
        if options.retain_intermediates and getattr(prediction, "confidence", None) is not None:
            confidence = np.asarray(prediction.confidence)
            sample["teacher"]["confidence"] = dataset_array_record(temporary, folder / "teacher-confidence.npy", confidence)
            sample["teacher"]["confidence_note"] = "Model confidence is not a measured error bound"
            if display_mode:
                sample["display_teacher"]["confidence"] = dict(sample["teacher"]["confidence"])
                sample["display_teacher"]["confidence_note"] = sample["teacher"]["confidence_note"]
        reference_path = _lookup(references, source)
        if reference_path is not None:
            try:
                reference = np.load(reference_path, allow_pickle=False)
            except (ValueError, OSError) as exc:
                raise DatasetError(f"Cannot read measured reference NPY: {reference_path}: {exc}") from exc
            if reference.shape != left.array.shape[:2] or reference.dtype.kind != "f":
                raise DatasetError("Measured reference must be a floating-point meter-depth NPY on the exact left grid")
            reference_valid = np.isfinite(reference) & (reference > 0)
            if not reference_valid.any():
                raise DatasetError("Measured reference contains no finite positive meter-depth samples")
            sample["reference"] = {
                "label_kind": "user_supplied_measured_reference",
                "units": "meters",
                "source_path": str(reference_path),
                "source_sha256": sha256_file(reference_path),
                "target": dataset_array_record(temporary, folder / "reference.npy", reference),
                "validity_policy": "stored_mask" if options.retain_intermediates else "positive_finite",
                "accuracy_note": "User must verify measurement accuracy and left-camera registration independently",
            }
            if options.retain_intermediates:
                sample["reference"]["valid_mask"] = dataset_array_record(temporary, folder / "reference-valid.npy", reference_valid)
        print(f"  Preserved {len(raw)} raw arrays and teacher targets", file=sys.stderr, flush=True)
        return sample

    def prepare_source(source: Path) -> Any:
        try:
            if source.suffix.lower() not in {".heic", ".heif", ".hif"}:
                raise DatasetError("Dataset inputs must be original HEIC/HEIF files, not depth previews")
            discovery = discover_file(source)
            from .spatial_scan import validate_spatial_discovery
            metadata = validate_spatial_discovery(discovery, require_apple_camera=options.require_apple_camera)
            if options.teacher_view == "display" and sum(asset.kind == "display_view" for asset in discovery.assets) != 1:
                raise DatasetError("Display-teacher datasets require exactly one full display image; stereo-view fallback is disabled")
            return discovery, metadata
        except Exception as exc:
            return exc

    def restore_source(entry: dict[str, Any]) -> Any:
        """Reload one photo's required views, without retaining decoded HEICs."""
        discovery = copy.copy(entry["discovery"])
        discovery.assets = [copy.copy(asset) for asset in discovery.assets]
        for ordinal, record in entry["raw"].items():
            asset = discovery.assets[ordinal]
            if asset.kind == "display_view" or asset.kind == "spatial_view" and asset.semantic_name in {"spatial_left", "spatial_right"}:
                asset.array = read_array(temporary / record["storage"]["path"])
        return discovery

    def needs_anchor(source: Path, *, display: bool) -> bool:
        for teacher_index, config in enumerate(teachers):
            mapping = display_teacher_results_by_id if display else teacher_results_by_id
            prediction = _lookup(mapping.get(teacher_ids[teacher_index]), source)
            if prediction is None and teacher_index == 0:
                prediction = _lookup(display_teacher_results if display else teacher_results, source)
            if display and options.teacher_view == "stereo-left" and not options.include_display_teacher and prediction is None:
                continue
            if prediction is not None:
                if prediction.metadata.get("units") != "meters":
                    return True
            elif getattr(config, "model", None) != "depthpro":
                return True
        return False

    # HEIC input size is compressed; reserve a conservative decoding window.
    # Extract once to disk, then process every photo with one resident teacher.
    def source_bytes(source: Path) -> int:
        try:
            return source.stat().st_size
        except OSError:
            # Admission reports the actual per-photo failure. An unreadable or
            # raced-out input must not leak the already-created staging folder.
            return 16 * 1024**2
    largest_source = max(map(source_bytes, inputs), default=1)
    discovery_workers = memory_limited_workers(worker_count, largest_source * 32)
    prepared_sources: list[dict[str, Any]] = []
    source_completions = [0] * len(inputs)

    def completed_sources() -> int:
        return sum(count == len(teachers) for count in source_completions)

    try:
        snapshot("generating", 0)
        emit("dataset_started", dataset_dir=str(temporary), output_dir=str(destination), total_sources=len(inputs))
        from .learned_depth import release_learned_depth_cache
        release_learned_depth_cache()
        emit("sources_started", dataset_dir=str(temporary), total_sources=len(inputs))
        for index, prepared in enumerate(ordered_map(prepare_source, inputs, workers=discovery_workers)):
            source = inputs[index]
            print(f"Dataset photo {index + 1}/{len(inputs)}: {source.name}", file=sys.stderr, flush=True)
            if isinstance(prepared, Exception):
                skip(source, prepared)
                source_completions[index] = len(teachers)
                snapshot("generating", completed_sources())
                continue
            discovery, photo_metadata = prepared
            views = {asset.semantic_name: asset for asset in discovery.assets if asset.kind == "spatial_view"}
            left, right = views["spatial_left"], views["spatial_right"]
            source_id = f"{index:05d}-{discovery.source_sha256[:16]}"
            photo_folder = temporary / source_id
            photo_folder.mkdir()
            # Store immutable source arrays once, shared by all teacher variants.
            # Storage failures are fatal, rather than silently excluding a photo.
            raw = store_raw_assets(discovery, photo_folder)
            left_index = next(i for i, asset in enumerate(discovery.assets) if asset is left)
            right_index = next(i for i, asset in enumerate(discovery.assets) if asset is right)
            blueprint = copy.copy(discovery)
            blueprint.assets = [copy.copy(asset) for asset in discovery.assets]
            for asset in blueprint.assets:
                asset.array = None
            prepared_sources.append({"source": source, "index": index, "discovery": blueprint,
                "photo_metadata": photo_metadata, "raw": raw, "left_index": left_index,
                "right_index": right_index, "source_id": source_id, "photo_folder": photo_folder})
            emit("source_ready", dataset_dir=str(temporary), source_id=source_id,
                 source_path=str(source), processed_sources=index + 1, total_sources=len(inputs))
            del discovery, prepared, views, left, right, blueprint
        if options.metric_anchor is not None:
            emit("teacher_started", dataset_dir=str(temporary), teacher_id="metric_anchor",
                 model=getattr(options.metric_anchor, "model", None), total_sources=len(prepared_sources),
                 teacher_index=0, total_teachers=len(teachers), role="metric_anchor")
            try:
                for entry in prepared_sources:
                    source = entry["source"]
                    discovery = restore_source(entry)
                    for display in ((True,) if options.teacher_view == "display" else (False, True)):
                        if not needs_anchor(source, display=display):
                            continue
                        candidates = [asset for asset in discovery.assets if asset.kind == "display_view"] if display else [discovery.assets[entry["left_index"]]]
                        if not candidates:
                            continue
                        rgb = candidates[0].array
                        key = str(source), display
                        try:
                            prediction = _lookup(display_metric_anchor_results if display else metric_anchor_results, source)
                            if prediction is None:
                                prediction = predict(rgb, anchor=True, focal_pixels=None if display else discovery.spatial_photo["left_camera"]["focal_length_x_pixels"],
                                                     reference_label="display" if display else "spatial_left")
                            target, native = np.asarray(prediction.source_depth), np.asarray(prediction.native_depth)
                            if (prediction.metadata.get("units") != "meters" or target.shape != rgb.shape[:2]
                                    or target.dtype != np.float32 or native.ndim != 2 or native.dtype != np.float32
                                    or prediction.metadata.get("input_rgb_sha256") != sha256_array(rgb)
                                    or not prediction.metadata.get("checkpoint_sha256")):
                                raise DatasetError("Metric anchor must preserve float32 meters, native depth and provenance on the exact selected RGB grid")
                            folder = entry["photo_folder"] / ("display-anchor" if display else "left-anchor")
                            folder.mkdir()
                            cached_anchors[key] = {"target": dataset_array_record(temporary, folder / "target.npy", target),
                                                   "native": dataset_array_record(temporary, folder / "native.npy", native),
                                                   "metadata": _jsonable(prediction.metadata)}
                            if options.retain_intermediates and getattr(prediction, "confidence", None) is not None:
                                cached_anchors[key]["confidence"] = dataset_array_record(temporary, folder / "confidence.npy", prediction.confidence)
                            emit("anchor_ready", dataset_dir=str(temporary), source_id=entry["source_id"],
                                 source_path=str(source), teacher_id="metric_anchor", reference_label="display" if display else "spatial_left")
                        except (OSError, _TeacherModelLoadError):
                            raise
                        except Exception as exc:
                            if not options.skip_bad_photos:
                                raise
                            anchor_errors[key] = str(exc)
                            emit("anchor_failed", dataset_dir=str(temporary), source_path=str(source), reason=str(exc))
                    del discovery
                    candidates = rgb = prediction = target = native = None
            except _TeacherModelLoadError as exc:
                if not options.skip_bad_photos:
                    raise
                emit("teacher_failed", dataset_dir=str(temporary), teacher_id="metric_anchor",
                     reason=str(exc), error_type=type(exc).__name__)
                for pending in prepared_sources:
                    for display in ((True,) if options.teacher_view == "display" else (False, True)):
                        key = str(pending["source"]), display
                        if key not in cached_anchors and needs_anchor(pending["source"], display=display):
                            anchor_errors[key] = str(exc)
            finally:
                release_predictors()
        teacher_order = {teacher_id: index for index, teacher_id in enumerate(teacher_ids)}
        source_order = {entry["source_id"]: entry["index"] for entry in prepared_sources}
        for teacher_index, teacher_id in enumerate(teacher_ids):
            emit("teacher_started", dataset_dir=str(temporary), teacher_id=teacher_id,
                 model=getattr(teachers[teacher_index], "model", None), teacher_index=teacher_index + 1,
                 total_teachers=len(teachers), total_sources=len(prepared_sources), role="teacher")
            try:
                for phase_position, entry in enumerate(prepared_sources):
                    source, index = entry["source"], entry["index"]
                    discovery = restore_source(entry)
                    left_index, right_index = entry["left_index"], entry["right_index"]
                    left, right = discovery.assets[left_index], discovery.assets[right_index]
                    raw, source_id, photo_metadata = entry["raw"], entry["source_id"], entry["photo_metadata"]
                    photo_folder = entry["photo_folder"]
                    print(f"Teacher {teacher_index + 1}/{len(teachers)} ({teacher_id}), photo {index + 1}/{len(inputs)}: {source.name}", file=sys.stderr, flush=True)
                    emit("photo_started", dataset_dir=str(temporary), teacher_id=teacher_id,
                         source_path=str(source), source_id=source_id, photo_index=index + 1, total_sources=len(inputs))
                    folder = photo_folder / f"teacher-{teacher_index + 1}"
                    folder.mkdir()
                    try:
                        sample = make_sample(source, discovery, index, teacher_index, raw, left, right,
                                             left_index, right_index, source_id, photo_metadata, folder)
                    except (OSError, _TeacherModelLoadError):
                        # Output and shared model failures must not masquerade as bad photos.
                        raise
                    except Exception as exc:
                        discard(folder)
                        skip(source, exc, teacher_id=teacher_id)
                    else:
                        samples.append(sample)
                        # Manifest ordering and stable IDs remain source-major even
                        # though inference is teacher-major.
                        samples.sort(key=lambda value: (source_order[value["source_id"]], teacher_order[value["teacher_id"]]))
                        source_completions[index] += 1
                        manifest = snapshot("generating", completed_sources())
                        emit("sample_ready", dataset_dir=str(temporary), sample_id=sample["id"], source_id=source_id,
                             source_photo_id=source_id, teacher_id=teacher_id, summary=manifest["summary"])
                        del discovery, left, right
                        continue
                    source_completions[index] += 1
                    snapshot("generating", completed_sources())
                    del discovery, left, right
            except _TeacherModelLoadError as exc:
                if not options.skip_bad_photos:
                    raise
                emit("teacher_failed", dataset_dir=str(temporary), teacher_id=teacher_id,
                     reason=str(exc), error_type=type(exc).__name__)
                # A model configuration failure applies to this entire phase.
                # Keep completed teachers and avoid retrying the same load for
                # every remaining photo.
                for pending in prepared_sources[phase_position:]:
                    skip(pending["source"], exc, teacher_id=teacher_id)
                    source_completions[pending["index"]] += 1
                snapshot("generating", completed_sources())
            finally:
                release_predictors()
            emit("teacher_complete", dataset_dir=str(temporary), teacher_id=teacher_id,
                 teacher_index=teacher_index + 1, total_teachers=len(teachers))
        if not samples:
            reasons = "; ".join(record["reason"] for record in skipped[:5])
            raise DatasetError(f"No usable calibrated spatial photos/teacher labels remain. {reasons}")
        # Two-pass anchor natives and wholly rejected photos may own shared
        # files. Prune by final references, never by photo folder ownership.
        from .dataset_review import _array_records
        referenced = {record["path"] for record in _array_records(samples)}
        for path in list(array_sizes):
            if path not in referenced:
                (temporary / path).unlink()
                del array_sizes[path]
        manifest = snapshot("complete", len(inputs))
        # Never replace an existing user dataset, including one created while inference ran.
        if destination.exists():
            raise DatasetError(f"Dataset destination appeared during inference: {destination}")
        # Use the same race-safe publisher as reviewed copies, avoiding a
        # second implementation of the platform-specific exclusive rename.
        from .dataset_review import _publish_new_directory
        _publish_new_directory(temporary, destination)
        emit("dataset_complete", dataset_dir=str(destination), output_dir=str(destination), summary=manifest["summary"])
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        release_predictors(report=False)


def _sample_scientific_records(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Require complete records before recursive discovery can omit a plane."""
    from .dataset_review import _ARRAY_FIELDS, _TEACHER_PAYLOAD_FIELDS, _array_records

    identity = sample.get("id", "")
    removed = sample.get("teacher_payload_removed", False)
    if type(removed) is not bool or removed and sample.get("excluded") is not True:
        raise DatasetError(f"Sample {identity} has an invalid removed-teacher payload declaration")

    def require_record(value: Any, field: str) -> None:
        if not isinstance(value, dict) or not _ARRAY_FIELDS.issubset(value):
            raise DatasetError(f"Sample {identity} has an incomplete scientific array record: {field}")

    for key in ("rgb", "right_rgb"):
        require_record(sample.get(key), key)
    raw_assets = sample.get("raw_assets")
    if not isinstance(raw_assets, list):
        raise DatasetError(f"Sample {identity} has malformed raw assets")
    for index, asset in enumerate(raw_assets):
        require_record(asset.get("storage") if isinstance(asset, dict) else None, f"raw_assets[{index}].storage")
    label_fields = {
        "teacher": ("target", "native_target", "valid_mask"),
        "raft_target": ("target", "valid_mask"),
        "display_teacher": ("target", "native_target"),
        "registered_display_teacher": ("target", "valid_mask"),
        "metric_anchor": ("target", "native_target"),
        "display_metric_anchor": ("target", "native_target"),
        "anchored_teacher": ("target", "valid_mask"),
        "anchored_display_teacher": ("target", "valid_mask"),
        "reference": ("target", "valid_mask"),
    }
    for key, required in label_fields.items():
        if removed and key in _TEACHER_PAYLOAD_FIELDS:
            continue
        if key != "teacher" and key not in sample:
            continue
        label = sample.get(key)
        if not isinstance(label, dict):
            raise DatasetError(f"Sample {identity} has malformed {key} depth label")
        for field in required:
            if field == "native_target" and field not in label and label.get("native_prediction_retained") is False:
                continue
            if (field == "valid_mask" and field not in label and key in {"teacher", "reference"}
                    and label.get("validity_policy") == "positive_finite"):
                continue
            require_record(label.get(field), f"{key}.{field}")
        for field in ("native_target", "valid_mask", "confidence"):
            if field in label:
                require_record(label[field], f"{key}.{field}")
    if "display_rgb" in sample:
        require_record(sample["display_rgb"], "display_rgb")

    def check_partial(value: Any, field: str) -> None:
        if isinstance(value, dict):
            present = _ARRAY_FIELDS.intersection(value)
            path = value.get("path")
            if len(present) >= 3 or (isinstance(path, str) and Path(path).suffix.lower() in {".npy", ".npz", ".exr", ".png"}):
                require_record(value, field)
            else:
                for key, child in value.items():
                    check_partial(child, f"{field}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_partial(child, f"{field}[{index}]")

    check_partial({key: value for key, value in sample.items() if not removed or key not in _TEACHER_PAYLOAD_FIELDS}, "sample")
    return list(_array_records(sample))


def load_dataset(directory: Path | str, *, verify: bool = True, workers: int | None = None,
                 metadata_only: bool = False, include_excluded: bool = True,
                 snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a snapshot and optionally verify the arrays that will be consumed.

    Metadata-only reads never open scientific arrays. Excluded records remain
    in the returned snapshot for restoration; consumers may skip their files.
    """
    from .dataset_review import _array_path, _read_manifest, _validate_manifest_metadata
    if snapshot is None:
        root, manifest = _read_manifest(directory, validate_files=False)
    else:
        root = Path(directory).expanduser().resolve()
        manifest = copy.deepcopy(dict(snapshot))
        _validate_manifest_metadata(root, manifest, validate_files=False)
    try:
        worker_count = resolve_workers(workers)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    records_by_sample = [(sample, _sample_scientific_records(sample)) for sample in manifest["samples"]]
    if metadata_only:
        return manifest
    # A raw asset, teacher/native plane and several teacher variants can all
    # reference one immutable file. Read it once, but reject conflicting
    # metadata instead of allowing deduplication to hide an invalid record.
    unique: dict[Path, dict[str, Any]] = {}
    for sample, records in records_by_sample:
        if not include_excluded and sample.get("excluded", False):
            continue
        for record in records:
            path = _array_path(root, record)
            if path in unique and any(unique[path][key] != record[key]
                                      for key in ("shape", "dtype", "file_sha256", "array_sha256")):
                raise DatasetError(f"Conflicting dataset array records: {record['path']}")
            unique[path] = record

    def check(item: tuple[Path, dict[str, Any]]) -> None:
        path, record = item
        try:
            array = read_array(path, mmap_mode="r")
            if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                raise DatasetError(f"Dataset array shape/dtype mismatch: {record['path']}")
            if verify and (sha256_file(path) != record["file_sha256"] or sha256_array(array) != record["array_sha256"]):
                raise DatasetError(f"Dataset checksum mismatch: {record['path']}")
        except (ValueError, OSError) as exc:
            raise DatasetError(f"Cannot read dataset array {record['path']}: {exc}") from exc

    sizes = [math.prod(record["shape"]) * np.dtype(record["dtype"]).itemsize for record in unique.values()]
    count = memory_limited_workers(worker_count, max(sizes, default=1) * 2)
    for _ in ordered_map(check, unique.items(), workers=count):
        pass
    return manifest
