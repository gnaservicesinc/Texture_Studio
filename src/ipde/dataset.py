"""Lossless, provenance-tracked spatial-photo datasets for depth experiments.

Teacher predictions are pseudo-labels. Saving a float32 estimate does not make
it a measured reference or restore any detail absent from the model output.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .extractor import _asset_record, _jsonable, discover_file
from .formats import sha256_array, sha256_file, verify_npy, write_npy

if TYPE_CHECKING:
    from .learned_depth import LearnedDepthConfig, LearnedDepthResult


class DatasetError(RuntimeError):
    """A dataset could not be built without losing alignment or provenance."""


@dataclass(frozen=True)
class DatasetOptions:
    teacher: LearnedDepthConfig
    group_ids: Mapping[str, str] | None = None
    validation_fraction: float = 0.2
    split_seed: int = 0
    reference_paths: Mapping[str, Path] | None = None
    include_display_teacher: bool = True
    prefer_registered_display_teacher: bool = True
    grouping_semantics: str = "capture"
    metric_anchor: LearnedDepthConfig | None = None


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
        hashes = sorted(samples[i]["source_sha256"] for i in indices)
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
    sources: Sequence[Path | str],
    output_dir: Path | str,
    options: DatasetOptions,
    *,
    teacher_results: Mapping[str, LearnedDepthResult] | None = None,
    display_teacher_results: Mapping[str, LearnedDepthResult] | None = None,
    metric_anchor_results: Mapping[str, LearnedDepthResult] | None = None,
    display_metric_anchor_results: Mapping[str, LearnedDepthResult] | None = None,
) -> dict[str, Any]:
    """Build a new dataset atomically from explicitly selected spatial HEICs.

    All discovered auxiliary/sample arrays retain their original dtype and
    bytes. The left and right views retain their original grids/calibration.
    Only the teacher's documented inference preprocessing modifies model input.
    Optional measured references must be positive depth in meters, stored as an
    HxW float NPY on the exact left-view grid. They are never implicitly resized.
    """
    destination = Path(output_dir).expanduser().resolve()
    inputs = [Path(source).expanduser().resolve() for source in sources]
    if not inputs:
        raise DatasetError("Select at least one original spatial HEIC photo")
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if any(source.suffix.lower() not in {".heic", ".heif", ".hif"} for source in inputs):
        raise DatasetError("Dataset inputs must be original HEIC/HEIF files, not depth previews")
    if len(set(inputs)) != len(inputs):
        raise DatasetError("A source path was selected more than once")
    if options.grouping_semantics not in {"capture", "scene"}:
        raise DatasetError("grouping_semantics must be capture or scene")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    predictors: dict[str, Any] = {}

    def predict(array: np.ndarray, *, anchor: bool = False, **kwargs: Any) -> LearnedDepthResult:
        name = "metric_anchor" if anchor else "teacher"
        if name not in predictors:
            from .learned_depth import LearnedDepthPredictor
            predictors[name] = LearnedDepthPredictor(options.metric_anchor if anchor else options.teacher)
        return predictors[name](array, **kwargs)

    def anchored_label(
        source: Path, rgb: np.ndarray, prediction: LearnedDepthResult, folder: Path,
        *, display: bool, focal_pixels: float | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        if prediction.metadata["units"] == "meters" or options.metric_anchor is None:
            return None, None, None
        print(f"  Applying explicit metric-model anchor on the {'DISPLAY' if display else 'LEFT'} grid", file=sys.stderr, flush=True)
        anchor_prediction = _lookup(display_metric_anchor_results if display else metric_anchor_results, source)
        if anchor_prediction is None:
            anchor_prediction = predict(rgb, anchor=True, focal_pixels=focal_pixels, reference_label="display" if display else "spatial_left")
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
            "target": _array_record(temporary, folder / f"{prefix}metric-anchor.npy", anchor_depth),
            "native_target": _array_record(temporary, folder / f"{prefix}metric-anchor-native.npy", anchor_native),
        }
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
            "target": _array_record(temporary, folder / f"{prefix}anchored-teacher.npy", anchored),
            "valid_mask": _array_record(temporary, folder / f"{prefix}anchored-valid.npy", accepted),
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
    try:
        for index, source in enumerate(inputs):
            print(f"Dataset photo {index + 1}/{len(inputs)}: {source.name}", file=sys.stderr, flush=True)
            discovery = discover_file(source)
            spatial = discovery.spatial_photo
            if spatial is None:
                raise DatasetError(f"{source.name} has no calibrated Apple spatial stereo pair")
            views = {
                asset.semantic_name: asset
                for asset in discovery.assets
                if asset.kind == "spatial_view"
            }
            if "spatial_left" not in views or "spatial_right" not in views:
                raise DatasetError(f"{source.name} lacks decoded left/right spatial views")
            left, right = views["spatial_left"], views["spatial_right"]
            if left.array.shape[:2] != right.array.shape[:2]:
                raise DatasetError("Left/right views must have exactly matching grids")
            sample_id = f"{index:05d}-{discovery.source_sha256[:16]}"
            folder = temporary / sample_id
            folder.mkdir()
            raw: list[dict[str, Any]] = []
            for ordinal, asset in enumerate(discovery.assets):
                record = _asset_record(asset)
                record["storage"] = _array_record(temporary, folder / f"raw-{ordinal:03d}.npy", asset.array)
                record["metadata_blocks"] = _jsonable(asset.metadata_blocks)
                raw.append(record)
            left_index = next(i for i, asset in enumerate(discovery.assets) if asset is left)
            right_index = next(i for i, asset in enumerate(discovery.assets) if asset is right)
            prediction = _lookup(teacher_results, source)
            if prediction is None:
                print("  Running selected teacher on the calibrated LEFT view", file=sys.stderr, flush=True)
                prediction = predict(
                    left.array,
                    focal_pixels=spatial["left_camera"]["focal_length_x_pixels"],
                    reference_label="spatial_left",
                )
            target = np.asarray(prediction.source_depth)
            native = np.asarray(prediction.native_depth)
            if target.shape != left.array.shape[:2] or target.dtype != np.float32:
                raise DatasetError("Teacher output must be float32 on the exact left-view HxW grid")
            if native.ndim != 2 or native.dtype != np.float32:
                raise DatasetError("Teacher native output must be a float32 HxW plane")
            metadata = dict(prediction.metadata)
            if metadata.get("input_rgb_sha256") != sha256_array(left.array):
                raise DatasetError("Teacher RGB provenance does not match the extracted left view")
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
                "source_path": str(source),
                "source_sha256": discovery.source_sha256,
                "source_bytes": discovery.source_size,
                "requested_group": normalized_groups.get(str(source)),
                "burst_ids": sorted(_burst_ids(discovery.top_level_images)),
                "rgb": {**raw[left_index]["storage"], "source_bit_depth": left.source_bit_depth},
                "right_rgb": {**raw[right_index]["storage"], "source_bit_depth": right.source_bit_depth},
                "coordinate_reference": "spatial_left: exact decoded sample coordinates, no EXIF rotation",
                "calibration": _jsonable(spatial),
                "top_level_images": _jsonable(discovery.top_level_images),
                "raw_assets": raw,
                "teacher": {
                    "label_kind": "pseudo_label",
                    "units": units,
                    "metadata": _jsonable(metadata),
                    "target": _array_record(temporary, folder / "teacher.npy", target),
                    "native_target": _array_record(temporary, folder / "teacher-native.npy", native),
                    "valid_mask": _array_record(temporary, folder / "teacher-valid.npy", valid),
                    "valid_pixel_count": int(valid.sum()),
                },
            }
            anchor_record, anchored_record, pseudo_calibration = anchored_label(
                source, left.array, prediction, folder, display=False,
                focal_pixels=spatial["left_camera"]["focal_length_x_pixels"],
            )
            if anchor_record is not None:
                sample["metric_anchor"] = anchor_record
                sample["pseudo_calibration"] = pseudo_calibration
            if anchored_record is not None:
                sample["anchored_teacher"] = anchored_record
                sample["training_target_choice"] = "anchored_teacher"
            training_target = target if units == "meters" else np.load(temporary / anchored_record["target"]["path"], allow_pickle=False) if anchored_record else None
            if training_target is not None and spatial.get("raft_stereo_ready"):
                flow, flow_valid, flow_details = teacher_depth_to_flow(training_target, spatial)
                sample["raft_target"] = {
                    "source_label": "anchored_teacher" if anchored_record else "teacher",
                    "target": _array_record(temporary, folder / "raft-teacher-flow.npy", flow),
                    "valid_mask": _array_record(temporary, folder / "raft-teacher-valid.npy", flow_valid),
                    "metadata": flow_details,
                }
            display_indices = [i for i, asset in enumerate(discovery.assets) if asset.kind == "display_view"]
            if display_indices:
                display_index = display_indices[0]
                display = discovery.assets[display_index]
                sample["display_rgb"] = {**raw[display_index]["storage"], "source_bit_depth": display.source_bit_depth}
                sample["display_note"] = "Separate display camera/framing; never resized onto the left/right stereo grid"
                display_prediction = _lookup(display_teacher_results, source)
                if display_prediction is None and options.include_display_teacher:
                    print("  Running selected teacher on the full display image", file=sys.stderr, flush=True)
                    display_prediction = predict(display.array, reference_label="display")
                if display_prediction is not None:
                    display_target = np.asarray(display_prediction.source_depth)
                    if display_target.shape != display.array.shape[:2] or display_target.dtype != np.float32:
                        raise DatasetError("Display teacher must use the exact display grid")
                    if display_prediction.metadata.get("input_rgb_sha256") != sha256_array(display.array):
                        raise DatasetError("Display teacher provenance does not match the display image")
                    display_metadata = display_prediction.metadata
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
                        "metadata": _jsonable(display_prediction.metadata),
                        "target": _array_record(temporary, folder / "display-teacher.npy", display_target),
                        "native_target": _array_record(temporary, folder / "display-teacher-native.npy", display_prediction.native_depth),
                        "coordinate_reference": "display; separate from RAFT's spatial_left reference",
                    }
                    display_anchor, anchored_display, display_calibration = anchored_label(
                        source, display.array, display_prediction, folder, display=True, focal_pixels=None,
                    )
                    if display_anchor is not None:
                        sample["display_metric_anchor"] = display_anchor
                        sample["display_pseudo_calibration"] = display_calibration
                    if anchored_display is not None:
                        sample["anchored_display_teacher"] = anchored_display
                    display_training_target = display_target if display_metadata["units"] == "meters" else np.load(temporary / anchored_display["target"]["path"], allow_pickle=False) if anchored_display else None
                    from .registration import estimate_display_registration, register_display_depth
                    registration = estimate_display_registration(discovery)
                    print(f"  Display registration: {registration.get('reason')}", file=sys.stderr, flush=True)
                    sample["display_registration"] = _jsonable(registration)
                    if registration.get("accepted") and display_training_target is not None:
                        registered, registered_valid = register_display_depth(display_training_target, discovery, registration)
                        sample["registered_display_teacher"] = {
                            "label_kind": "registered_display_teacher_pseudo_label",
                            "units": "meters",
                            "target": _array_record(temporary, folder / "registered-display-teacher.npy", registered),
                            "valid_mask": _array_record(temporary, folder / "registered-display-valid.npy", registered_valid),
                            "reference_role": registration["reference_role"],
                            "metadata": anchored_display["metadata"] if anchored_display else _jsonable(display_prediction.metadata),
                            "registration": _jsonable(registration),
                            "precision_note": "Approximate same-camera depth transport only inside independently validated registration cells",
                        }
                    use_display = (options.prefer_registered_display_teacher and registration.get("accepted")
                                   and registration.get("reference_role") == "left" and display_training_target is not None)
                    sample["training_target_choice"] = "registered_display_teacher" if use_display else "anchored_teacher" if anchored_record else "teacher"
                    sample["training_target_choice_note"] = (
                        "Use registered display teacher only where its empirical LEFT-camera registration is supported; holes stay excluded"
                        if use_display else "Use separate LEFT-grid supervision; display registration/metric anchoring is unavailable, rejected, or belongs to the RIGHT camera")
            if getattr(prediction, "confidence", None) is not None:
                confidence = np.asarray(prediction.confidence)
                sample["teacher"]["confidence"] = _array_record(temporary, folder / "teacher-confidence.npy", confidence)
                sample["teacher"]["confidence_note"] = "Model confidence is not a measured error bound"
            reference_path = _lookup(references, source)
            if reference_path is not None:
                try:
                    reference = np.load(reference_path, allow_pickle=False)
                except (ValueError, OSError) as exc:
                    raise DatasetError(f"Cannot read measured reference NPY: {reference_path}: {exc}") from exc
                if reference.shape != target.shape or reference.dtype.kind != "f":
                    raise DatasetError("Measured reference must be a floating-point meter-depth NPY on the exact left grid")
                reference_valid = np.isfinite(reference) & (reference > 0)
                if not reference_valid.any():
                    raise DatasetError("Measured reference contains no finite positive meter-depth samples")
                sample["reference"] = {
                    "label_kind": "user_supplied_measured_reference",
                    "units": "meters",
                    "source_path": str(reference_path),
                    "source_sha256": sha256_file(reference_path),
                    "target": _array_record(temporary, folder / "reference.npy", reference),
                    "valid_mask": _array_record(temporary, folder / "reference-valid.npy", reference_valid),
                    "accuracy_note": "User must verify measurement accuracy and left-camera registration independently",
                }
            samples.append(sample)
            print(f"  Preserved {len(raw)} raw arrays and teacher targets", file=sys.stderr, flush=True)
        groups = assign_grouped_splits(samples, options.validation_fraction, options.split_seed)
        if len(groups) < 2:
            warnings.append("Only one independent group is present; training requires another held-out scene/group")
        manifest = {
            "schema": "ipde-depth-dataset-v1",
            "precision_policy": "Raw samples/auxiliaries preserved bit-for-bit in NPY; no normalization, gamma, or resampling",
            "split_seed": options.split_seed,
            "validation_fraction": options.validation_fraction,
            "group_ids": groups,
            "explicit_scene_groups": options.grouping_semantics == "scene" and all(sample["requested_group"] for sample in samples),
            "grouping_semantics": options.grouping_semantics,
            "warnings": warnings,
            "samples": samples,
            "summary": {
                "samples": len(samples),
                "groups": len(groups),
                "train_samples": sum(sample["split"] == "train" for sample in samples),
                "validation_samples": sum(sample["split"] == "validation" for sample in samples),
            },
        }
        (temporary / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Never replace an existing user dataset, including one created while inference ran.
        if destination.exists():
            raise DatasetError(f"Dataset destination appeared during inference: {destination}")
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_dataset(directory: Path | str, *, verify: bool = True) -> dict[str, Any]:
    """Load and verify a dataset, excluding pickle arrays and escaping paths."""
    root = Path(directory).expanduser().resolve()
    try:
        manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise DatasetError(f"Cannot read dataset manifest: {exc}") from exc
    if manifest.get("schema") != "ipde-depth-dataset-v1" or not manifest.get("samples"):
        raise DatasetError("Not a supported, nonempty IPDE depth dataset")
    for sample in manifest["samples"]:
        records = [sample["rgb"], sample["right_rgb"]]
        records.extend(asset["storage"] for asset in sample["raw_assets"])
        records.extend(sample["teacher"][key] for key in ("target", "native_target", "valid_mask"))
        if "confidence" in sample["teacher"]:
            records.append(sample["teacher"]["confidence"])
        if "raft_target" in sample:
            records.extend(sample["raft_target"][key] for key in ("target", "valid_mask"))
        if "display_rgb" in sample:
            records.append(sample["display_rgb"])
        if "display_teacher" in sample:
            records.extend(sample["display_teacher"][key] for key in ("target", "native_target"))
        if "registered_display_teacher" in sample:
            records.extend(sample["registered_display_teacher"][key] for key in ("target", "valid_mask"))
        for key in ("metric_anchor", "display_metric_anchor"):
            if key in sample:
                records.extend(sample[key][name] for name in ("target", "native_target"))
        for key in ("anchored_teacher", "anchored_display_teacher"):
            if key in sample:
                records.extend(sample[key][name] for name in ("target", "valid_mask"))
        if "reference" in sample:
            records.extend(sample["reference"][key] for key in ("target", "valid_mask"))
        for record in records:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root):
                raise DatasetError("Dataset array path escapes the dataset directory")
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                    raise DatasetError(f"Dataset array shape/dtype mismatch: {record['path']}")
                if verify and (sha256_file(path) != record["file_sha256"] or sha256_array(array) != record["array_sha256"]):
                    raise DatasetError(f"Dataset checksum mismatch: {record['path']}")
            except (ValueError, OSError) as exc:
                raise DatasetError(f"Cannot read dataset array {record['path']}: {exc}") from exc
    return manifest
