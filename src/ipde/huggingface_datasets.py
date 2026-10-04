"""Optional Hugging Face import of explicitly mapped precision-safe stereo data.

Generic image/text datasets cannot train calibrated stereo. This adapter accepts
NPY/NPZ planes (as bytes or sandboxed local paths), declared meter-depth labels,
and matching calibration. It never decodes JPEGs/Pillow images, guesses a depth
scale, resizes, normalizes, downloads row URLs, or executes dataset scripts.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .array_storage import array_record, read_array
from .dataset import DatasetError, assign_grouped_splits, load_dataset, teacher_depth_to_flow
from .dataset_review import _publish_new_directory
from .formats import sha256_array
from .concurrency import memory_limited_workers, ordered_map, resolve_workers


_REQUIRED_COLUMNS = {"left", "right", "depth", "calibration"}
_COLUMN_KEYS = _REQUIRED_COLUMNS | {"id", "source_sha256", "scene", "category", "metadata", "burst_ids"}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _validate_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise DatasetError("Hugging Face mapping must be a JSON object")
    columns = mapping.get("columns")
    if not isinstance(columns, Mapping) or not _REQUIRED_COLUMNS.issubset(columns):
        raise DatasetError("Explicit columns left, right, depth and calibration are required; generic HF datasets are not calibrated stereo datasets")
    if set(columns) - _COLUMN_KEYS or any(not isinstance(value, str) or not value for value in columns.values()):
        raise DatasetError("Unsupported or empty Hugging Face column mapping")
    if mapping.get("depth_units") != "meters" or mapping.get("coordinate_reference") != "spatial_left":
        raise DatasetError("Import requires declared meters on the exact spatial_left stereo grid")
    role = mapping.get("depth_role")
    if role not in {"measured", "pseudo_label"}:
        raise DatasetError("Declare depth_role as measured or pseudo_label; accuracy is not inferred")
    if not isinstance(mapping.get("verified_scene_groups", False), bool):
        raise DatasetError("verified_scene_groups must be a boolean")
    if role == "pseudo_label":
        teacher = mapping.get("teacher_metadata")
        if not isinstance(teacher, Mapping) or not _SHA256.fullmatch(str(teacher.get("checkpoint_sha256", ""))):
            raise DatasetError("Imported pseudo labels must identify their actual teacher checkpoint_sha256")
    try:
        return json.loads(json.dumps(dict(mapping), allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise DatasetError("Hugging Face mapping contains unsupported metadata values") from exc


def _mapping_value(row: Mapping[str, Any], columns: Mapping[str, str], key: str, default: Any = None) -> Any:
    column = columns.get(key)
    if column is None:
        return default
    if column not in row:
        raise DatasetError(f"Mapped {key} column is missing: {column}")
    return row[column]


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise DatasetError(f"{label} must be an object or JSON object text") from exc
    if not isinstance(value, Mapping):
        raise DatasetError(f"{label} must be an object or JSON object text")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{label} contains unsupported metadata values") from exc


def _plane(value: Any, asset_dir: Path | None) -> np.ndarray:
    # Arrow/Python lists have no trustworthy original ndarray dtype. Require
    # an explicitly encoded NPY payload rather than silently guessing one.
    if isinstance(value, np.ndarray):
        result = value
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if not (raw.startswith(b"\x93NUMPY") or raw.startswith(b"PK\x03\x04")):
            raise DatasetError("Imported planes must be lossless NPY/NPZ bytes, not display images")
        try:
            decoded = np.load(io.BytesIO(raw), allow_pickle=False)
            if isinstance(decoded, np.lib.npyio.NpzFile):
                with decoded:
                    if decoded.files != ["data"]:
                        raise DatasetError("Imported NPZ must contain exactly one plane named data")
                    result = decoded["data"]
            else:
                result = decoded
        except (OSError, ValueError, EOFError) as exc:
            raise DatasetError(f"Cannot decode precision-preserved NPY/NPZ plane: {exc}") from exc
    elif isinstance(value, (str, Path)):
        if asset_dir is None:
            raise DatasetError("Local array paths require an explicit asset_dir; remote row URLs are not downloaded")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or str(value).startswith(("http:", "https:", "hf:")):
            raise DatasetError("Array paths must be relative paths inside asset_dir")
        path = (asset_dir / relative).resolve()
        if not path.is_relative_to(asset_dir) or not path.is_file():
            raise DatasetError("Array path escapes asset_dir or is missing")
        try:
            result = read_array(path)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot read imported array: {exc}") from exc
    elif isinstance(value, Mapping) and value.get("bytes") is not None:
        result = _plane(value["bytes"], asset_dir)
    else:
        raise DatasetError("Plane must be NPY/NPZ bytes or a local array path; generic images and inferred-dtype lists are unsupported")
    if result.dtype.hasobject or not result.size:
        raise DatasetError("Imported planes must be nonempty numeric arrays without pickle/object data")
    return result


def import_dataset_rows(
    rows: Iterable[Mapping[str, Any]],
    output_dir: Path | str,
    mapping: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
    asset_dir: Path | str | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    workers: int | None = None,
) -> dict[str, Any]:
    """Import compatible rows; skip malformed rows and report every rejection.

    A measured declaration is a user assertion, not a measurement certificate.
    Non-Apple external datasets remain explicitly identified as external data.
    """
    specification = _validate_mapping(mapping)
    try:
        worker_count = resolve_workers(workers)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    assign_grouped_splits([], validation_fraction, split_seed)
    columns = specification["columns"]
    origin = _json_object(provenance or {}, "Import provenance")
    assets = Path(asset_dir).expanduser().resolve() if asset_dir is not None else None
    if assets is not None and not assets.is_dir():
        raise DatasetError("asset_dir must be an existing directory")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if assets is not None and destination.is_relative_to(assets):
        raise DatasetError("Imported dataset destination must be outside asset_dir")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    written: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}

    def store_values(values: list[np.ndarray]) -> list[dict[str, Any]]:
        # Row iteration and deduplication remain on the caller. Each worker
        # publishes one distinct immutable array file, never shared metadata.
        keys = [(value.dtype.str, tuple(value.shape), sha256_array(value)) for value in values]
        planned = {key: value for key, value in zip(keys, values) if key not in written}

        def save(item: Any) -> Any:
            key, value = item
            identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            return key, array_record(temporary, temporary / "arrays" / f"{identity}.npz", value, compressed=True)

        count = memory_limited_workers(worker_count, max((value.nbytes for value in planned.values()), default=1) * 3)
        for key, record in ordered_map(save, planned.items(), workers=count):
            written[key] = record
        return [dict(written[key]) for key in keys]

    try:
        for row_index, row in enumerate(rows):
            try:
                if not isinstance(row, Mapping):
                    raise DatasetError("Dataset row must be an object")
                left = _plane(_mapping_value(row, columns, "left"), assets)
                right = _plane(_mapping_value(row, columns, "right"), assets)
                depth = _plane(_mapping_value(row, columns, "depth"), assets)
                if left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
                    raise DatasetError("RAFT import requires matching native uint8 HxWx3 RGB stereo views; higher-bit data is never quantized automatically")
                if depth.shape != left.shape[:2] or depth.dtype.kind != "f":
                    raise DatasetError("Meter depth must be a floating-point HxW plane on the exact left grid; no resizing")
                calibration = _json_object(_mapping_value(row, columns, "calibration"), "Stereo calibration")
                if calibration.get("raft_stereo_ready") is not True:
                    raise DatasetError("Import requires explicitly validated rectified stereo calibration")
                if any(tuple((calibration.get(camera) or {}).get(key) for key in ("height", "width")) != left.shape[:2]
                       for camera in ("left_camera", "right_camera")):
                    raise DatasetError("Left and right calibration dimensions must match both native stereo views")
                flow, flow_valid, geometry = teacher_depth_to_flow(depth, calibration)
                if specification["depth_role"] == "measured":
                    geometry.update({"label_kind": "measured_reference_derived_flow",
                                     "precision_note": "Calibrated depth-to-flow representation change assumes the user-declared measurement and left-camera registration are accurate"})
                if not flow_valid.any():
                    raise DatasetError("No positive finite in-image calibrated depth correspondences remain")
                source_hash = _mapping_value(row, columns, "source_sha256")
                if source_hash is None:
                    source_hash = hashlib.sha256((sha256_array(left) + ":" + sha256_array(right)).encode()).hexdigest()
                    identity_kind = "derived_stereo_pair_array_hashes"
                else:
                    if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
                        raise DatasetError("source_sha256 must be a SHA-256 hex string")
                    source_hash = source_hash.lower()
                    identity_kind = "dataset_declared_original_source_hash"
                external_id = _mapping_value(row, columns, "id", str(row_index))
                if not isinstance(external_id, (str, int)) or isinstance(external_id, bool):
                    raise DatasetError("External row ID must be a string or integer")
                metadata = _json_object(_mapping_value(row, columns, "metadata", {}), "Row metadata")
                scene = _mapping_value(row, columns, "scene")
                category = _mapping_value(row, columns, "category")
                if any(item is not None and (not isinstance(item, str) or not item.strip()) for item in (scene, category)):
                    raise DatasetError("Scene/category labels must be nonempty strings when present")
                bursts = _mapping_value(row, columns, "burst_ids", [])
                if not isinstance(bursts, list) or any(not isinstance(item, str) or not item for item in bursts):
                    raise DatasetError("burst_ids must be a list of nonempty strings")
                left_record, right_record, depth_record, valid_record, flow_record, flow_valid_record = store_values(
                    [left, right, depth, np.isfinite(depth) & (depth > 0), flow, flow_valid])
                label_metadata = dict(specification.get("teacher_metadata", {}))
                label_metadata.update({"label_data_sha256": sha256_array(depth), "input_rgb_sha256": sha256_array(left),
                                       "units": "meters", "reference_label": "spatial_left", "import_provenance": origin,
                                       "provenance_kind": specification["depth_role"]})
                teacher = {"label_kind": "imported_pseudo_label" if specification["depth_role"] == "pseudo_label" else "user_declared_measured_reference",
                           "units": "meters", "metadata": label_metadata, "target": depth_record,
                           "native_target": depth_record, "valid_mask": valid_record}
                sample = {
                    "id": hashlib.sha256(f"{row_index}:{source_hash}:{external_id}".encode()).hexdigest(),
                    "source_path": f"external-dataset:{origin.get('source', 'mapped-rows')}:{external_id}",
                    "source_sha256": source_hash, "source_identity_kind": identity_kind,
                    "rgb": {**left_record, "source_bit_depth": 8}, "right_rgb": {**right_record, "source_bit_depth": 8},
                    "raw_assets": [{"kind": "spatial_left", "storage": left_record}, {"kind": "spatial_right", "storage": right_record},
                                   {"kind": "imported_depth", "storage": depth_record}],
                    "coordinate_reference": "spatial_left: exact imported array coordinates",
                    "calibration": calibration, "requested_group": scene, "category_label": category, "burst_ids": bursts,
                    "teacher": teacher, "training_target_choice": "teacher",
                    "raft_target": {"source_label": "reference" if specification["depth_role"] == "measured" else "teacher",
                                    "target": flow_record, "valid_mask": flow_valid_record, "metadata": geometry},
                    "external_import": {"row_index": row_index, "external_id": str(external_id), "metadata": metadata, "provenance": origin},
                }
                if specification["depth_role"] == "measured":
                    sample["reference"] = {**teacher, "label_kind": "user_supplied_measured_reference",
                                           "accuracy_note": "Imported measurement/registration accuracy is declared by the user, not certified by IPDE"}
                    sample["training_target_choice"] = "reference"
                samples.append(sample)
            except OSError:
                # Disk/permission failures are not malformed dataset rows.
                raise
            except (DatasetError, ValueError, TypeError, KeyError) as exc:
                skipped.append({"row_index": row_index, "reason": str(exc)})
        if not samples:
            reason = skipped[0]["reason"] if skipped else "the selected split was empty"
            raise DatasetError(f"No compatible calibrated stereo rows were imported: {reason}")
        groups = assign_grouped_splits(samples, validation_fraction, split_seed)
        warnings = ["External stereo calibration and measured-label accuracy require independent verification; import does not certify Apple camera authenticity"]
        if len(groups) < 2:
            warnings.append("Only one independent photo group; add another dataset before training")
        if skipped:
            warnings.append(f"Skipped {len(skipped)} incompatible or malformed rows; review the import report")
        explicit_scenes = specification.get("verified_scene_groups", False) and all(sample["requested_group"] for sample in samples)
        report = {"provenance": origin, "mapping": specification, "skipped_rows": skipped,
                  "imported_samples": len(samples), "unique_arrays": len(written), "remote_code_execution_allowed": False,
                  "file_workers": worker_count}
        summary = {"samples": len(samples), "groups": len(groups), "train_samples": sum(s["split"] == "train" for s in samples),
                   "validation_samples": sum(s["split"] == "validation" for s in samples)}
        manifest = {"schema": "ipde-depth-dataset-v1", "precision_policy": "Imported NPY values/dtypes preserved bit-for-bit in lossless NPZ; no normalization, gamma or resampling",
                    "split_seed": split_seed, "validation_fraction": validation_fraction, "group_ids": groups,
                    "grouping_semantics": "scene" if explicit_scenes else "capture", "explicit_scene_groups": bool(explicit_scenes),
                    "samples": samples, "summary": summary, "warnings": warnings, "external_import": report}
        (temporary / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        load_dataset(temporary, verify=True, workers=worker_count)
        _publish_new_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"dataset_path": str(destination), "summary": summary, "warnings": warnings, "external_import": report}


def import_huggingface_dataset(
    source: str,
    output_dir: Path | str,
    mapping: Mapping[str, Any],
    *,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
    data_files: str | Mapping[str, Any] | None = None,
    asset_dir: Path | str | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    workers: int | None = None,
) -> dict[str, Any]:
    """Load a local data builder or Hub dataset using the optional datasets lib.

    Callers must provide the compatible column mapping. Hub planes must arrive
    as encoded NPY/NPZ bytes; path columns require an explicit local asset_dir.
    Loading scripts and automatic installs are unsupported.
    """
    _validate_mapping(mapping)
    if not isinstance(source, str) or not source.strip() or source.lower().endswith(".py"):
        raise DatasetError("Select a data-only Hugging Face dataset or json/parquet/csv builder; loading scripts are unsupported")
    try:
        import datasets
    except ImportError as exc:
        raise DatasetError("Optional Hugging Face import requires the datasets Python library in the configured runtime; no packages were installed") from exc
    kwargs: dict[str, Any] = {"split": split, "trust_remote_code": False, "streaming": False}
    if config is not None:
        kwargs["name"] = config
    if revision is not None:
        kwargs["revision"] = revision
    if data_files is not None:
        kwargs["data_files"] = data_files
    try:
        rows = datasets.load_dataset(source, **kwargs)
    except Exception as exc:
        raise DatasetError(f"Cannot load data-only Hugging Face dataset: {exc}") from exc
    fingerprint = getattr(rows, "_fingerprint", None)
    provenance = {"backend": "huggingface-datasets", "source": source, "config": config, "split": split,
                  "requested_revision": revision, "fingerprint": fingerprint, "datasets_version": getattr(datasets, "__version__", None),
                  "data_files": data_files, "trust_remote_code": False}
    return import_dataset_rows(rows, output_dir, mapping, provenance=provenance, asset_dir=asset_dir,
                               validation_fraction=validation_fraction, split_seed=split_seed, workers=workers)
