"""Add or enable display teachers for selected photos in an existing dataset."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from .array_storage import array_record
from .concurrency import resolve_workers
from .dataset import DatasetError, load_dataset
from .dataset_collection import _require_complete
from .dataset_review import (_array_records, _publish_new_directory, _read_manifest_snapshot,
                             _summary, _verified_array)
from .extractor import _jsonable
from .formats import sha256_array, sha256_file
from .learned_depth import LearnedDepthConfig
from .resource_lock import resource_lock


def _photo_key(sample: Mapping[str, Any]) -> tuple[str, str]:
    display = sample.get("display_rgb")
    if not isinstance(display, Mapping) or not display.get("array_sha256") or not sample.get("source_sha256"):
        raise DatasetError("Selected photos require preserved full display RGB and source-photo provenance")
    return sample["source_sha256"], display["array_sha256"]


def _model(sample: Mapping[str, Any]) -> str | None:
    name = sample.get("teacher_model")
    supported = {"depthpro", "depth-anything-v2", "depth-anything-3"}
    if isinstance(name, str) and name in supported:
        return name
    metadata = (sample.get("display_teacher") or sample.get("teacher") or {}).get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    if isinstance(metadata.get("model"), str) and metadata["model"] in supported:
        return metadata["model"]
    return {"apple/DepthPro": "depthpro", "depth-anything/Depth-Anything-V2-Large": "depth-anything-v2",
            "depth-anything/DA3-GIANT-1.1": "depth-anything-3"}.get(metadata.get("model_id"))


def _display_label(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    label = sample.get("display_teacher") or sample.get("teacher")
    if not isinstance(label, Mapping) or not str(label.get("coordinate_reference", "")).startswith("display"):
        return None
    metadata = label.get("metadata", {})
    if (not isinstance(metadata, Mapping) or metadata.get("reference_label") != "display"
            or metadata.get("input_rgb_sha256") != (sample.get("display_rgb") or {}).get("array_sha256")):
        return None
    return label


def generate_teacher(
    directory: Path | str,
    sample_ids: Sequence[str],
    teacher: LearnedDepthConfig,
    *,
    metric_anchor: LearnedDepthConfig | None = None,
    expected_manifest_sha256: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Generate only missing model variants, or enable an existing stored result.

    Selected sample IDs identify photos, so selecting two teacher entries for
    one photo still performs one inference. Checksummed stored display images
    supply the input; original HEIC files may be offline. Every existing array,
    sample ID, group and split stays intact. Same-photo DepthPro display depth
    is reused to estimate meter scale for a new relative teacher when present.
    An explicit anchor configuration allows inference when that cache is absent.
    New files and membership publish together under an optimistic manifest lock;
    concurrent edits cause a clean refusal, leaving existing work untouched.
    """
    from .display_training import display_target_eligibility
    from .learned_depth import LearnedDepthPredictor, validate_learned_depth_input
    from .pseudo_calibration import PseudoCalibrationError, anchor_relative_depth

    if teacher.model not in {"depthpro", "depth-anything-v2", "depth-anything-3"}:
        raise DatasetError("Choose DepthPro, Depth Anything V2 or Depth Anything 3")
    if metric_anchor is not None and metric_anchor.model != "depthpro":
        raise DatasetError("Metric anchoring requires an explicit DepthPro model configuration")
    if not sample_ids or any(not isinstance(value, str) or not value for value in sample_ids):
        raise DatasetError("Select at least one existing photo sample ID")
    if expected_manifest_sha256 is not None and (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in expected_manifest_sha256)):
        raise DatasetError("expected_manifest_sha256 must be a SHA-256 hex digest")
    worker_count = resolve_workers(workers)
    root = Path(directory).expanduser().resolve()
    generated: list[str] = []
    reenabled: list[str] = []
    unchanged: list[str] = []
    warnings: list[str] = []

    def emit(event: str, **details: Any):
        if progress_callback:
            progress_callback({"phase": "teacher_generation", "event": event, "model": teacher.model, **details})

    def report(manifest, digest):
        return {"dataset_path": str(root), "manifest_sha256": digest,
                "edit_revision": manifest.get("edit_revision", 0), "model": teacher.model,
                "generated_samples": generated, "reenabled_samples": reenabled,
                "unchanged_samples": unchanged, "summary": manifest.get("summary", {}),
                "warnings": warnings, "training_eligibility": display_target_eligibility(manifest)}

    with resource_lock(root, shared=True), resource_lock(root / ".ipde-teacher-generation-owner"):
        _, original, initial_digest = _read_manifest_snapshot(root, validate_files=False)
        _require_complete(original)
        load_dataset(root, metadata_only=True, snapshot=original)
        if expected_manifest_sha256 is not None and initial_digest != expected_manifest_sha256:
            raise DatasetError("Dataset changed since it was loaded. Reload before generating a teacher")
        samples = original["samples"]
        by_id = {sample["id"]: sample for sample in samples}
        unknown = set(sample_ids) - set(by_id)
        if unknown:
            raise DatasetError("Unknown photo sample IDs: " + ", ".join(sorted(unknown)))
        selected: dict[tuple[str, str], dict[str, Any]] = {}
        for identity in sample_ids:
            base = by_id[identity]
            selected.setdefault(_photo_key(base), base)
        missing: list[dict[str, Any]] = []
        restore_ids: dict[tuple[str, str], str] = {}
        cached_anchors: dict[tuple[str, str], Mapping[str, Any]] = {}
        for key, base in selected.items():
            related = [sample for sample in samples if isinstance(sample.get("display_rgb"), Mapping)
                       and (sample.get("source_sha256"), sample["display_rgb"].get("array_sha256")) == key]
            if any((sample["group_id"], sample["split"]) != (base["group_id"], base["split"]) for sample in related):
                raise DatasetError("Teacher variants of a selected photo have conflicting groups or splits; resolve those before generating")
            variants = [sample for sample in related if _model(sample) == teacher.model and _display_label(sample) is not None]
            matches = [sample for sample in variants if not sample.get("teacher_payload_removed", False)]
            for sample in matches:
                (reenabled if sample.get("excluded", False) else unchanged).append(sample["id"])
            if not matches:
                missing.append(base)
                removed = next((sample for sample in variants if sample.get("teacher_payload_removed", False)), None)
                if removed is not None:
                    restore_ids[key] = removed["id"]
            anchor = next((_display_label(sample) for sample in related if _model(sample) == "depthpro"
                           and not sample.get("teacher_payload_removed", False)
                           and _display_label(sample) is not None and _display_label(sample).get("units") == "meters"), None)
            if anchor is not None:
                cached_anchors[key] = anchor
        if not missing and not reenabled:
            return report(original, initial_digest)
        # Stored shapes permit admission checks before model loading, RGB reads
        # or staging files, leaving an existing dataset untouched on refusal.
        for sample in missing:
            validate_learned_depth_input(sample["display_rgb"]["shape"], teacher)
        manifest = copy.deepcopy(original)
        for sample in manifest["samples"]:
            if sample["id"] in reenabled:
                sample["excluded"] = False
        policy = original.get("storage_policy", {})
        if not isinstance(policy, Mapping):
            raise DatasetError("Dataset storage policy is malformed")
        storage = policy.get("array_format", "images")
        compressed = original.get("array_storage") != "npy"
        retain = policy.get("retain_intermediates", False)
        if storage not in {"images", "numpy"} or type(retain) is not bool:
            raise DatasetError("Dataset storage format or intermediate retention policy is malformed")
        operation = uuid.uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix=".teacher-generation-", dir=root.parent))
        payload = staging / "payload"
        payload.mkdir()
        final_payload = root / "arrays" / "teachers" / operation
        manifest_stage: Path | None = None
        published_payload = False
        manifest_committed = False
        published_digest: str | None = None
        new_samples: list[dict[str, Any]] = []
        saved_records: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}
        sample_bytes = 0

        def save(value, name):
            nonlocal sample_bytes
            value = np.asarray(value)
            identity = value.dtype.str, value.shape, sha256_array(value)
            if identity not in saved_records:
                record = array_record(payload, payload / name, value, compressed=compressed, storage=storage)
                record["path"] = (final_payload.relative_to(root) / record["path"]).as_posix()
                saved_records[identity] = record
                sample_bytes += value.nbytes
            return dict(saved_records[identity])

        def prediction_label(prediction, rgb, prefix, *, kind="pseudo_label"):
            target, native = np.asarray(prediction.source_depth), np.asarray(prediction.native_depth)
            metadata = dict(prediction.metadata)
            if target.dtype != np.float32 or target.shape != rgb.shape[:2] or native.dtype != np.float32 or native.ndim != 2:
                raise DatasetError("Teacher must preserve a float32 depth plane on the exact full display grid")
            if metadata.get("input_rgb_sha256") != sha256_array(rgb) or not metadata.get("checkpoint_sha256"):
                raise DatasetError("Teacher provenance does not match the verified selected display image and model checkpoint")
            if metadata.get("units") not in {"meters", "relative_depth", "relative_inverse_depth"}:
                raise DatasetError("Teacher output has unsupported units")
            valid = np.isfinite(target) & (target > 0)
            if not valid.any():
                raise DatasetError("Teacher returned no positive finite display depth values")
            metadata.update(dataset_teacher_view="display", reference_label="display", input_rgb_shape=list(rgb.shape),
                            stored_target_shape=list(target.shape), native_prediction_shape=list(native.shape))
            label = {"label_kind": kind, "units": metadata["units"], "metadata": _jsonable(metadata),
                     "target": save(target, prefix + "-depth"), "native_prediction_retained": retain,
                     "validity_policy": "stored_mask" if retain else "positive_finite",
                     "valid_pixel_count": int(valid.sum()), "coordinate_reference": "display; separate from RAFT's spatial_left reference"}
            if retain:
                label["native_target"] = save(native, prefix + "-native")
                label["valid_mask"] = save(valid, prefix + "-valid")
                if prediction.confidence is not None:
                    label["confidence"] = save(np.asarray(prediction.confidence), prefix + "-confidence")
                    label["confidence_note"] = "Model confidence is not a measured error bound"
            return label

        def close(predictor):
            if predictor is not None:
                predictor.close()

        try:
            predictor = None
            try:
                if missing:
                    emit("model_started", total=len(missing), completed=0)
                    predictor = LearnedDepthPredictor(teacher)
                for index, base in enumerate(missing):
                    rgb = _verified_array(root, base["display_rgb"])
                    result = predictor(rgb, reference_label="display")
                    label = prediction_label(result, rgb, f"photo-{index}-teacher")
                    restored_id = restore_ids.get(_photo_key(base))
                    sample = copy.deepcopy(by_id[restored_id] if restored_id else base)
                    for key in ("teacher", "display_teacher", "metric_anchor", "anchored_teacher", "raft_target",
                                "registered_display_teacher", "display_metric_anchor", "anchored_display_teacher",
                                "pseudo_calibration", "display_pseudo_calibration", "display_registration",
                                "metric_anchor_provenance", "original_teacher_provenance"):
                        sample.pop(key, None)
                    identity = hashlib.sha256((teacher.model + ":" + ":".join(_photo_key(base))).encode()).hexdigest()
                    sample.update(id=restored_id or "teacher-" + identity, teacher_id=teacher.model + "-on-demand", teacher_model=teacher.model,
                                  teacher_view="display", excluded=False, display_teacher=label,
                                  teacher={**copy.deepcopy(label), "alias_of": "display_teacher"},
                                  training_target_choice="display_teacher",
                                  training_target_choice_note="Full display teacher generated from the exact preserved display image; no stereo registration or resizing",
                                  display_grid_policy="Preserve full display coordinates; no automatic fixed registration or stereo-view inference")
                    sample.pop("teacher_payload_removed", None)
                    sample.pop("teacher_payload_removal", None)
                    sample["teacher_generation"] = {"base_sample_id": base["id"], "source_display_array_sha256": base["display_rgb"]["array_sha256"],
                                                    "uses_preserved_display_rgb": True}
                    if sample["id"] in by_id and sample["id"] != restored_id:
                        raise DatasetError("An incompatible stored result occupies this teacher's sample identity")
                    new_samples.append(sample)
                    generated.append(sample["id"])
                    emit("photo_ready", sample_id=base["id"], completed=index + 1, total=len(missing))
                    del result, rgb
            finally:
                close(predictor)
            # All requested teacher predictions finish and their model unloads
            # before any missing metric-anchor model is loaded.
            predictor = None
            try:
                for index, sample in enumerate(new_samples):
                    label = sample["display_teacher"]
                    if label["units"] == "meters":
                        continue
                    cached = cached_anchors.get(_photo_key(sample))
                    if cached is None and metric_anchor is None:
                        warnings.append(f"{sample['source_path']}: {teacher.model} uses {label['units']}; no same-photo DepthPro meter anchor is available. Keep unit conventions in separate training runs")
                        continue
                    if cached is not None:
                        metric_depth = _verified_array(root, cached["target"])
                        anchor_label = copy.deepcopy(dict(cached))
                        sample["teacher_generation"]["metric_anchor_reused"] = True
                    else:
                        if predictor is None:
                            emit("anchor_model_started")
                            predictor = LearnedDepthPredictor(metric_anchor)
                        rgb = _verified_array(root, sample["display_rgb"])
                        result = predictor(rgb, reference_label="display")
                        anchor_label = prediction_label(result, rgb, f"photo-{index}-anchor", kind="metric_model_anchor_pseudo_label")
                        metric_depth = np.asarray(result.source_depth)
                        del result, rgb
                    if anchor_label["units"] != "meters" or list(metric_depth.shape) != sample["display_rgb"]["shape"][:2]:
                        raise DatasetError("Metric anchor must use meters on the exact selected display grid")
                    if retain:
                        sample["display_metric_anchor"] = anchor_label
                    sample["metric_anchor_provenance"] = {**copy.deepcopy(anchor_label["metadata"]),
                        "target_array_sha256": anchor_label["target"]["array_sha256"],
                        "target_shape": anchor_label["target"]["shape"], "target_dtype": anchor_label["target"]["dtype"]}
                    from .array_storage import read_array
                    relative = read_array(payload / Path(label["target"]["path"]).name)
                    try:
                        anchored, valid, calibration = anchor_relative_depth(relative, label["units"], metric_depth)
                    except PseudoCalibrationError as exc:
                        sample["display_pseudo_calibration"] = {"accepted": False, "reason": str(exc), "anchor_is_measured": False}
                        warnings.append(f"{sample['source_path']}: relative teacher scale could not be anchored: {exc}")
                    else:
                        calibration = _jsonable(calibration)
                        sample["display_pseudo_calibration"] = calibration
                        sample["anchored_display_teacher"] = {"label_kind": "model_anchored_pseudo_depth", "units": "meters",
                            "coordinate_reference": label["coordinate_reference"], "target": save(anchored, f"photo-{index}-anchored"),
                            "valid_mask": save(valid, f"photo-{index}-anchored-valid"), "calibration": calibration,
                            "native_prediction_retained": False,
                            "metadata": {**label["metadata"], "units": "meters", "original_units": label["units"],
                                         "metric_anchor_checkpoint_sha256": anchor_label["metadata"]["checkpoint_sha256"],
                                         "pseudo_calibration": calibration}}
                        sample["training_target_choice"] = "anchored_display_teacher"
                        if not retain:
                            sample["original_teacher_provenance"] = {"units": label["units"],
                                "target_array_sha256": label["target"]["array_sha256"],
                                "target_shape": label["target"]["shape"], "target_dtype": label["target"]["dtype"],
                                "payload_retained": False}
                            selected_label = sample["anchored_display_teacher"]
                            sample["display_teacher"] = copy.deepcopy(selected_label)
                            sample["teacher"] = {**copy.deepcopy(selected_label), "alias_of": "display_teacher"}
                    del relative, metric_depth
            finally:
                close(predictor)
            # New records already use their unique final destination. Cached
            # anchors and core RGB records keep their exact existing paths.
            used = {record["path"] for record in _array_records(new_samples)}
            for record in saved_records.values():
                if record["path"] not in used:
                    (payload / Path(record["path"]).name).unlink()
            sample_bytes = sum(int(np.prod(record["shape"])) * np.dtype(record["dtype"]).itemsize
                               for record in saved_records.values() if record["path"] in used)
            replacements = {sample["id"]: sample for sample in new_samples if sample["id"] in by_id}
            manifest["samples"] = [replacements.get(sample["id"], sample) for sample in manifest["samples"]]
            manifest["samples"].extend(sample for sample in new_samples if sample["id"] not in replacements)
            teachers = manifest.setdefault("teachers", [])
            if not isinstance(teachers, list) or any(not isinstance(entry, Mapping) for entry in teachers):
                raise DatasetError("Dataset teacher registry is malformed")
            if new_samples and not any(entry.get("id") == teacher.model + "-on-demand" for entry in teachers):
                teachers.append({"id": teacher.model + "-on-demand", "model": teacher.model})
            active = [sample for sample in manifest["samples"] if not sample.get("excluded", False)]
            manifest["summary"] = {**manifest.get("summary", {}), **_summary(active),
                                   "source_photos": len({sample["source_sha256"] for sample in active})}
            added_bytes = sum(path.stat().st_size for path in payload.iterdir() if path.is_file())
            for name, increase in (("array_storage_bytes", added_bytes), ("array_sample_bytes", sample_bytes)):
                previous = manifest["summary"].get(name)
                if isinstance(previous, int) and not isinstance(previous, bool):
                    manifest["summary"][name] = previous + increase
            revision = original.get("edit_revision", 0)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise DatasetError("Dataset edit revision is malformed")
            manifest["edit_revision"] = revision + 1
            manifest["dataset_update"] = {"metadata_only": not new_samples, "added_samples": len(new_samples) - len(replacements),
                                          "excluded_samples": len(manifest["samples"]) - len(active),
                                          "split_changes_apply_to": "none; existing photo groups and splits preserved"}
            manifest["teacher_generation"] = {"model": teacher.model, "generated_samples": generated,
                                               "reenabled_samples": reenabled, "previous_manifest_sha256": initial_digest,
                                               "source_array_bytes_preserved": True, "existing_splits_preserved": True,
                                               "added_array_storage_bytes": added_bytes, "file_workers": worker_count}
            with resource_lock(root / "dataset.json"):
                if sha256_file(root / "dataset.json") != initial_digest:
                    raise DatasetError("Dataset changed during teacher generation. Reload before retrying; existing work was preserved")
                if new_samples:
                    if not final_payload.parent.resolve().is_relative_to(root):
                        raise DatasetError("Teacher output path escapes the dataset directory")
                    final_payload.parent.mkdir(parents=True, exist_ok=True)
                    _publish_new_directory(payload, final_payload)
                    published_payload = True
                # Validate all metadata, including purposeful retained/omitted
                # records, without rescanning existing scientific payloads.
                load_dataset(root, metadata_only=True, snapshot=manifest)
                serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
                published_digest = hashlib.sha256(serialized).hexdigest()
                descriptor, name = tempfile.mkstemp(prefix=".dataset-teachers-", suffix=".json", dir=root)
                manifest_stage = Path(name)
                with os.fdopen(descriptor, "wb") as stream:
                    os.fchmod(stream.fileno(), (root / "dataset.json").stat().st_mode & 0o777)
                    stream.write(serialized)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest_stage, root / "dataset.json")
                manifest_committed = True
                manifest_stage = None
            emit("complete", generated=len(generated), reenabled=len(reenabled))
            return report(manifest, published_digest)
        except BaseException:
            # Cancellation after replacement must never delete files already
            # referenced by the successfully committed manifest.
            try:
                committed = manifest_committed or (published_digest is not None and sha256_file(root / "dataset.json") == published_digest)
                if not committed and published_payload:
                    current = json.loads((root / "dataset.json").read_bytes())
                    prefix = final_payload.relative_to(root).as_posix() + "/"
                    committed = any(record["path"].startswith(prefix) for record in _array_records(current))
            except OSError:
                committed = True  # Uncertain publication must never delete a potentially referenced file.
            except (ValueError, TypeError, KeyError):
                committed = True
            if published_payload and not committed:
                shutil.rmtree(final_payload, ignore_errors=True)
            raise
        finally:
            if manifest_stage is not None:
                manifest_stage.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)


def disable_teacher(
    directory: Path | str,
    sample_ids: Sequence[str],
    model: str,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Turn a per-photo teacher off and discard only its unreferenced payloads.

    An exclusive dataset lock refuses active readers/training promptly. Core
    RGB, originals and shared labels stay untouched. The manifest commits its
    excluded placeholder first; interrupted garbage collection merely leaves
    orphan files, never a retained sample referring to a deleted payload.
    Restoring this placeholder must generate its teacher again.
    """
    from .display_training import display_target_eligibility
    if model not in {"depthpro", "depth-anything-v2", "depth-anything-3"}:
        raise DatasetError("Choose DepthPro, Depth Anything V2 or Depth Anything 3")
    if not sample_ids or any(not isinstance(value, str) or not value for value in sample_ids):
        raise DatasetError("Select at least one existing photo sample ID")
    root = Path(directory).expanduser().resolve()
    with resource_lock(root), resource_lock(root / "dataset.json"):
        _, original, digest = _read_manifest_snapshot(root, validate_files=False)
        _require_complete(original)
        load_dataset(root, metadata_only=True, snapshot=original)
        if expected_manifest_sha256 is not None and expected_manifest_sha256 != digest:
            raise DatasetError("Dataset changed since it was loaded. Reload before disabling a teacher")
        by_id = {sample["id"]: sample for sample in original["samples"]}
        unknown = set(sample_ids) - set(by_id)
        if unknown:
            raise DatasetError("Unknown photo sample IDs: " + ", ".join(sorted(unknown)))
        keys = {_photo_key(by_id[identity]) for identity in sample_ids}
        manifest = copy.deepcopy(original)
        removed = []
        candidates = {}
        labels = {"teacher", "display_teacher", "metric_anchor", "anchored_teacher", "raft_target",
                  "registered_display_teacher", "display_metric_anchor", "anchored_display_teacher"}
        for sample in manifest["samples"]:
            if (_model(sample) != model or not isinstance(sample.get("display_rgb"), Mapping)
                    or (sample.get("source_sha256"), sample["display_rgb"].get("array_sha256")) not in keys):
                continue
            for key in labels:
                for record in _array_records(sample.get(key, {})):
                    candidates.setdefault(record["path"], record)
            sample["excluded"] = True
            sample["teacher_payload_removed"] = True
            sample["teacher_payload_removal"] = {"model": model, "restore_requires_generation": True}
            removed.append(sample["id"])
        if not removed:
            raise DatasetError("The selected photos have no stored result for this teacher")
        retained = {record["path"] for record in _array_records(manifest["samples"])}
        # Core RGB/raw planes are also protected independently of the generic
        # traversal, guarding against a future removed-label schema expansion.
        for sample in manifest["samples"]:
            for key in ("rgb", "right_rgb", "display_rgb", "raw_assets", "reference"):
                retained.update(record["path"] for record in _array_records(sample.get(key, {})))
        from .dataset_review import _array_path
        deleting = []
        for name, record in candidates.items():
            if name in retained:
                continue
            path = _array_path(root, record, validate_file=False)
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise DatasetError("A teacher payload path escapes the dataset directory")
            if path.exists():
                # Hash only the teacher files being deleted, not every input.
                if not path.is_file() or sha256_file(path) != record["file_sha256"]:
                    raise DatasetError("Teacher payload changed since it was generated; inspect it before discarding")
                deleting.append(path)
        active = [sample for sample in manifest["samples"] if not sample.get("excluded", False)]
        manifest["summary"] = {**manifest.get("summary", {}), **_summary(active),
                               "source_photos": len({sample["source_sha256"] for sample in active})}
        revision = original.get("edit_revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise DatasetError("Dataset edit revision is malformed")
        manifest["edit_revision"] = revision + 1
        discarded_bytes = sum(path.stat().st_size for path in deleting)
        previous = manifest["summary"].get("array_storage_bytes")
        if isinstance(previous, int) and not isinstance(previous, bool):
            manifest["summary"]["array_storage_bytes"] = max(0, previous - discarded_bytes)
        manifest["dataset_update"] = {"metadata_only": False, "added_samples": 0,
                                      "excluded_samples": len(manifest["samples"]) - len(active),
                                      "split_changes_apply_to": "none; existing photo groups and splits preserved"}
        manifest["teacher_payload_removal"] = {"model": model, "sample_ids": removed,
                                               "discarded_array_storage_bytes": discarded_bytes,
                                               "core_rgb_preserved": True, "restore_requires_generation": True}
        load_dataset(root, metadata_only=True, snapshot=manifest)
        serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        if sha256_file(root / "dataset.json") != digest:
            raise DatasetError("Dataset changed while disabling a teacher; reload before retrying")
        descriptor, name = tempfile.mkstemp(prefix=".dataset-teacher-off-", suffix=".json", dir=root)
        staging = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), (root / "dataset.json").stat().st_mode & 0o777)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, root / "dataset.json")
        finally:
            staging.unlink(missing_ok=True)
        warnings = []
        deleted = 0
        for path in deleting:
            try:
                size = path.stat().st_size
                path.unlink()
                deleted += size
            except OSError as exc:
                warnings.append(f"Teacher is off, but its unused file could not be removed: {path.name}: {exc}")
        return {"dataset_path": str(root), "manifest_sha256": hashlib.sha256(serialized).hexdigest(),
                "edit_revision": manifest["edit_revision"], "model": model, "removed_samples": removed,
                "discarded_array_storage_bytes": deleted, "summary": manifest["summary"],
                "warnings": warnings, "training_eligibility": display_target_eligibility(manifest)}
