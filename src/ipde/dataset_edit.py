"""Edit dataset membership and splits without rewriting scientific arrays.

Edits publish a fresh verified dataset. Sources remain usable and immutable;
original NPY/NPZ file bytes, dtypes, NaN payloads and hashes are preserved.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .concurrency import memory_limited_workers, ordered_map, resolve_workers
from .dataset import DatasetError, load_dataset
from .dataset_collection import _require_complete, _share_array_file
from .dataset_review import _array_path, _array_records, _publish_new_directory, _read_manifest, _summary, _verified_array
from .formats import sha256_file


def _split_components(samples: list[dict[str, Any]], origins: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Use the same duplicate/burst/scene identities as dataset composition.

    Also retain every source's existing group boundaries, including a group
    whose connecting sample was removed. Scene names stay scoped to their
    original dataset namespace; category labels never join groups.
    """
    parents = list(range(len(samples)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, ...], int] = {}
    for index, (sample, origin) in enumerate(zip(samples, origins)):
        source_hash = sample.get("source_sha256")
        rgb_hash = sample["rgb"].get("array_sha256")
        if not isinstance(source_hash, str) or not source_hash or not isinstance(rgb_hash, str) or not rgb_hash:
            raise DatasetError("Every edited sample must identify its original photo and RGB array hashes")
        bursts = sample.get("burst_ids", [])
        if not isinstance(bursts, list) or any(not isinstance(item, str) or not item for item in bursts):
            raise DatasetError(f"Malformed capture burst IDs for sample {origin['source_sample_id']}")
        tokens = [("source", source_hash), ("rgb", rgb_hash),
                  ("existing-group", origin["source_dataset_path"], origin["source_group_id"]),
                  ("supplied-group", origin["group_namespace"], origin["source_group_id"])]
        tokens.extend(("burst", burst) for burst in bursts)
        prior_edit = sample.get("edit_provenance") or {}
        while prior_edit:
            if not isinstance(prior_edit, dict):
                raise DatasetError(f"Malformed edit provenance for sample {origin['source_sample_id']}")
            prior_root, prior_group = prior_edit.get("source_dataset_path"), prior_edit.get("source_group_id")
            if not isinstance(prior_root, str) or not prior_root or not isinstance(prior_group, str) or not prior_group:
                raise DatasetError(f"Malformed edit group provenance for sample {origin['source_sample_id']}")
            tokens.append(("existing-group", prior_root, prior_group))
            prior_edit = prior_edit.get("previous") or {}
        prior = sample.get("collection_provenance") or {}
        if not isinstance(prior, dict):
            raise DatasetError(f"Malformed collection provenance for sample {origin['source_sample_id']}")
        namespace = prior.get("group_namespace", origin["group_namespace"])
        if not isinstance(namespace, str) or not namespace:
            namespace = origin["group_namespace"]
        if prior.get("source_group_id"):
            if not isinstance(prior["source_group_id"], str):
                raise DatasetError(f"Malformed collection group for sample {origin['source_sample_id']}")
            tokens.append(("supplied-group", namespace, prior["source_group_id"]))
        requested = prior.get("source_requested_group", sample.get("requested_group"))
        if requested:
            if not isinstance(requested, str):
                raise DatasetError(f"Malformed scene group for sample {origin['source_sample_id']}")
            tokens.append(("scene", namespace, requested))
        for token in tokens:
            if token in seen:
                parents[root(index)] = root(seen[token])
            seen[token] = index
    grouped: dict[int, list[int]] = {}
    for index in range(len(samples)):
        grouped.setdefault(root(index), []).append(index)
    components = {}
    for indices in grouped.values():
        identities = sorted({samples[index]["source_sha256"] for index in indices})
        group = hashlib.sha256("\n".join(identities).encode()).hexdigest()
        components[group] = indices
        for index in indices:
            samples[index]["group_id"] = group
    return components


def review_split_components(directory: Path | str, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return inexpensive management group IDs matching durable edit rules.

    This reads metadata only and changes neither source samples nor their
    original groups. Dataset review already checks manifest/path boundaries.
    """
    root = Path(directory).expanduser().resolve()
    lineage = (manifest.get("curation") or {}).get("source_dataset_path")
    lineage = lineage or (manifest.get("dataset_edit") or {}).get("group_namespace") or str(root)
    samples = copy.deepcopy(manifest["samples"])
    origins = []
    for sample in samples:
        namespace = (sample.get("edit_provenance") or {}).get("group_namespace", lineage)
        if not isinstance(namespace, str) or not namespace:
            namespace = str(root)
        origins.append({"source_dataset_path": str(root), "source_sample_id": sample["id"],
                        "source_group_id": sample["group_id"], "group_namespace": namespace})
    _split_components(samples, origins)
    return {sample["id"]: sample["group_id"] for sample in samples}


def edit_dataset(
    directory: Path | str,
    edits: Mapping[str, Any],
    output_dir: Path | str,
    *,
    add_datasets: Sequence[Path | str] = (),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Keep base sample IDs, import complete datasets, and change group splits.

    ``edits`` accepts ``keep`` (base sample IDs, omitted means all) and
    ``splits`` (sample ID to train/validation, omitted means existing splits).
    Added IDs are deterministically namespaced and recorded in provenance.
    A split choice applies to its entire connected photo/teacher/burst/scene
    component. Conflicting choices, or inconsistent defaults without a choice,
    raise an actionable error instead of allowing validation leakage.
    """
    if not isinstance(edits, Mapping) or set(edits) - {"keep", "splits"}:
        raise DatasetError("Dataset edits must be an object containing only keep and splits")
    keep = edits.get("keep")
    if keep is not None and (not isinstance(keep, list) or any(not isinstance(item, str) or not item for item in keep)):
        raise DatasetError("keep must be a list of nonempty base sample IDs")
    if keep is not None and len(keep) != len(set(keep)):
        raise DatasetError("Duplicate kept sample IDs are not allowed")
    splits = edits.get("splits", {})
    if not isinstance(splits, dict) or any(not isinstance(key, str) or not key or not isinstance(value, str) or value not in {"train", "validation"} for key, value in splits.items()):
        raise DatasetError("splits must map sample IDs to train or validation")
    try:
        worker_count = resolve_workers(workers)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    inputs = [Path(directory).expanduser().resolve(), *(Path(item).expanduser().resolve() for item in add_datasets)]
    if len(set(inputs)) != len(inputs):
        raise DatasetError("A dataset was added more than once or is already the dataset being edited")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if any(destination.is_relative_to(root) for root in inputs):
        raise DatasetError("Edited dataset must be outside every source dataset")

    source_reports, samples, origins, warnings = [], [], [], []
    base_manifest: dict[str, Any] = {}
    for source_index, source_root in enumerate(inputs):
        if progress_callback:
            progress_callback({"phase": "verifying_dataset", "dataset_path": str(source_root),
                               "processed": source_index + 1, "total": len(inputs)})
        initial_hash = sha256_file(source_root / "dataset.json")
        _, manifest = _read_manifest(source_root)
        _require_complete(manifest)
        if load_dataset(source_root, verify=True, workers=worker_count) != manifest or sha256_file(source_root / "dataset.json") != initial_hash:
            raise DatasetError("Source dataset manifest changed before editing")
        source_id = hashlib.sha256((str(source_root) + ":" + initial_hash).encode()).hexdigest()
        if source_index == 0:
            base_manifest = manifest
            known = {sample["id"] for sample in manifest["samples"]}
            unknown = set(keep or ()) - known
            if unknown:
                raise DatasetError(f"Unknown base sample IDs: {', '.join(sorted(unknown))}")
        lineage = (manifest.get("curation") or {}).get("source_dataset_path")
        lineage = lineage or (manifest.get("dataset_edit") or {}).get("group_namespace") or str(source_root)
        kept_ids = set(keep) if keep is not None else None
        selected = [sample for sample in manifest["samples"] if source_index or kept_ids is None or sample["id"] in kept_ids]
        source_reports.append({"dataset_path": str(source_root), "source_manifest_sha256": initial_hash,
                               "source_samples": len(manifest["samples"]), "kept_samples": len(selected),
                               "metadata": {key: copy.deepcopy(value) for key, value in manifest.items() if key != "samples"}})
        for source_sample in selected:
            sample = copy.deepcopy(source_sample)
            prior_edit = source_sample.get("edit_provenance") or {}
            namespace = prior_edit.get("group_namespace", lineage)
            if not isinstance(namespace, str) or not namespace:
                namespace = str(source_root)
            origin = {"source_dataset_path": str(source_root), "source_manifest_sha256": initial_hash,
                      "source_sample_id": source_sample["id"], "source_group_id": source_sample["group_id"],
                      "source_split": source_sample["split"], "group_namespace": namespace}
            if prior_edit:
                origin["previous"] = copy.deepcopy(prior_edit)
            if source_index:
                sample["id"] = "added-" + hashlib.sha256((source_id + ":" + source_sample["id"]).encode()).hexdigest()
            # Preserve an existing entry's scientific/provenance metadata as-is.
            sample["edit_provenance"] = origin
            samples.append(sample)
            origins.append(origin)
        warnings.extend(item for item in manifest.get("warnings", []) if isinstance(item, str))
    if not samples:
        raise DatasetError("Keep at least one image or add a dataset; the edited dataset cannot be empty")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise DatasetError("An added sample ID collides with an existing ID; choose another source dataset")
    unknown = set(splits) - {sample["id"] for sample in samples}
    if unknown:
        raise DatasetError(f"Split choices refer to removed or unknown sample IDs: {', '.join(sorted(unknown))}")
    components = _split_components(samples, origins)
    for indices in components.values():
        explicit = {splits[samples[index]["id"]] for index in indices if samples[index]["id"] in splits}
        names = ", ".join(origins[index]["source_sample_id"] for index in indices[:4])
        if len(explicit) > 1:
            raise DatasetError(f"Conflicting split choices for linked photos ({names}). Give every linked photo, teacher variant, burst, or scene the same split")
        defaults = {samples[index]["split"] for index in indices}
        if not explicit and len(defaults) > 1:
            raise DatasetError(f"Linked photos have different existing splits ({names}). Set one of these images to Train or Validation to move the whole photo/teacher/burst/scene group together")
        chosen = next(iter(explicit or defaults))
        for index in indices:
            samples[index]["split"] = chosen
    summary = _summary(samples)
    if not summary["train_samples"] or not summary["validation_samples"]:
        warnings.append("Both training and validation images are required before training; assign an independent image group to each split")
    manifest = copy.deepcopy(base_manifest)
    manifest.update({"name": destination.name, "samples": samples, "summary": summary,
                     "group_ids": sorted(components), "generation_state": "complete", "splits_provisional": False,
                     "precision_policy": "Original NPY/NPZ files preserved byte-for-byte using copy-on-write clones or immutable hard links; no normalization, gamma, resampling or array rewriting"})
    if "generation_output_dir" in manifest:
        manifest["generation_output_dir"] = str(destination)
    for inherited in ("curation", "storage_compaction"):
        manifest.pop(inherited, None)  # Preserved in source metadata, no longer current storage statistics.
    editing = {"sources": source_reports, "group_namespace": str(inputs[0]),
               "kept_sample_ids": [sample["id"] for sample in samples[:source_reports[0]["kept_samples"]]],
               "excluded_sample_ids": [sample["id"] for sample in base_manifest["samples"] if keep is not None and sample["id"] not in keep],
               "added_samples": len(samples) - source_reports[0]["kept_samples"], "requested_splits": dict(splits),
               "array_bytes_preserved": True, "split_changes_apply_to": "linked photo/teacher/burst/scene components"}
    manifest["dataset_edit"] = editing
    manifest["explicit_scene_groups"] = all(report["metadata"].get("explicit_scene_groups") and report["metadata"].get("grouping_semantics") == "scene" for report in source_reports if report["kept_samples"])
    manifest["grouping_semantics"] = "scene" if manifest["explicit_scene_groups"] else "capture"
    storage: dict[str, Any] = {"storage_mode": "shared", "array_bytes_preserved": True, "file_workers": worker_count}
    manifest["collection"] = storage
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        planned: dict[Any, tuple[Path, dict[str, Any], list[dict[str, Any]]]] = {}
        for sample, origin in zip(samples, origins):
            root = Path(origin["source_dataset_path"])
            for record in _array_records(sample):
                # File hash stays part of the key: differing lossless containers
                # may have identical arrays but their original hashes must survive.
                key = (record["dtype"], tuple(record["shape"]), record["array_sha256"], record["file_sha256"], Path(record["path"]).suffix.lower())
                if key not in planned:
                    planned[key] = root, dict(record), []
                planned[key][2].append(record)

        def transfer(item: Any) -> tuple[Any, str, str]:
            key, (root, record, _) = item
            _verified_array(root, record)
            identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            source = _array_path(root, record)
            target = temporary / "arrays" / f"{identity}{source.suffix.lower()}"
            method = _share_array_file(source, target)
            if sha256_file(target) != record["file_sha256"]:
                raise DatasetError(f"Dataset checksum mismatch while reusing array: {record['path']}")
            return key, target.relative_to(temporary).as_posix(), method

        largest = max((math.prod(key[1]) * np.dtype(key[0]).itemsize for key in planned), default=1)
        transfer_workers = memory_limited_workers(worker_count, largest * 2)
        methods: dict[str, int] = {}
        for processed, (key, path, method) in enumerate(ordered_map(transfer, planned.items(), workers=transfer_workers), 1):
            methods[method] = methods.get(method, 0) + 1
            for record in planned[key][2]:
                record["path"] = path
            if progress_callback:
                progress_callback({"phase": "editing_dataset", "processed": processed, "total": len(planned), "storage_mode": "shared"})
        for report in source_reports:
            if sha256_file(Path(report["dataset_path"]) / "dataset.json") != report["source_manifest_sha256"]:
                raise DatasetError("Source dataset manifest changed during editing")
        array_bytes = sum(path.stat().st_size for path in (temporary / "arrays").iterdir())
        storage.update({"unique_arrays": len(planned), "storage_methods": methods, "transfer_workers": transfer_workers,
                        "array_storage_bytes": array_bytes, "reused_array_storage_bytes": array_bytes,
                        "added_array_storage_bytes": 0, "shared_array_files_immutable": bool(methods.get("hardlink")),
                        "storage_format": "deduplicated original NPY/NPZ files"})
        if methods.get("hardlink"):
            warnings.append("Edited arrays share immutable files with their sources. Archiving or removing a source keeps this dataset usable; do not edit shared array files in place.")
        manifest["warnings"] = sorted(set(warnings))
        storage["added_metadata_bytes"] = 0
        while True:
            serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if storage["added_metadata_bytes"] == len(serialized):
                break
            storage["added_metadata_bytes"] = len(serialized)
            storage["added_storage_bytes"] = len(serialized)
        (temporary / "dataset.json").write_bytes(serialized)
        if progress_callback:
            progress_callback({"phase": "verifying_output"})
        _read_manifest(temporary)
        load_dataset(temporary, verify=True, workers=worker_count)
        if progress_callback:
            progress_callback({"phase": "publishing_dataset"})
        _publish_new_directory(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"dataset_path": str(destination), "summary": summary, "warnings": manifest["warnings"],
            "dataset_edit": editing, "collection": storage}
