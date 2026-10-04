"""Compose reviewed datasets without changing any stored sample values.

Subject/category labels are annotations. A split unit is a connected component
of duplicate photos, teacher variants, known bursts and (optionally) supplied
capture/scene groups. Ignoring supplied groups never disables duplicate/burst
protection. The resulting ordinary v1 dataset is usable by the existing trainer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .array_storage import array_record, read_array
from .dataset import DatasetError, load_dataset
from .dataset_review import _array_path, _array_records, _publish_new_directory, _read_manifest, _verified_array
from .formats import sha256_file


@dataclass(frozen=True)
class CollectionOptions:
    split_mode: str = "global-random"
    validation_fraction: float = 0.2
    split_seed: int = 0
    grouping: str = "preserve"
    validation_count_per_dataset: int | None = None


def _require_complete(manifest: Mapping[str, Any]) -> None:
    if manifest.get("generation_state") not in {None, "complete"} or manifest.get("splits_provisional"):
        raise DatasetError("Finish dataset generation before composing or compacting it; photos can still be reviewed while generation runs")


def _validate_options(options: CollectionOptions) -> None:
    if options.split_mode not in {"global-random", "equal-per-dataset", "explicit"}:
        raise DatasetError("split_mode must be global-random, equal-per-dataset, or explicit")
    if options.grouping not in {"preserve", "ignore"}:
        raise DatasetError("grouping must be preserve or ignore")
    if not math.isfinite(options.validation_fraction) or not 0 < options.validation_fraction < 1:
        raise DatasetError("validation_fraction must be strictly between zero and one")
    if not isinstance(options.split_seed, int) or isinstance(options.split_seed, bool):
        raise DatasetError("split_seed must be an integer")
    count = options.validation_count_per_dataset
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise DatasetError("validation_count_per_dataset must be a positive integer")
    if count is not None and options.split_mode != "equal-per-dataset":
        raise DatasetError("A per-dataset validation count requires equal-per-dataset splitting")


def _components(samples: list[dict[str, Any]], grouping: str) -> dict[str, list[int]]:
    parents = list(range(len(samples)))

    def root(i: int) -> int:
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[str, int] = {}
    for index, sample in enumerate(samples):
        origin = sample["collection_provenance"]
        source_hash, rgb_hash = sample.get("source_sha256"), sample["rgb"].get("array_sha256")
        if not isinstance(source_hash, str) or not source_hash or not isinstance(rgb_hash, str) or not rgb_hash:
            raise DatasetError("Every collection sample must identify the original photo and RGB array hashes")
        tokens = [f"source:{source_hash}", f"rgb:{rgb_hash}"]
        bursts = sample.get("burst_ids", [])
        if not isinstance(bursts, list) or any(not isinstance(item, str) or not item for item in bursts):
            raise DatasetError("Malformed capture burst IDs in a source dataset")
        tokens.extend(f"burst:{item}" for item in bursts)
        if grouping == "preserve":
            namespace = origin["group_namespace"]
            tokens.append(f"supplied:{namespace}:{origin['source_group_id']}")
            if origin.get("source_requested_group"):
                tokens.append(f"scene:{namespace}:{origin['source_requested_group']}")
        for token in tokens:
            if token in seen:
                parents[root(index)] = root(seen[token])
            seen[token] = index
    grouped: dict[int, list[int]] = {}
    for index in range(len(samples)):
        grouped.setdefault(root(index), []).append(index)
    components: dict[str, list[int]] = {}
    for indices in grouped.values():
        # Scene namespaces determine membership, while immutable source identity
        # determines the split seed. Editing teacher choices/provenance or adding
        # variants cannot perturb an unchanged collection of photo components.
        identities = sorted({samples[index]["source_sha256"] for index in indices})
        group = hashlib.sha256("\n".join(identities).encode()).hexdigest()
        components[group] = indices
        for index in indices:
            samples[index]["group_id"] = group
    return components


def _assign_splits(samples: list[dict[str, Any]], options: CollectionOptions, source_ids: list[str]) -> list[str]:
    components = _components(samples, options.grouping)
    if len(components) < 2:
        raise DatasetError("Training collection needs at least two independent photo groups; teacher variants do not add independent photos")
    warnings: list[str] = []
    order = sorted(components, key=lambda key: hashlib.sha256(f"{options.split_seed}:{key}".encode()).hexdigest())
    validation: set[str] = set()
    if options.split_mode == "explicit":
        for group, indices in components.items():
            roles = {samples[index]["collection_provenance"]["role"] for index in indices}
            if len(roles) > 1:
                raise DatasetError("Explicit validation overlaps training: a duplicate photo, teacher variant, burst, or preserved group occurs in both; remove that overlap")
            if roles == {"validation"}:
                validation.add(group)
    elif options.split_mode == "global-random":
        count = min(len(order) - 1, max(1, round(len(order) * options.validation_fraction)))
        validation.update(order[:count])
    else:
        membership = {
            group: {samples[index]["collection_provenance"]["dataset_id"] for index in indices}
            for group, indices in components.items()
        }
        shared = {group for group in order if len(membership[group]) > 1}
        pools = {source: [group for group in order if membership[group] == {source}] for source in source_ids}
        capacities = []
        for source in source_ids:
            has_shared_training = any(source in membership[group] for group in shared)
            capacities.append(len(pools[source]) - (0 if has_shared_training else 1))
        capacity = min(capacities)
        if capacity < 1:
            raise DatasetError("Equal-per-dataset validation needs independent photos in every dataset after duplicate/burst/group protection; use global random splitting or add independent photos")
        count = options.validation_count_per_dataset
        if count is None:
            count = min(capacity, max(1, round(min(len(pool) for pool in pools.values()) * options.validation_fraction)))
        if count > capacity:
            raise DatasetError(f"Requested {count} validation groups per dataset, but only {capacity} can be held out while preserving training and leakage protection")
        for source in source_ids:
            validation.update(pools[source][:count])
        if shared:
            warnings.append("Shared photo/burst/group components remain in training so each dataset contributes exactly the same number of independent validation groups")
        warnings.append("Equal splitting balances independent photo groups; teacher variants and groups with multiple captures may yield different sample-entry counts")
    for sample in samples:
        sample["split"] = "validation" if sample["group_id"] in validation else "train"
    if not validation or validation == set(components):
        raise DatasetError("Training collection must contain both training and validation groups")
    return warnings


def compose_datasets(
    datasets: Sequence[Path | str],
    output_dir: Path | str,
    options: CollectionOptions | None = None,
    *,
    validation_datasets: Sequence[Path | str] = (),
    categories: Mapping[str, str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Make an atomic, lossless training collection from reviewed v1 datasets.

    ``categories`` maps absolute dataset paths to subject labels (for example
    flowers or architecture). They never become scene/leakage groups. Global
    random splitting samples independent components from the whole collection;
    equal splitting samples equal counts from each dataset. External validation
    datasets are held out completely and overlapping captures are rejected.
    """
    options = options or CollectionOptions()
    _validate_options(options)
    if not datasets:
        raise DatasetError("Choose at least one training dataset")
    if options.split_mode == "explicit" and not validation_datasets:
        raise DatasetError("Choose a separate validation dataset for explicit validation")
    if options.split_mode != "explicit" and validation_datasets:
        raise DatasetError("Separate validation datasets require explicit splitting")
    inputs = [(Path(item).expanduser().resolve(), "train") for item in datasets]
    inputs.extend((Path(item).expanduser().resolve(), "validation") for item in validation_datasets)
    if len({root for root, _ in inputs}) != len(inputs):
        raise DatasetError("A dataset was selected more than once or as both training and validation")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if any(destination.is_relative_to(root) for root, _ in inputs):
        raise DatasetError("Training collection must be outside every source dataset")
    normalized_categories = {str(Path(key).expanduser().resolve()): value for key, value in (categories or {}).items()}
    if any(not isinstance(value, str) or not value.strip() for value in normalized_categories.values()):
        raise DatasetError("Dataset category labels must be nonempty strings")
    samples: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for source_index, (root, role) in enumerate(inputs, 1):
        if progress_callback is not None:
            progress_callback({"phase": "verifying_dataset", "dataset_path": str(root), "role": role,
                               "processed": source_index, "total": len(inputs)})
        initial_hash = sha256_file(root / "dataset.json")
        _, manifest = _read_manifest(root)
        _require_complete(manifest)
        if load_dataset(root, verify=True) != manifest or sha256_file(root / "dataset.json") != initial_hash:
            raise DatasetError("Source dataset manifest changed before composition")
        # Include optional future array records as well as all current v1 fields.
        for sample in manifest["samples"]:
            for record in _array_records(sample):
                _verified_array(root, record)
        dataset_id = hashlib.sha256((str(root) + ":" + initial_hash).encode()).hexdigest()
        sources[dataset_id] = root
        manifests[dataset_id] = manifest
        category = normalized_categories.get(str(root), manifest.get("category", manifest.get("category_label")))
        if not isinstance(category, str) or not category.strip():
            category = None
        source_reports.append({"dataset_id": dataset_id, "dataset_path": str(root), "source_manifest_sha256": initial_hash,
                               "name": manifest.get("name") or root.name,
                               "role": role, "category_label": category, "source_samples": len(manifest["samples"])})
        for source_sample in manifest["samples"]:
            sample = copy.deepcopy(source_sample)
            lineage = (manifest.get("curation") or {}).get("source_dataset_path")
            prior = source_sample.get("collection_provenance") or {}
            namespace = prior.get("group_namespace", lineage)
            if not isinstance(namespace, str) or not namespace:
                namespace = str(root)
            sample["id"] = hashlib.sha256((dataset_id + ":" + source_sample["id"]).encode()).hexdigest()
            sample["collection_provenance"] = {
                "dataset_id": dataset_id, "source_dataset_path": str(root), "source_manifest_sha256": initial_hash,
                "source_sample_id": source_sample["id"], "source_group_id": source_sample["group_id"],
                "source_requested_group": source_sample.get("requested_group"), "source_split": source_sample["split"],
                "role": role, "category_label": category, "group_namespace": namespace,
            }
            original_group = source_sample.get("requested_group")
            sample["requested_group"] = f"{namespace}:{original_group}" if original_group and options.grouping == "preserve" else None
            if category is not None and not sample.get("category_label"):
                sample["category_label"] = category
            samples.append(sample)
        warnings.extend(str(item) for item in manifest.get("warnings", []) if isinstance(item, str))
    warnings.extend(_assign_splits(samples, options, [report["dataset_id"] for report in source_reports]))
    if options.grouping == "ignore":
        warnings.append("Supplied capture/scene groups ignored for this experiment; duplicate photos, teacher variants and known bursts still remain together")
    for report in source_reports:
        subset = [sample for sample in samples if sample["collection_provenance"]["dataset_id"] == report["dataset_id"]]
        report.update({
            "train_samples": sum(sample["split"] == "train" for sample in subset),
            "validation_samples": sum(sample["split"] == "validation" for sample in subset),
            "train_groups": len({sample["group_id"] for sample in subset if sample["split"] == "train"}),
            "validation_groups": len({sample["group_id"] for sample in subset if sample["split"] == "validation"}),
        })
    explicit_scenes = options.grouping == "preserve" and all(manifest.get("explicit_scene_groups") and manifest.get("grouping_semantics") == "scene" for manifest in manifests.values())
    if not explicit_scenes:
        warnings.append("Category labels do not establish independent scenes; unknown related captures may require explicit scene grouping")
    summary = {
        "samples": len(samples), "groups": len({sample["group_id"] for sample in samples}),
        "train_samples": sum(sample["split"] == "train" for sample in samples),
        "validation_samples": sum(sample["split"] == "validation" for sample in samples),
    }
    collection = {"sources": source_reports, "split_mode": options.split_mode, "grouping": options.grouping,
                  "validation_count_per_dataset": options.validation_count_per_dataset,
                  "random_sampling_unit": "independent photo/burst/scene component; all teacher variants stay together",
                  "array_bytes_preserved": True, "original_split_assignments_preserved": False}
    manifest = {
        "schema": "ipde-depth-dataset-v1", "precision_policy": "Array payloads preserved bit-for-bit in deduplicated lossless NPZ without changing any array values",
        "split_seed": options.split_seed, "validation_fraction": options.validation_fraction,
        "group_ids": sorted({sample["group_id"] for sample in samples}), "explicit_scene_groups": bool(explicit_scenes),
        "grouping_semantics": "scene" if explicit_scenes else "capture", "warnings": sorted(set(warnings)),
        "samples": samples, "summary": summary, "collection": collection,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        copied: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}
        for sample_index, sample in enumerate(samples, 1):
            origin = sample["collection_provenance"]
            dataset_id, root = origin["dataset_id"], sources[origin["dataset_id"]]
            for record in _array_records(sample):
                source_name = record["path"]
                key = record["dtype"], tuple(record["shape"]), record["array_sha256"]
                if key not in copied:
                    # All identical planes share one physical file, including
                    # RGB repeated in raw_assets and teacher-variant datasets.
                    identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()
                    value = _verified_array(root, record)
                    copied[key] = array_record(temporary, temporary / "arrays" / f"{identity}.npz", value, compressed=True)
                    if any(copied[key][field] != record[field] for field in ("array_sha256", "shape", "dtype")):
                        raise DatasetError(f"Lossless composition changed source array values: {source_name}")
                record.update(copied[key])
            if progress_callback is not None:
                progress_callback({"phase": "sample_composed", "sample_id": sample["id"], "unique_arrays": len(copied),
                                   "processed": sample_index, "total": len(samples)})
        collection["unique_arrays"] = len(copied)
        collection["storage_format"] = "deduplicated lossless NPZ"
        for report in source_reports:
            if sha256_file(sources[report["dataset_id"]] / "dataset.json") != report["source_manifest_sha256"]:
                raise DatasetError("Source dataset manifest changed during composition")
        (temporary / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if progress_callback is not None:
            progress_callback({"phase": "verifying_output"})
        load_dataset(temporary, verify=True)
        if progress_callback is not None:
            progress_callback({"phase": "publishing_dataset"})
        _publish_new_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"dataset_path": str(destination), "summary": summary, "warnings": manifest["warnings"], "collection": collection}


def compress_dataset(
    directory: Path | str,
    output_dir: Path | str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Publish a new deduplicated lossless ZIP-compressed array dataset.

    Numerical payloads (including NaNs, signed zero, endian and dtype) must have
    exactly the same array hash. The original remains usable and untouched.
    Each unique plane is stored once, regardless of how many sample records or
    teachers reference it. Publication refuses existing/raced-in destinations.
    """
    root, manifest = _read_manifest(directory)
    _require_complete(manifest)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}; select a new directory")
    if destination.is_relative_to(root):
        raise DatasetError("Compressed dataset must be outside the original dataset")
    source_manifest_hash = sha256_file(root / "dataset.json")
    if load_dataset(root, verify=True) != manifest:
        raise DatasetError("Source dataset manifest changed before compression")
    compact = copy.deepcopy(manifest)
    records = list(_array_records(compact["samples"]))
    source_files = {_array_path(root, record) for record in records}
    before = sum(path.stat().st_size for path in source_files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        written: dict[tuple[str, tuple[int, ...], str], dict[str, Any]] = {}
        for index, record in enumerate(records):
            key = record["dtype"], tuple(record["shape"]), record["array_sha256"]
            # Verification includes every source file, even if another copy of
            # the same plane has already been compacted successfully.
            _verified_array(root, record)
            if key not in written:
                identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()
                value = read_array(_array_path(root, record))
                saved = array_record(temporary, temporary / "arrays" / f"{identity}.npz", value, compressed=True)
                if any(saved[field] != record[field] for field in ("array_sha256", "shape", "dtype")):
                    raise DatasetError("Lossless compression changed a source array's values, shape or dtype")
                written[key] = saved
            record.update(written[key])
            if progress_callback is not None:
                progress_callback({"phase": "compressing", "processed": index + 1, "total": len(records), "unique_arrays": len(written)})
        if sha256_file(root / "dataset.json") != source_manifest_hash:
            raise DatasetError("Source dataset manifest changed during compression")
        after = sum((temporary / record["path"]).stat().st_size for record in written.values())
        compaction = {
            "source_dataset_path": str(root), "source_manifest_sha256": source_manifest_hash,
            "array_bytes_before": before, "array_bytes_after": after,
            "compression_ratio": after / before if before else 1.0,
            "unique_arrays": len(written), "source_array_files": len(source_files),
            "array_values_preserved": True, "split_assignments_preserved": True,
            "format": "NPZ: ZIP deflate of one data.npy plane per unique array",
        }
        compact["storage_compaction"] = compaction
        if "generation_output_dir" in compact:
            compact["generation_output_dir"] = str(destination)
        compact["precision_policy"] = "Lossless ZIP-compressed NPY planes, verified by dtype/shape/array hashes; no normalization, gamma or resampling"
        (temporary / "dataset.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        load_dataset(temporary, verify=True)
        _publish_new_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"dataset_path": str(destination), "summary": compact.get("summary", {}),
            "warnings": compact.get("warnings", []), "storage_compaction": compaction}
