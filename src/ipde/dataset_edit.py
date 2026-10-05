"""Edit dataset metadata without reading or rewriting scientific arrays.

Ordinary edits update one manifest atomically. The optional legacy versioning
API still publishes a separately verified dataset for explicit copy workflows.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
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


def _metadata_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _committed_content_digest(manifest: Mapping[str, Any]) -> str:
    """Hash all committed metadata, excluding only this digest's own field."""
    metadata = dict(manifest)
    audit = dict(metadata["dataset_update"])
    audit.pop("committed_content_sha256", None)
    metadata["dataset_update"] = audit
    return _metadata_digest(metadata)


def _update_report(root: Path, manifest: Mapping[str, Any], manifest_hash: str, *, already_applied: bool) -> dict[str, Any]:
    from .training import training_target_eligibility
    samples = manifest["samples"]
    active = [sample for sample in samples if not sample.get("excluded", False)]
    audit = manifest["dataset_update"]
    return {"dataset_path": str(root), "summary": manifest["summary"],
            "excluded_samples": len(samples) - len(active), "added_samples": audit["added_samples"],
            "manifest_sha256": manifest_hash, "edit_revision": manifest["edit_revision"],
            "validation_fraction": manifest.get("validation_fraction"), "split_seed": manifest.get("split_seed"),
            "pending_photos": manifest.get("pending_photos", []),
            "training_eligibility": training_target_eligibility(manifest),
            "warnings": manifest.get("warnings", []), "dataset_update": audit,
            "operation_id": audit.get("operation_id"), "already_applied": already_applied}


def apply_dataset_edits(
    directory: Path | str,
    edits: Mapping[str, Any],
    *,
    add_datasets: Sequence[Path | str] = (),
    expected_manifest_sha256: str | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Atomically apply membership and split changes to the existing dataset.

    ``keep`` is a complete snapshot of included base sample IDs; excluded
    entries and files remain available to Restore. ``splits`` moves complete
    related capture groups. ``validation_fraction`` (0 through 1 inclusive)
    replaces split assignments deterministically using optional integer
    ``seed``. None of these operations opens, hashes, or decodes array files.
    A saved automatic fraction is reapplied when photos are removed, restored,
    or added. Individual split choices disable that automatic split policy.

    Prepared additions use hard links to *only* their new files. Files on a
    different filesystem are rejected rather than quietly starting a large
    copy. A shared dataset lock protects immutable files from cleanup while an
    exclusive manifest lock serializes editors. Training uses its captured
    manifest and can continue during edits. An expected hash rejects stale GUI
    changes before any write occurs.
    An ``operation_id`` plus expected hash permits safe replay after a crash
    between atomic commit and acknowledgement. Only an identical request with
    verified unchanged committed metadata is acknowledged without writing.
    """
    from .dataset_review import _read_manifest_snapshot
    from .resource_lock import resource_lock
    from .training import training_target_eligibility

    allowed = {"keep", "splits", "validation_fraction", "seed", "pending_photos"}
    if not isinstance(edits, Mapping) or set(edits) - allowed:
        raise DatasetError("Dataset updates must contain only keep, splits, validation_fraction, seed and pending_photos")
    keep = edits.get("keep")
    if keep is not None and (not isinstance(keep, list) or any(not isinstance(item, str) or not item for item in keep)):
        raise DatasetError("keep must be a list of nonempty base sample IDs")
    if keep is not None and len(keep) != len(set(keep)):
        raise DatasetError("Duplicate kept sample IDs are not allowed")
    splits = edits.get("splits", {})
    if not isinstance(splits, dict) or any(not isinstance(key, str) or not key or not isinstance(value, str)
                                           or value not in {"train", "validation"} for key, value in splits.items()):
        raise DatasetError("splits must map sample IDs to train or validation")
    pending = edits.get("pending_photos")
    if "pending_photos" in edits and (not isinstance(pending, list) or any(
            not isinstance(path, str) or not path or not Path(path).is_absolute() for path in pending)
            or len(pending) != len(set(pending))):
        raise DatasetError("pending_photos must be a list of unique absolute photo paths")
    fraction = edits.get("validation_fraction")
    if "validation_fraction" in edits and (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                                             or not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise DatasetError("validation_fraction must be between zero and one inclusive")
    if "validation_fraction" in edits and splits:
        raise DatasetError("Choose either an automatic validation fraction or individual split assignments")
    seed = edits.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError("seed must be an integer")
    if expected_manifest_sha256 is not None and (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)):
        raise DatasetError("expected_manifest_sha256 must be a SHA-256 hex digest")
    if operation_id is not None:
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128 or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-" for character in operation_id):
            raise DatasetError("operation_id must be a nonempty unique save ID using letters, digits, underscore, hyphen, colon or period")
        if expected_manifest_sha256 is None:
            raise DatasetError("A recoverable operation requires expected_manifest_sha256 from the loaded dataset")
    root = Path(directory).expanduser().resolve()
    additions = [Path(item).expanduser().resolve() for item in add_datasets]
    if root in additions or len(additions) != len(set(additions)):
        raise DatasetError("A dataset was added more than once or is already the dataset being edited")
    request_hash = _metadata_digest({"edits": dict(edits), "add_datasets": [str(path) for path in additions]}) if operation_id is not None else None
    with ExitStack() as locks:
        for path in sorted({root, *additions}):
            locks.enter_context(resource_lock(path, shared=True))
        locks.enter_context(resource_lock(root / "dataset.json"))
        _, original, original_hash = _read_manifest_snapshot(root, validate_files=False)
        _require_complete(original)
        prior_update = original.get("dataset_update")
        if operation_id is not None and isinstance(prior_update, dict) and prior_update.get("operation_id") == operation_id:
            if prior_update.get("previous_manifest_sha256") != expected_manifest_sha256:
                raise DatasetError("This saved operation belongs to a different dataset snapshot. Reload before applying new edits")
            if prior_update.get("request_sha256") != request_hash:
                raise DatasetError("This saved operation ID was already used for different edits. Recover its original request before applying new edits")
            if prior_update.get("committed_content_sha256") != _committed_content_digest(original):
                raise DatasetError("Dataset changed after this save committed. Reload before applying pending changes")
            # Additions may already have been deleted after successful linking.
            # Proof comes from the committed metadata, never their payloads.
            return _update_report(root, original, original_hash, already_applied=True)
        if expected_manifest_sha256 is not None and original_hash != expected_manifest_sha256:
            raise DatasetError("Dataset changed since it was loaded. Reload its photos before applying your changes")
        manifest = copy.deepcopy(original)
        if pending is not None:
            manifest["pending_photos"] = list(pending)
        samples = manifest["samples"]
        base_sample_count = len(samples)
        known = {sample["id"] for sample in samples}
        unknown = set(keep or ()) - known
        if unknown:
            raise DatasetError(f"Unknown base sample IDs: {', '.join(sorted(unknown))}")
        membership_changed = False
        if keep is not None:
            included = set(keep)
            for sample in samples:
                excluded = sample["id"] not in included
                if not excluded and sample.get("teacher_payload_removed", False):
                    raise DatasetError("This teacher's depth files were discarded. Generate its teacher again to restore it")
                membership_changed |= excluded != sample.get("excluded", False)
                sample["excluded"] = excluded

        def origin_for(sample: Mapping[str, Any], source: Path, source_hash: str) -> dict[str, Any]:
            previous = sample.get("edit_provenance") or {}
            lineage = (manifest.get("curation") or {}).get("source_dataset_path") if source == root else None
            namespace = previous.get("group_namespace", lineage or str(source))
            if not isinstance(namespace, str) or not namespace:
                namespace = str(source)
            result = {"source_dataset_path": str(source), "source_manifest_sha256": source_hash,
                      "source_sample_id": sample["id"], "source_group_id": sample["group_id"],
                      "source_split": sample["split"], "group_namespace": namespace}
            if previous:
                result["previous"] = copy.deepcopy(previous)
            return result

        origins = [origin_for(sample, root, original_hash) for sample in samples]
        transfers: dict[str, tuple[Path, str]] = {}
        source_hashes: dict[Path, str] = {}
        added_count = 0
        completed_photos: set[str] = set()
        for source in additions:
            _, addition, source_hash = _read_manifest_snapshot(source, validate_files=False)
            _require_complete(addition)
            source_hashes[source] = source_hash
            for entry in addition["samples"]:
                if entry.get("excluded", False):
                    continue
                identity = hashlib.sha256((str(source) + ":" + entry["id"]).encode()).hexdigest()
                sample_id = "added-" + identity
                if sample_id in known:
                    completed_photos.add(entry["source_path"])
                    continue  # Re-attaching an already imported dataset is idempotent.
                sample = copy.deepcopy(entry)
                origin = origin_for(entry, source, source_hash)
                sample["id"] = sample_id
                sample["excluded"] = False
                sample["edit_provenance"] = origin
                for record in _array_records(sample):
                    # Only newly added records touch the filesystem. Existing
                    # entries retain their path, bytes, provenance and hashes.
                    source_path = _array_path(source, record)
                    file_identity = hashlib.sha256((str(source) + ":" + record["path"] + ":"
                                                      + record["file_sha256"]).encode()).hexdigest()
                    target_name = f"arrays/added/{file_identity}{source_path.suffix.lower()}"
                    transfers.setdefault(target_name, (source_path, record["file_sha256"]))
                    record["path"] = target_name
                samples.append(sample)
                origins.append(origin)
                known.add(sample_id)
                added_count += 1
                completed_photos.add(entry["source_path"])
        if completed_photos:
            # Generation records canonical source paths; a queued selection can
            # retain a symlinked folder spelling. Resolve identities without
            # opening the photos, allowing missing queued sources for recovery.
            completed_paths = {Path(path).expanduser().resolve(strict=False) for path in completed_photos}
            manifest["pending_photos"] = [path for path in manifest.get("pending_photos", [])
                                          if Path(path).expanduser().resolve(strict=False) not in completed_paths]
        unknown = set(splits) - known
        if unknown:
            raise DatasetError(f"Split choices refer to unknown sample IDs: {', '.join(sorted(unknown))}")
        components = _split_components(samples, origins)
        automatic_split = "validation_fraction" in edits
        if not automatic_split and not splits and (membership_changed or added_count) and original.get("validation_fraction") is not None:
            # Automatic settings describe current included membership rather
            # than a historical one-time generation decision. Restore/add must
            # not silently contradict the percentage shown in Dataset Studio.
            fraction = original["validation_fraction"]
            seed = edits.get("seed", original.get("split_seed", 0))
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise DatasetError("Saved split seed must be an integer")
            automatic_split = True
        if automatic_split:
            active_groups = [group for group, indices in components.items()
                             if any(not samples[index].get("excluded", False) for index in indices)]
            order = sorted(active_groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
            count = len(order) if fraction == 1 else 0 if fraction == 0 else (
                min(len(order) - 1, max(1, round(len(order) * fraction))) if len(order) >= 2 else 0)
            validation = set(order[:count])
            for group, indices in components.items():
                for index in indices:
                    samples[index]["split"] = "validation" if group in validation else "train"
            manifest["validation_fraction"] = fraction
            manifest["split_seed"] = seed
        else:
            for indices in components.values():
                explicit = {splits[samples[index]["id"]] for index in indices if samples[index]["id"] in splits}
                if len(explicit) > 1:
                    raise DatasetError("Conflicting split choices for linked photos. Give every linked photo, teacher variant, burst or scene the same split")
                # A generated addition has its own provisional/random split.
                # Existing capture membership is authoritative when the added
                # photo/teacher/burst belongs to a group already in this data.
                base_defaults = {samples[index]["split"] for index in indices if index < base_sample_count}
                defaults = base_defaults or {samples[index]["split"] for index in indices}
                if not explicit and len(defaults) > 1:
                    raise DatasetError("Linked photos have different existing splits. Set one of these images to Train or Validation to move the whole related capture group together")
                selected = next(iter(explicit or defaults))
                for index in indices:
                    samples[index]["split"] = selected
            if splits:
                manifest.pop("validation_fraction", None)
        active = [sample for sample in samples if not sample.get("excluded", False)]
        manifest["summary"] = {**manifest.get("summary", {}), **_summary(active),
                               "source_photos": len({sample["source_sha256"] for sample in active})}
        stored_bytes = manifest["summary"].get("array_storage_bytes")
        if not original.get("dataset_update") and original.get("storage_compaction"):
            stored_bytes = original["storage_compaction"].get("array_bytes_after")
        elif stored_bytes is None and not original.get("curation"):
            stored_bytes = (original.get("collection") or {}).get("array_storage_bytes")
        if isinstance(stored_bytes, int) and not isinstance(stored_bytes, bool) and stored_bytes >= 0:
            manifest["summary"]["array_storage_bytes"] = stored_bytes + sum(source.stat().st_size for source, _ in transfers.values())
        manifest["group_ids"] = sorted(components)
        manifest["splits_provisional"] = False
        revision = original.get("edit_revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise DatasetError("Dataset edit revision is malformed")
        manifest["edit_revision"] = revision + 1
        manifest["dataset_update"] = {"metadata_only": True, "excluded_samples": len(samples) - len(active),
                                      "added_samples": added_count,
                                      "split_changes_apply_to": "linked photo/teacher/burst/scene components"}
        if operation_id is not None:
            manifest["dataset_update"].update(operation_id=operation_id, previous_manifest_sha256=original_hash,
                                               request_sha256=request_hash)
            manifest["dataset_update"]["committed_content_sha256"] = _committed_content_digest(manifest)
        eligibility = training_target_eligibility(manifest)
        serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
        published_hash = hashlib.sha256(serialized).hexdigest()
        linked: list[Path] = []
        created_parent = False
        staging: Path | None = None
        try:
            for name, (source, _) in transfers.items():
                target = root / name
                if not target.resolve().is_relative_to(root):
                    raise DatasetError(f"An added array path escapes the dataset directory: {name}")
                if target.exists():
                    if not target.samefile(source):
                        raise DatasetError(f"An unrelated file already occupies an added array path: {name}")
                    continue
                if not target.parent.exists():
                    target.parent.mkdir(parents=True)
                    created_parent = True
                try:
                    os.link(source, target)
                except OSError as exc:
                    raise DatasetError("Cannot attach added photos without copying their arrays. Generate the added photos on the same filesystem as this dataset") from exc
                linked.append(target)
            if hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest() != original_hash:
                raise DatasetError("Dataset manifest changed during updating; reload its photos before applying changes")
            for source, expected_hash in source_hashes.items():
                if hashlib.sha256((source / "dataset.json").read_bytes()).hexdigest() != expected_hash:
                    raise DatasetError("An added dataset changed during updating; retry with its latest manifest")
            descriptor, name = tempfile.mkstemp(prefix=".dataset-update-", suffix=".json", dir=root)
            staging = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), (root / "dataset.json").stat().st_mode & 0o777)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, root / "dataset.json")
            staging = None
            # Best-effort directory sync ensures rename durability where the
            # platform/filesystem supports it. Failure does not undo a commit.
            try:
                descriptor = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
        except BaseException:
            if staging is not None:
                staging.unlink(missing_ok=True)
            # A cancellation can arrive just after atomic replacement returns.
            # Never remove newly linked files if their manifest was committed.
            try:
                committed = hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest() == published_hash
            except OSError:
                committed = False
            if not committed:
                for path in linked:
                    path.unlink(missing_ok=True)
            if created_parent and not committed:
                try:
                    (root / "arrays" / "added").rmdir()
                    (root / "arrays").rmdir()
                except OSError:
                    pass
            raise
        return _update_report(root, manifest, published_hash, already_applied=False)


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
                     "precision_policy": "Original scientific array files preserved byte-for-byte using copy-on-write clones or immutable hard links; no normalization, gamma, resampling or array rewriting"})
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
                        "storage_format": "deduplicated original scientific array files"})
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
