"""Finalize verified completed labels after a dataset producer has exited.

Recovery changes the manifest and may rename its directory. It never rewrites,
prunes, or normalizes any scientific array, partial payload, or original photo.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .dataset import DatasetError, assign_grouped_splits, load_dataset
from .dataset_review import (_array_records, _publish_new_directory,
    _read_manifest_snapshot, _summary, _validate_manifest_metadata, _warnings)
from .resource_lock import resource_lock


def _intended_output(root: Path, manifest: Mapping[str, Any]) -> Path | None:
    value = manifest.get("generation_output_dir")
    if not isinstance(value, str) or not Path(value).is_absolute():
        return None
    destination = Path(value).expanduser().resolve()
    # Only generation's recorded sibling destination can be promoted. Imported
    # manifests cannot authorize moving data elsewhere or replacing a dataset.
    if destination == root:
        return destination
    if destination.parent == root.parent and root.name.startswith(f".{destination.name}-"):
        return destination
    return None


def _producer_locks(root: Path, destination: Path | None, *, reviewing: bool):
    owner = (destination or root) / ".ipde-generation-owner"
    yield owner, False
    # Legacy CLI producers hold the intended output's lock, while current
    # Python producers also hold the owner lock. A review already has a shared
    # root reader lock, so use another shared lock when root == destination.
    if destination is not None:
        yield destination, reviewing and destination == root
    yield root / "dataset.json", False


def generation_status(directory: Path | str, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if manifest is None:
        root, manifest, _ = _read_manifest_snapshot(root, validate_files=False)
    destination = _intended_output(root, manifest)
    result = {"active": False, "recoverable": False, "status": "complete",
              "reason": "Dataset generation is complete.",
              "intended_output_dir": str(destination) if destination is not None else None}
    if manifest.get("generation_state") != "generating" and not manifest.get("splits_provisional"):
        return result
    try:
        with ExitStack() as locks:
            for path, shared in _producer_locks(root, destination, reviewing=True):
                locks.enter_context(resource_lock(path, shared=shared))
    except RuntimeError:
        result.update(active=True, status="active",
                      reason="A dataset producer or writer still owns this generation. Wait for it to finish.")
    else:
        result.update(recoverable=True, status="interrupted",
                      reason="Generation stopped before finalizing. Completed labels can be verified and recovered without changing array files.")
    return result


def _content_digest(manifest: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(manifest))
    value.get("generation_recovery", {}).pop("committed_content_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=".dataset-recovery-", suffix=".json", dir=root)
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), (root / "dataset.json").stat().st_mode & 0o777)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, root / "dataset.json")
        _sync_directory(root)
    finally:
        staging.unlink(missing_ok=True)


def recover_dataset(directory: Path | str, *, expected_manifest_sha256: str,
                    promote: bool = True, workers: int | None = None) -> dict[str, Any]:
    """Verify completed samples, finalize splits, and safely publish staging.

    The expected snapshot digest is mandatory. A repeated committed request is
    accepted only when its recorded content proof still matches the manifest.
    """
    if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)):
        raise DatasetError("Recovery requires the SHA-256 digest from the loaded dataset snapshot")
    root, initial, _ = _read_manifest_snapshot(directory, validate_files=False)
    destination = _intended_output(root, initial)
    with ExitStack() as locks:
        # Include the CLI parent's intended output lock before touching its
        # staging snapshot. A dead PID alone is never evidence of ownership.
        paths = {path: shared for path, shared in _producer_locks(root, destination, reviewing=False)}
        paths[root] = False
        for path in sorted(paths):
            locks.enter_context(resource_lock(path, shared=False))
        _, original, original_hash = _read_manifest_snapshot(root, validate_files=False)
        if _intended_output(root, original) != destination:
            raise DatasetError("Dataset generation destination changed; reload before recovering")
        prior = original.get("generation_recovery", {})
        committed = (isinstance(prior, dict) and prior.get("previous_manifest_sha256") == expected_manifest_sha256
                     and prior.get("committed_content_sha256") == _content_digest(original)
                     and original.get("generation_state") == "complete" and not original.get("splits_provisional"))
        if original_hash != expected_manifest_sha256 and not committed:
            raise DatasetError("Dataset manifest changed; reload before recovering completed labels")
        if not committed and original.get("generation_state") != "generating" and not original.get("splits_provisional"):
            raise DatasetError("This dataset is already complete and does not require recovery")
        # Decode and checksum every referenced raw/target array, including
        # excluded completed samples. Unreferenced partial files remain intact.
        load_dataset(root, verify=True, workers=workers, snapshot=original)
        if hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest() != original_hash:
            raise DatasetError("Dataset manifest changed during verification; reload before recovering")
        can_promote = bool(promote and destination is not None and destination != root and not destination.exists())
        manifest = copy.deepcopy(original)
        if not committed:
            fraction, seed = manifest.get("validation_fraction", .2), manifest.get("split_seed", 0)
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise DatasetError("Dataset split seed must be an integer")
            groups = assign_grouped_splits(manifest["samples"], .5 if fraction in (0, 1) else fraction, seed)
            if fraction in (0, 1):
                for sample in manifest["samples"]:
                    sample["split"] = "validation" if fraction == 1 else "train"
            source_count = len({sample["source_sha256"] for sample in manifest["samples"]})
            files = {record["path"] for record in _array_records(manifest["samples"])}
            expected_teachers = {value.get("id") for value in manifest.get("teachers", []) if isinstance(value, dict) and isinstance(value.get("id"), str)}
            by_photo: dict[str, set[str]] = {}
            for sample in manifest["samples"]:
                by_photo.setdefault(sample["source_sha256"], set()).add(sample.get("teacher_id", ""))
            missing = sum(len(expected_teachers - teachers) for teachers in by_photo.values())
            manifest.update(generation_state="complete", splits_provisional=False, group_ids=groups,
                            generation_output_dir=str(destination if can_promote else root))
            manifest["summary"] = {**manifest.get("summary", {}), **_summary(manifest["samples"]),
                                   "source_photos": source_count, "processed_sources": source_count}
            manifest["generation_recovery"] = {
                "previous_manifest_sha256": original_hash, "original_generation_state": original.get("generation_state"),
                "original_generation_output_dir": original.get("generation_output_dir"),
                "original_summary": original.get("summary", {}), "staging_directory": str(root),
                "recovered_at_utc": datetime.now(timezone.utc).isoformat(), "completed_samples": len(manifest["samples"]),
                "verified_array_files": len(files), "missing_requested_teacher_pairs": missing,
                "scientific_array_files_unchanged": True, "unreferenced_partial_payloads_retained": True,
                "split_policy": "recorded fraction and seed, grouped by duplicate photo/RGB, explicit scene and burst",
            }
            manifest["warnings"] = _warnings(manifest)
            manifest["warnings"].append("Recovered completed labels after interrupted generation. All partial files and original metadata were retained; missing teacher results can be generated separately.")
            _validate_manifest_metadata(root, manifest, validate_files=False)
            manifest["generation_recovery"]["committed_content_sha256"] = _content_digest(manifest)
            _atomic_manifest(root, manifest)
        final_root, promoted = root, False
        warnings = list(manifest.get("warnings", []))
        if can_promote:
            try:
                _publish_new_directory(root, destination)
            except (OSError, DatasetError) as exc:
                warnings.append(f"Verified recovery completed in its staging directory; publication was unavailable: {exc}")
            else:
                final_root, promoted = destination, True
                _sync_directory(destination.parent)
        elif promote and destination is not None and destination != root:
            warnings.append("The intended output already exists; recovered data remains in staging without replacing it.")
        return {"dataset_path": str(final_root), "promoted": promoted,
                "recovered_samples": len(manifest["samples"]), "generation_state": "complete", "splits_provisional": False,
                "manifest_sha256": hashlib.sha256((final_root / "dataset.json").read_bytes()).hexdigest(),
                "summary": manifest["summary"], "warnings": warnings,
                "generation_recovery": manifest["generation_recovery"], "recovered_committed_request": committed}
