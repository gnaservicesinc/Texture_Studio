"""Explicit cleanup of owned generated data, keeping source photos and models."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from .formats import sha256_file
from .resource_lock import resource_lock


def _owned(path: Path | str, workspace: Path | str, category: str) -> tuple[Path, Path]:
    original = Path(path).expanduser().absolute()
    root = Path(workspace).expanduser().resolve(strict=True)
    owner = root / category
    resolved = original.resolve(strict=True)
    if not root.is_dir() or owner.is_symlink() or not owner.is_dir():
        raise ValueError("Cleanup requires an existing project workspace")
    if original.is_symlink() or original != resolved or resolved.parent != owner:
        raise ValueError(f"Cleanup is limited to generated folders owned by this workspace's {category} directory")
    if not resolved.is_dir():
        raise ValueError("Cleanup target must be a generated directory")
    return root, resolved


def _files(directory: Path) -> list[Path]:
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Cleanup refuses folders containing symbolic links")
    if any(not path.is_file() and not path.is_dir() for path in entries):
        raise ValueError("Cleanup refuses folders containing special files")
    return [path for path in entries if path.is_file()]


def _contains_source(value, directory: Path) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_path" and isinstance(item, str) and not item.startswith("external-dataset:"):
                candidate = Path(item).expanduser()
                if candidate.is_absolute() and candidate.resolve().is_relative_to(directory):
                    return True
                if not candidate.is_absolute() and (directory / candidate).is_file():
                    return True
            elif _contains_source(item, directory):
                return True
    elif isinstance(value, list):
        return any(_contains_source(item, directory) for item in value)
    return False


def _reject_dependencies(root: Path, directory: Path) -> None:
    from .dataset_review import _array_records
    for other in (root / "datasets").glob("*/dataset.json"):
        if other.parent == directory:
            continue
        data = json.loads(other.read_text(encoding="utf-8"))
        for record in _array_records(data):
            if (other.parent / record["path"]).resolve().is_relative_to(directory):
                raise ValueError(f"Another dataset depends on these array files: {other.parent.name}")


def archive_dataset(dataset: Path | str, workspace: Path | str) -> dict:
    root, directory = _owned(dataset, workspace, "datasets")
    with resource_lock(directory):
        _files(directory)
        metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema") != "ipde-depth-dataset-v1":
            raise ValueError("Archive requires a generated IPDE dataset")
        _reject_dependencies(root, directory)
        archive = root / "archived"
        if archive.is_symlink():
            raise ValueError("Archive destination cannot be a symbolic link")
        archive.mkdir(exist_ok=True)
        destination = archive / f"dataset-{directory.name}-{uuid.uuid4().hex}"
        directory.rename(destination)
    return {"archived_dataset": str(destination), "source_dataset": str(directory)}


def cleanup_dataset(dataset: Path | str, workspace: Path | str, *, confirm: bool = False) -> dict:
    if not confirm:
        raise ValueError("Dataset cleanup requires explicit confirmation (--confirm)")
    root, directory = _owned(dataset, workspace, "datasets")
    with resource_lock(directory):
        files = _files(directory)
        manifest = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != "ipde-depth-dataset-v1":
            raise ValueError("Cleanup requires a generated IPDE dataset")
        if _contains_source(manifest, directory):
            raise ValueError("Dataset contains original source photographs; move them out before cleanup")
        if any(path.suffix.lower() in {".pth", ".pt", ".safetensors", ".onnx", ".heic", ".heif", ".dng", ".raw"} for path in files):
            raise ValueError("Dataset contains source photos or model files; move them out before cleanup")
        # Normal prepared sets own independent clone/hardlink directory entries.
        # Older or hand-edited manifests with external array paths must not be
        # left dangling by cleanup. Provenance-only paths do not need their source.
        _reject_dependencies(root, directory)
        byte_count = sum(path.stat().st_size for path in files)
        shutil.rmtree(directory)
    return {"cleaned_dataset": str(directory), "removed_files": len(files), "removed_logical_bytes": byte_count,
            "note": "Original source photographs and trained models were retained. Shared array blocks may remain in other datasets."}


def cleanup_run(checkpoint: Path | str, workspace: Path | str, *, confirm: bool = False) -> dict:
    if not confirm:
        raise ValueError("Run cleanup requires explicit confirmation (--confirm)")
    original = Path(checkpoint).expanduser().absolute()
    _, directory = _owned(original.parent, workspace, "runs")
    if original.is_symlink() or original != original.resolve(strict=True) or original.suffix != ".pth":
        raise ValueError("Cleanup requires a regular owned RAFT checkpoint")
    with resource_lock(directory):
        files = _files(directory)
        report = original.with_suffix(original.suffix + ".json")
        metadata = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema") not in {"ipde-raft-training-report-v1", "ipde-raft-training-report-v2", "ipde-display-training-report-v1"} or metadata.get("checkpoint_sha256") != sha256_file(original):
            raise ValueError("Checkpoint provenance/hash verification failed; no files were removed")
        if _contains_source(metadata, directory) or any(path.name == "dataset.json" or path.suffix.lower() in {".heic", ".heif", ".dng", ".raw"} for path in files):
            raise ValueError("Run contains source data; move it out before cleanup")
        models = {path for path in files if path.suffix.lower() in {".pth", ".pt", ".safetensors", ".onnx"}}
        keep = models | {report} | {path.with_suffix(path.suffix + ".json") for path in models}
        keep |= {path for path in files if path.name == "model.json"}
        remove = [path for path in files if path not in keep]
        byte_count = sum(path.stat().st_size for path in remove)
        for path in remove:
            path.unlink()
        for path in sorted((path for path in directory.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
    return {"checkpoint_path": str(original), "kept_files": [str(path.relative_to(directory)) for path in files if path in keep],
            "removed_files": len(remove), "removed_logical_bytes": byte_count}
