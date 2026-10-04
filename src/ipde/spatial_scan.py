"""Find structurally valid Apple spatial captures without following links.

Container metadata is evidence of calibration, not a cryptographic attestation:
this module cannot establish whether pixels were synthesized or metadata forged.
No decoded samples are normalized, rotated, resized, or otherwise modified.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .extractor import Discovery, discover_file
from .concurrency import memory_limited_workers, ordered_map, resolve_workers


class SpatialScanError(ValueError):
    """A capture cannot safely be used as calibrated stereo training input."""


def capture_metadata(discovery: Discovery) -> dict[str, Any]:
    """Read capture identity from EXIF; retain original date/time and UTC offset."""
    result: dict[str, Any] = {}
    aliases = {"make": "camera_make", "model": "camera_model", "datetimeoriginal": "captured_at",
               "datetimedigitized": "digitized_at", "offsettimeoriginal": "utc_offset",
               "lensmodel": "lens_model", "software": "software", "subsectimeoriginal": "subsecond"}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = aliases.get(str(key).replace(" ", "").lower())
                if name and isinstance(item, (str, int, float)) and str(item).strip():
                    result.setdefault(name, str(item).strip())
                visit(item)
            if str(value.get("type", "")).lower() != "exif":
                return
            raw = value.get("data")
            if isinstance(raw, Mapping) and raw.get("encoding") == "base64":
                try:
                    raw = base64.b64decode(raw["data"], validate=True)
                except (ValueError, KeyError):
                    return
            if not isinstance(raw, bytes):
                return
            # HEIF metadata items begin with a four-byte big-endian offset to
            # the TIFF header; the item may also retain an Exif\0\0 signature.
            if raw[4:8] in (b"II*\0", b"MM\0*") or raw[4:10] == b"Exif\0\0":
                raw = raw[4:]
            elif len(raw) >= 8:
                offset = int.from_bytes(raw[:4], "big") + 4
                if raw[offset:offset + 4] in (b"II*\0", b"MM\0*"):
                    raw = raw[offset:]
            try:
                from PIL import Image
                exif = Image.Exif()
                exif.load(raw)
                tags = dict(exif)
                if 34665 in tags:
                    tags.update(exif.get_ifd(34665))
                for tag, name in {271: "camera_make", 272: "camera_model", 36867: "captured_at",
                                  36868: "digitized_at", 36881: "utc_offset", 42036: "lens_model",
                                  305: "software", 37521: "subsecond"}.items():
                    if tag in tags and isinstance(tags[tag], (str, int, float)):
                        result.setdefault(name, str(tags[tag]).strip())
            except (OSError, ValueError, TypeError, KeyError, SyntaxError):
                # Malformed optional EXIF is never allowed to alter stereo arrays.
                result.setdefault("metadata_warning", "Some EXIF capture metadata could not be parsed")
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(discovery.top_level_images)
    return result


def validate_spatial_discovery(discovery: Discovery, *, require_apple_camera: bool = False) -> dict[str, Any]:
    """Require distinct calibrated, rectified cameras and valid decoded RGB grids."""
    spatial = discovery.spatial_photo
    if not spatial:
        raise SpatialScanError(f"{discovery.source.name} has no calibrated Apple spatial stereo pair")
    if not spatial.get("raft_stereo_ready"):
        notes = spatial.get("raft_stereo_validation_notes") or ["calibration is not rectified"]
        raise SpatialScanError("Stereo calibration is unsuitable for RAFT: " + "; ".join(map(str, notes)))
    indices = (spatial.get("left_image_index"), spatial.get("right_image_index"))
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices) or indices[0] == indices[1]:
        raise SpatialScanError("Stereo cameras must refer to distinct container images")
    views = {asset.semantic_name: asset for asset in discovery.assets if asset.kind == "spatial_view"}
    if not {"spatial_left", "spatial_right"}.issubset(views):
        raise SpatialScanError("Capture lacks decoded left/right spatial views")
    shapes = []
    for role in ("left", "right"):
        asset = views[f"spatial_{role}"]
        if asset.parent_image_index != spatial[f"{role}_image_index"]:
            raise SpatialScanError(f"Decoded {role} view does not belong to its calibrated container image")
        array = np.asarray(asset.array)
        if array.ndim != 3 or array.shape[2] not in (3, 4) or not array.size or array.dtype.kind not in "uf":
            raise SpatialScanError(f"The {role} view must contain nonempty RGB samples")
        if not np.isfinite(array).all():
            raise SpatialScanError(f"The {role} view contains nonfinite RGB samples")
        camera = spatial.get(f"{role}_camera", {})
        if tuple(array.shape[:2]) != (camera.get("height"), camera.get("width")):
            raise SpatialScanError(f"Decoded {role} grid differs from camera calibration")
        try:
            focal = float(camera["focal_length_x_pixels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SpatialScanError(f"The {role} camera has no valid focal length") from exc
        if not np.isfinite(focal) or focal <= 0:
            raise SpatialScanError(f"The {role} camera focal length must be positive and finite")
        shapes.append(array.shape[:2])
    if shapes[0] != shapes[1]:
        raise SpatialScanError("Left/right views must have exactly matching grids")
    if np.array_equal(views["spatial_left"].array, views["spatial_right"].array):
        raise SpatialScanError("Left/right views are identical; the pair contains no independent stereo evidence")
    try:
        baseline = float(spatial["baseline_meters"])
        offset = float(spatial["principal_point_delta_x_pixels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialScanError("Incomplete metric stereo calibration") from exc
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(offset):
        raise SpatialScanError("Invalid metric stereo baseline or principal-point offset")
    metadata = capture_metadata(discovery)
    maker = metadata.get("camera_make", "").strip().lower()
    if maker and maker != "apple":
        raise SpatialScanError(f"Camera maker {metadata['camera_make']!r} is not an Apple spatial capture")
    if require_apple_camera and not maker:
        raise SpatialScanError("No Apple camera identity was found in capture metadata")
    return metadata


def scan_spatial_directory(
    directory: Path | str, *, recursive: bool = True, require_apple_camera: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Discover supported captures recursively; skip every symlink and bad file."""
    requested_root = Path(directory).expanduser().absolute()
    if requested_root.is_symlink():
        raise SpatialScanError("The scan directory must not be a symbolic link")
    # macOS commonly exposes /var and /tmp as system directory aliases. Resolve
    # the explicitly selected parent once; never follow links encountered below.
    root = requested_root.resolve()
    if not root.is_dir():
        raise SpatialScanError(f"Scan directory does not exist: {root}")
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending = [root]
    candidates = 0
    visited_directories = 0
    worker_count = resolve_workers(workers)

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    emit({"event": "scan_started", "directory": str(root), "recursive": recursive})
    paths: list[Path] = []
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name.casefold())
        except OSError as exc:
            if folder == root:
                raise SpatialScanError(f"Could not read scan directory {root}: {exc}") from exc
            record = {"source_path": str(folder), "reason": str(exc), "kind": "directory_error"}
            skipped.append(record)
            emit({"event": "photo_skipped", **record})
            continue
        visited_directories += 1
        subdirectories = []
        for entry in ordered:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    record = {"source_path": str(path), "reason": "Symbolic link excluded", "kind": "link"}
                    skipped.append(record)
                    emit({"event": "photo_skipped", **record})
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if recursive:
                        subdirectories.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False) or path.suffix.lower() not in {".heic", ".heif", ".hif"}:
                    continue
                paths.append(path)
            except Exception as exc:
                record = {"source_path": str(path), "reason": str(exc), "kind": "invalid_capture"}
                skipped.append(record)
                emit({"event": "photo_skipped", **record})
        pending.extend(reversed(subdirectories))
    # Decoding returns only small metadata; pixel arrays are released inside
    # each worker. Bound concurrent native decoders as well as queued tasks.
    largest = max((path.stat().st_size for path in paths if path.is_file()), default=1)
    decoding_workers = memory_limited_workers(worker_count, max(128 * 1024**2, largest * 48))

    def admitted_paths():
        nonlocal candidates
        for path in paths:
            candidates += 1
            emit({"event": "scan_photo_started", "source_path": str(path),
                  "candidate_count": candidates, "accepted_count": len(accepted)})
            yield path

    def decode(path: Path):
        try:
            # Recheck links after directory enumeration before decoding.
            if path.is_symlink():
                return False, {"source_path": str(path), "reason": "Symbolic link excluded", "kind": "link"}
            discovery = discover_file(path)
            metadata = validate_spatial_discovery(discovery, require_apple_camera=require_apple_camera)
            return True, {"source_path": str(path), "source_sha256": discovery.source_sha256,
                          "source_bytes": discovery.source_size, "photo_metadata": metadata,
                          "calibration": discovery.spatial_photo}
        except Exception as exc:
            return False, {"source_path": str(path), "reason": str(exc), "kind": "invalid_capture"}

    for valid, record in ordered_map(decode, admitted_paths(), workers=decoding_workers):
        if valid:
            accepted.append(record)
            emit({"event": "spatial_photo_found", **record, "accepted_count": len(accepted)})
        else:
            skipped.append(record)
            emit({"event": "photo_skipped", **record})
    return {"directory": str(root), "recursive": recursive, "accepted": accepted, "skipped": skipped,
            "file_workers": worker_count, "decoding_workers": decoding_workers,
            "summary": {"accepted": len(accepted), "skipped": len(skipped),
                        "candidates": candidates, "visited_directories": visited_directories},
            "authenticity_note": "Validated Apple camera identity, container structure, decoded grids and calibration; forged metadata or AI-generated pixels cannot be ruled out."}
