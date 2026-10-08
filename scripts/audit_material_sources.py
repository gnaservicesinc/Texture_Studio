#!/usr/bin/env python3
"""Compare unchanged parent PNGs with Poly Haven's published checksums.

No source images are written, decoded for display, or gamma transformed. The
report is a timestamped snapshot; source folders may be renamed concurrently.
API JSON may be cached, with its actual fetch provenance recorded separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Any
from urllib.parse import quote, unquote, urlparse

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KNOWN_ROLES = "anisotropy_rotation|anisotropy_strength|spec_ior|nor_gl|nor_dx|disp_gl|rough_ao|translucent|diffuse|displacement|roughness|diff|disp|rough|metal|ao|bump|spec|arm"
PARENT = re.compile(r"^(?P<asset>[a-z0-9_]+?)_(?P<role>" + KNOWN_ROLES + r")_(?P<resolution>\d+k)\.png$", re.I)
STANDARD = {"diff": "input", "diffuse": "input", "disp": "height", "disp_gl": "height", "displacement": "height", "nor_gl": "normal", "nor_dx": "normal", "rough": "roughness", "roughness": "roughness"}
REQUIRED = {"input", "height", "normal", "roughness"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat_signature(value: os.stat_result) -> dict[str, int]:
    # Parent-directory renaming does not alter these content/inode identifiers.
    return {"device": value.st_dev, "inode": value.st_ino,
            "size": value.st_size, "mtime_ns": value.st_mtime_ns}


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    asset_id: str
    role: str
    resolution: str
    stat: dict[str, int]


def snapshot_sources(sources: Path) -> tuple[list[SourceSnapshot], list[dict[str, str]], list[str]]:
    snapshots, errors, ignored = [], [], []
    for path in sorted(sources.rglob("*.png")):
        match = PARENT.fullmatch(path.name)
        if not match:
            ignored.append(str(path))
            continue
        try:
            snapshots.append(SourceSnapshot(path, match["asset"].lower(), match["role"].lower(),
                                            match["resolution"].lower(), stat_signature(path.stat())))
        except OSError as error:
            errors.append({"path": str(path), "error": str(error)})
    return snapshots, errors, ignored


def api_metadata(asset_id: str, cache: Path) -> dict[str, Any]:
    url = "https://api.polyhaven.com/files/" + quote(asset_id, safe="")
    path = cache / f"{asset_id}.json"
    sidecar = cache / f"{asset_id}.cache-provenance.json"
    if path.exists():
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Cached API response must be an object")
            sha = hashlib.sha256(raw).hexdigest()
            provenance = {"mode": "cached", "api_url": url, "cache_path": str(path),
                          "payload_sha256": sha, "fetched_utc": None,
                          "cache_file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                          "fetch_time_known": False}
            if sidecar.exists():
                details = json.loads(sidecar.read_text(encoding="utf-8"))
                if details.get("payload_sha256") == sha and details.get("api_url") == url:
                    provenance.update(fetched_utc=details.get("fetched_utc"),
                                      fetch_time_known=bool(details.get("fetched_utc")))
            return {"files": payload, "provenance": provenance}
        except (OSError, ValueError) as error:
            return {"error": f"Invalid existing API cache (left untouched): {error}",
                    "provenance": {"mode": "cached_error", "api_url": url, "cache_path": str(path)}}
    started = utc_now()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{asset_id}-", dir=cache) as temporary:
        downloaded = Path(temporary) / "api.json"
        command = ["curl", "--fail", "--silent", "--show-error", "--location",
                   "--retry", "1", "--retry-max-time", "45", "--connect-timeout", "10", "--max-time", "30",
                   "--user-agent", "TextureStudio/0.1 (local dataset provenance)",
                   "--output", str(downloaded), "--write-out", "%{http_code}", url]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=95)
            if result.returncode:
                raise ValueError(result.stderr.strip() or f"curl exited {result.returncode}")
            raw = downloaded.read_bytes()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("API response must be an object")
            provenance = {"mode": "fresh", "api_url": url, "cache_path": str(path),
                          "payload_sha256": hashlib.sha256(raw).hexdigest(), "fetch_started_utc": started,
                          "fetched_utc": utc_now(), "fetch_time_known": True, "http_status": result.stdout.strip()}
            # Preserve a concurrently written cache rather than overwriting it.
            try:
                with path.open("xb") as stream:
                    stream.write(raw)
            except FileExistsError:
                return api_metadata(asset_id, cache)
            with sidecar.open("x", encoding="utf-8") as stream:
                json.dump(provenance, stream, indent=2)
                stream.write("\n")
            return {"files": payload, "provenance": provenance}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return {"error": str(error), "provenance": {"mode": "fresh_error", "api_url": url,
                                                          "fetch_started_utc": started, "fetch_finished_utc": utc_now()}}


def published_records(value: Any):
    if isinstance(value, dict):
        if all(key in value for key in ("url", "md5", "size")):
            yield value
        else:
            for child in value.values():
                yield from published_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from published_records(child)


def resolve_snapshot(sources: Path, snapshot: SourceSnapshot) -> Path:
    candidates = [snapshot.path, sources / snapshot.asset_id / snapshot.path.name]
    for candidate in candidates:
        if candidate.exists():
            signature = stat_signature(candidate.stat())
            if (signature["device"], signature["inode"]) == (snapshot.stat["device"], snapshot.stat["inode"]):
                return candidate
    # Names in URL-derived filenames identify a moved folder; retain inode
    # identity so duplicate filenames cannot accidentally stand in for it.
    matches = []
    for candidate in sources.rglob(snapshot.path.name):
        try:
            signature = stat_signature(candidate.stat())
        except FileNotFoundError:
            continue
        if (signature["device"], signature["inode"]) == (snapshot.stat["device"], snapshot.stat["inode"]):
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if snapshot.path.exists():
        return snapshot.path  # A replaced inode is reported as source_changed.
    canonical = sources / snapshot.asset_id / snapshot.path.name
    if canonical.exists():
        return canonical  # The folder moved and its file inode was replaced.
    raise FileNotFoundError(f"Original source inode missing or ambiguous after folder rename: {snapshot.path}")


def stream_md5(stream) -> str:
    digest = hashlib.md5()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def normal_advisory(path: Path) -> dict[str, Any]:
    # Optional CPU-only decoding. Concurrency is bounded by --file-workers <=3.
    import cv2
    import numpy as np
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] < 3:
        raise ValueError("Normal is not an RGB PNG")
    codes = raw[::16, ::16, :3][..., ::-1].astype(np.float64) / np.iinfo(raw.dtype).max
    def statistics(values):
        lengths = np.linalg.norm(values * 2 - 1, axis=2)
        return {"rgb_mean": values.mean(axis=(0, 1)).tolist(),
                "unit_length_mean_abs_error": float(np.abs(lengths - 1).mean())}
    decoded = np.where(codes <= .04045, codes / 12.92, ((codes + .055) / 1.055) ** 2.4)
    return {"raw_codes": statistics(codes), "srgb_decoded_codes": statistics(decoded),
            "sampling_stride": 16, "semantic_inference_only": True}


def inspect_source(sources: Path, snapshot: SourceSnapshot, metadata: dict[str, Any], advisory: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"asset_id": snapshot.asset_id, "role": snapshot.role,
                              "resolution": snapshot.resolution, "snapshot_path": str(snapshot.path),
                              "filename": snapshot.path.name, "snapshot_stat": snapshot.stat,
                              "api_provenance": metadata["provenance"], "published_exact_match": False}
    try:
        for attempt in range(3):
            path = resolve_snapshot(sources, snapshot)
            try:
                stream = path.open("rb")
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise
        with stream:
            before = stat_signature(os.fstat(stream.fileno()))
            prefix = stream.read(33)
            if prefix[:8] != PNG_SIGNATURE or prefix[12:16] != b"IHDR" or len(prefix) != 33:
                raise ValueError("Not a PNG with a complete IHDR")
            width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", prefix[16:29])
            stream.seek(0)
            local_md5 = stream_md5(stream)
            after = stat_signature(os.fstat(stream.fileno()))
        current_path = resolve_snapshot(sources, snapshot)
        current = stat_signature(current_path.stat())
        record.update(path=str(current_path), path_renamed=current_path != snapshot.path,
                      local_md5=local_md5, local_bytes=after["size"], before_stat=before, after_stat=after,
                      current_stat=current, width=width, height=height, sample_bits=bits, png_color_type=color,
                      interlace=interlace, source_stable=before == after == current == snapshot.stat)
        if not record["source_stable"]:
            record.update(status="source_changed", error="Source changed between discovery/open/hash/readback; no unchanged-source claim")
            return record
        if "error" in metadata:
            record.update(status="api_error", error=metadata["error"])
            return record
        matching = {json.dumps(item, sort_keys=True): item for item in published_records(metadata["files"])
                    if Path(unquote(urlparse(item["url"]).path)).name == snapshot.path.name}
        # Included blend/gltf records may repeat the same actual published map.
        unique = {(item["url"], item["md5"], item["size"]): item for item in matching.values()}
        if len(unique) != 1:
            record.update(status="published_metadata_missing", error=f"Expected one unique published filename record, found {len(unique)}")
            return record
        expected = next(iter(unique.values()))
        if not isinstance(expected["md5"], str) or not re.fullmatch(r"[0-9a-fA-F]{32}", expected["md5"]):
            record.update(status="published_metadata_invalid", error="Published MD5 is not exactly 32 hex digits; not padded or guessed")
            return record
        if isinstance(expected["size"], bool) or not isinstance(expected["size"], int) or expected["size"] < 0:
            raise ValueError("Published byte size is invalid")
        record.update(published_url=expected["url"], published_md5=expected["md5"], published_bytes=expected["size"],
                      published_exact_match=local_md5 == expected["md5"].lower() and after["size"] == expected["size"])
        record["status"] = "matched" if record["published_exact_match"] else "mismatch"
        if advisory and snapshot.role.startswith("nor_"):
            record["normal_vector_advisory"] = normal_advisory(current_path)
            # A concurrent image change during this optional decode also voids
            # the published-byte claim even though the earlier hash was stable.
            final = stat_signature(current_path.stat())
            if final != current:
                record.update(status="source_changed", source_stable=False, published_exact_match=False,
                              error="Source changed during optional normal advisory")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        record.update(status="source_missing" if isinstance(error, FileNotFoundError) else "error",
                      published_exact_match=False, source_stable=False, error=str(error))
    return record


def audit(sources: Path, cache: Path, api_workers: int = 4, file_workers: int = 2, advisory: bool = False) -> dict[str, Any]:
    started = utc_now()
    snapshots, discovery_errors, ignored = snapshot_sources(sources)
    assets = sorted({item.asset_id for item in snapshots})
    with ThreadPoolExecutor(max_workers=api_workers) as pool:
        metadata = dict(zip(assets, pool.map(lambda asset: api_metadata(asset, cache), assets)))
    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        records = list(pool.map(lambda item: inspect_source(sources, item, metadata[item.asset_id], advisory), snapshots))
    materials = []
    for asset in assets:
        members = [item for item in snapshots if item.asset_id == asset]
        present = {STANDARD.get(item.role) for item in members}
        sets = []
        for resolution in sorted({item.resolution for item in members}):
            roles = {STANDARD.get(item.role) for item in members if item.resolution == resolution}
            dimensions = {record["role"]: [record["width"], record["height"]] for record in records
                          if record["asset_id"] == asset and record["resolution"] == resolution
                          and record["role"] in STANDARD and "width" in record}
            sets.append({"resolution": resolution, "missing_required_maps": sorted(REQUIRED - roles),
                         "complete_standard_map_set": not (REQUIRED - roles),
                         "standard_map_dimensions": dimensions,
                         "dimension_mismatch": len({tuple(value) for value in dimensions.values()}) > 1})
        materials.append({"asset_id": asset, "parent_png_count": len(members),
                          "resolutions": sorted({item.resolution for item in members}),
                          "missing_required_maps": sorted(REQUIRED - present),
                          "complete_standard_map_set": all(item["complete_standard_map_set"] for item in sets),
                          "map_sets": sets})
    ending, ending_errors, _ = snapshot_sources(sources)
    original_ids = {(item.stat["device"], item.stat["inode"]) for item in snapshots}
    ending_ids = {(item.stat["device"], item.stat["inode"]) for item in ending}
    counts = {status: sum(record["status"] == status for record in records)
              for status in sorted({record["status"] for record in records})}
    return {"schema": "texture-studio-published-source-audit-v1", "snapshot_started_utc": started,
            "snapshot_completed_utc": utc_now(), "source_images_modified": False, "sources": str(sources),
            "method": "Full-file MD5 plus byte-count equality to official API; inode/size/mtime are checked before and after hashing. Numeric pixels are never gamma transformed. Published byte equality proves unchanged download bytes, not physical height accuracy.",
            "snapshot_asset_count": len(assets), "snapshot_parent_png_count": len(snapshots),
            "status_counts": counts, "all_snapshotted_parent_pngs_match": bool(records) and all(record["status"] == "matched" for record in records),
            "api_mode_counts": {mode: sum(value["provenance"]["mode"] == mode for value in metadata.values())
                                for mode in sorted({value["provenance"]["mode"] for value in metadata.values()})},
            "api_cache": str(cache), "materials": materials, "files": records,
            "discovery_errors": discovery_errors, "end_discovery_errors": ending_errors, "ignored_png_paths": ignored,
            "source_collection_changed_during_audit": original_ids != ending_ids,
            "parents_added_after_snapshot": [str(item.path) for item in ending if (item.stat["device"], item.stat["inode"]) not in original_ids],
            "parents_missing_after_snapshot": [str(item.path) for item in snapshots if (item.stat["device"], item.stat["inode"]) not in ending_ids],
            "normal_advisory_enabled": advisory, "concurrency": {"api_workers": api_workers, "file_workers": file_workers}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report JSON; existing snapshots are never overwritten")
    parser.add_argument("--api-cache", type=Path, required=True)
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--file-workers", type=int, default=2)
    parser.add_argument("--normal-advisory", action="store_true", help="Optional CPU-only sampled vector checks; never changes pixels")
    args = parser.parse_args()
    if not args.sources.is_dir():
        parser.error("--sources must be an existing directory")
    if not 1 <= args.api_workers <= 6 or not 1 <= args.file_workers <= 3:
        parser.error("API workers must be 1..6 and file workers 1..3")
    if args.output.exists():
        parser.error("--output already exists; choose a new snapshot path")
    report = audit(args.sources.resolve(), args.api_cache.resolve(), args.api_workers, args.file_workers, args.normal_advisory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()), "assets": report["snapshot_asset_count"],
                      "parent_pngs": report["snapshot_parent_png_count"], "status_counts": report["status_counts"],
                      "api_mode_counts": report["api_mode_counts"], "source_collection_changed_during_audit": report["source_collection_changed_during_audit"],
                      "incomplete_materials": [item["asset_id"] for item in report["materials"] if not item["complete_standard_map_set"]]}))
    if not report["all_snapshotted_parent_pngs_match"] or report["discovery_errors"] or report["end_discovery_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
