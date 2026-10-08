#!/usr/bin/env python3
"""Plan/download original published Poly Haven 2K PNG map sets, without resize.

The default only plans. --download is explicit; existing changed files are never
overwritten. This does not prepare crops or change the active dataset index.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse
import zlib

ROLES = {"input": ("diffuse", "diff"), "height": ("displacement", "disp"), "normal": ("nor_gl",), "roughness": ("rough", "roughness")}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
METADATA_LIMIT = 2 * 1024 * 1024
FILE_LIMIT = 256 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def asset_id(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
        raise ValueError(f"Use the official normalized Poly Haven asset ID: {value!r}")
    return value


def publish_bytes_once(path: Path, payload: bytes) -> bool:
    """Atomically publish complete metadata without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".metadata-tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def digest_file(path: Path) -> dict[str, Any]:
    before = path.stat()
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while checksumming: {path}")
    return {"bytes": after.st_size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def png_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(33)
    if len(data) != 33 or data[:8] != PNG_SIGNATURE or data[8:16] != b"\0\0\0\rIHDR" or zlib.crc32(data[12:29]) & 0xffffffff != struct.unpack(">I", data[29:33])[0]:
        raise ValueError(f"Missing/invalid PNG IHDR: {path}")
    width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if min(width, height) < 1 or color not in (0, 2, 6) or bits not in (8, 16) or compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ValueError(f"Unsupported photographic/data PNG header: {path}")
    return {"width": width, "height": height, "sample_bits": bits, "sample_dtype": f"uint{bits}", "png_color_type": color, "channels": {0: 1, 2: 3, 6: 4}[color], "interlace": interlace, "header_crc_verified": True}


def dataset_assets(dataset: Path) -> tuple[dict[str, dict], list[dict[str, str]], str]:
    index_path = dataset / "dataset.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    grouped: dict[str, list[dict]] = {}
    for row in index.get("samples", []):
        directory = (dataset / row["path"]).resolve()
        if not directory.is_relative_to(dataset.resolve()):
            raise ValueError("Dataset sample path escapes the dataset")
        metadata = json.loads((directory / "sample.json").read_text())
        if any(metadata.get(key) != row.get(key) for key in ("sample_id", "material_id", "split", "status")):
            raise ValueError(f"Dataset index/sample mismatch: {directory}")
        grouped.setdefault(asset_id(row["material_id"]), []).append(metadata)
    verified, skipped = {}, []
    for material, samples in grouped.items():
        candidates = []
        for metadata in samples:
            sources = [metadata.get("map_metadata", {}).get(role, {}).get("source", {}) for role in ROLES]
            if metadata.get("source_provider") == "Poly Haven" and metadata.get("source_license") == "CC0-1.0" and all(source.get("provider") == "Poly Haven" and source.get("license") == "CC0-1.0" and source.get("published_api_url") == f"https://api.polyhaven.com/files/{material}" and source.get("file_bytes") == source.get("published_bytes") and isinstance(source.get("file_md5"), str) and source["file_md5"].lower() == str(source.get("published_md5", "")).lower() and re.fullmatch(r"[0-9a-fA-F]{32}", source["file_md5"]) for source in sources):
                candidates.append(metadata)
        if not candidates:
            skipped.append({"material_id": material, "reason": "No complete checksum-verified Poly Haven provenance in prepared dataset; other providers handled separately"})
            continue
        representative = candidates[0]
        parents = {}
        for role in ROLES:
            source = representative["map_metadata"][role]["source"]
            parents[role] = {key: source.get(key) for key in ("filename", "resolution_label", "width", "height", "sample_bits", "file_sha256", "published_md5", "published_url")}
        regions = [{"sample_id": sample["sample_id"], "split": sample["split"], "crop_rectangle_top_left_xywh": sample.get("crop_rectangle_top_left_xywh"), "source_pixel_dimensions": sample.get("source_pixel_dimensions"), "source_input_sha256": sample.get("map_metadata", {}).get("input", {}).get("source", {}).get("file_sha256")} for sample in samples]
        verified[material] = {"verified_existing_parent_maps": parents, "active_dataset_source_regions": regions}
    return verified, skipped, hashlib.sha256(index_bytes).hexdigest()


def fetch_metadata(material: str, cache: Path | None = None) -> tuple[dict, dict]:
    url = f"https://api.polyhaven.com/files/{asset_id(material)}"
    started = utc_now()
    result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--proto", "=https", "--connect-timeout", "10", "--max-time", "30", "--max-filesize", str(METADATA_LIMIT), "--user-agent", "TextureStudio/0.1 (local material resolution plan)", url], capture_output=True, timeout=35)
    if result.returncode or len(result.stdout) > METADATA_LIMIT:
        raise ValueError(f"Official API fetch failed for {material}: {result.stderr.decode(errors='replace')[:300]}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Official API response is not an object: {material}")
    checksum = hashlib.sha256(result.stdout).hexdigest()
    provenance = {"api_url": url, "mode": "fresh", "fetch_started_utc": started, "fetched_utc": utc_now(), "payload_sha256": checksum, "payload_bytes": len(result.stdout)}
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{material}-{checksum}.json"
        if not publish_bytes_once(path, result.stdout):
            if path.read_bytes() != result.stdout:
                raise ValueError("Content-addressed API cache differs from official payload")
        provenance["cache_path"] = str(path.resolve())
    return payload, provenance


def selected_map(material: str, role: str, payload: dict, resolution: str) -> dict:
    aliases = ROLES[role]
    matches = [(name, item) for name, item in payload.items() if name.casefold() in aliases]
    if len(matches) != 1:
        raise ValueError(f"Need one official {role} map for {material}, got {len(matches)}")
    provider_role, variants = matches[0]
    record = variants.get(resolution, {}).get("png")
    if not isinstance(record, dict):
        raise ValueError(f"Missing published {resolution} PNG for {material}/{role}; JPG/EXR is not substituted")
    url = record.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    prefix = f"/file/ph-assets/Textures/png/{resolution}/{material}/"
    if not parsed or parsed.scheme != "https" or parsed.hostname != "dl.polyhaven.org" or parsed.port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.path.startswith(prefix):
        raise ValueError(f"Unsafe/unexpected official download URL: {url!r}")
    filename = unquote(parsed.path[len(prefix):])
    if not re.fullmatch(re.escape(material) + r"_[a-z0-9_]+_" + re.escape(resolution) + r"\.png", filename):
        raise ValueError(f"Unsafe/nonstandard map filename: {filename!r}")
    size, md5 = record.get("size"), record.get("md5")
    if type(size) is not int or not 1 <= size <= FILE_LIMIT or not isinstance(md5, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", md5):
        raise ValueError(f"Missing valid bounded byte count / full MD5: {material}/{role}")
    return {"role": role, "provider_map_name": provider_role, "filename": filename, "url": url, "published_bytes": size, "published_md5": md5.lower(), "format": "PNG", "actual_header_verified": False, "required_data_bits": 16 if role == "height" else None}


def make_plan(material: str, payload: dict, provenance: dict, references: dict, resolution: str) -> dict:
    maps = {role: selected_map(material, role, payload, resolution) for role in ROLES}
    return {"material_id": material, "provider": "Poly Haven", "license": "CC0-1.0", "license_url": "https://polyhaven.com/license", "asset_url": f"https://polyhaven.com/a/{material}", "resolution": resolution, "published_original_bytes": True, "resized_from_existing_4k": False, "distinct_camera_exposure_claimed": False, "api": provenance, "maps": maps, "published_bytes": sum(item["published_bytes"] for item in maps.values()), "existing_dataset_references": references, "cross_resolution_registration_verified": False, "training_split_assigned": False, "leakage_policy": "Same material may represent the same photographed surface at another published resolution. Register/verify geographic footprints before mixing with existing train/validation regions; resolution difference is not an independent observation."}


def verify_map(path: Path, entry: dict, resolution: str) -> dict:
    if path.is_symlink():
        raise ValueError(f"Existing map symlink is not adopted: {path}")
    checksums = digest_file(path)
    if checksums["bytes"] != entry["published_bytes"] or checksums["md5"] != entry["published_md5"]:
        raise ValueError(f"Existing/downloaded map differs from published bytes; left untouched: {path}")
    header = png_header(path)
    if max(header["width"], header["height"]) != int(resolution[:-1]) * 1024:
        raise ValueError(f"Published map has unexpected native {resolution} dimensions: {path}, {header}")
    if entry["role"] == "height" and header["sample_bits"] != 16:
        raise ValueError(f"Published height map is {header['sample_bits']}-bit; UInt16 required, no upconversion: {path}")
    if entry["role"] == "normal" and header["channels"] not in (3, 4):
        raise ValueError(f"Published normal map is not RGB(A): {path}")
    note = f"Published original is UInt{header['sample_bits']}; retained without upconversion or invented precision"
    return dict(checksums, png_header=header, actual_header_verified=True, original_precision_preserved=True, source_precision_note=note, numeric_values_transformed=False)


def check_pairs(records: dict[str, dict]) -> list[int]:
    dimensions = {(item["png_header"]["width"], item["png_header"]["height"]) for item in records.values()}
    if len(records) != len(ROLES) or len(dimensions) != 1:
        raise ValueError("Published diffuse/height/normal/roughness native pixel dimensions differ")
    return list(next(iter(dimensions)))


def download_asset(plan: dict, destination: Path) -> dict:
    directory = destination / plan["material_id"]
    if directory.is_symlink():
        raise ValueError(f"Destination asset directory is a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    records, pending = {}, []
    try:
        for role, entry in plan["maps"].items():
            target = directory / entry["filename"]
            if target.exists() or target.is_symlink():
                records[role] = dict(verify_map(target, entry, plan["resolution"]), path=str(target.resolve()), action="reused_verified")
                continue
            descriptor, temporary_name = tempfile.mkstemp(prefix="." + entry["filename"] + "-", suffix=".download", dir=directory)
            os.close(descriptor)
            temporary = Path(temporary_name)
            pending.append((temporary, target, role, entry))
            result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--proto", "=https", "--connect-timeout", "10", "--max-time", "120", "--max-filesize", str(entry["published_bytes"]), "--user-agent", "TextureStudio/0.1 (original published material maps)", "--output", str(temporary), entry["url"]], capture_output=True, timeout=125)
            if result.returncode:
                raise ValueError(f"Published map download failed ({role}): {result.stderr.decode(errors='replace')[:300]}")
            records[role] = dict(verify_map(temporary, entry, plan["resolution"]), path=str(target.absolute()), action="downloaded_verified")
        dimensions = check_pairs(records)
        # Stage the complete coherent set first, then publish without replacing
        # any existing path. Hard-link creation provides atomic no-clobber.
        for temporary, target, role, entry in pending:
            try:
                os.link(temporary, target)
            except FileExistsError:
                records[role] = dict(verify_map(target, entry, plan["resolution"]), path=str(target.resolve()), action="concurrent_verified_reuse")
        result = dict(plan, downloaded_maps=records, actual_native_pixel_dimensions=dimensions, actual_headers_verified=True, status="downloaded_or_reused_verified", completed_utc=utc_now())
        manifest = directory / "material-source.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous.get("material_id") != plan["material_id"] or previous.get("resolution") != plan["resolution"] or {role: item.get("sha256") for role, item in previous.get("downloaded_maps", {}).items()} != {role: item["sha256"] for role, item in records.items()}:
                raise ValueError(f"Existing provenance manifest differs; left untouched: {manifest}")
            result["existing_manifest_reused"] = True
        else:
            if not publish_bytes_once(manifest, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()):
                raise FileExistsError(f"Concurrent provenance manifest appeared; files retained for verified reuse: {manifest}")
            result["existing_manifest_reused"] = False
        result["manifest_path"] = str(manifest.resolve())
        return result
    finally:
        for temporary, _, _, _ in pending:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--materials", action="append", help="Repeat this flag or provide comma-separated official asset IDs")
    parser.add_argument("--resolution", choices=("2k",), default="2k")
    parser.add_argument("--destination", type=Path, default=Path("/opt/ipde/material-dataset/sources-2k"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-cache", type=Path)
    parser.add_argument("--max-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--api-workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Default: inspect official metadata; download no PNGs")
    mode.add_argument("--download", action="store_true", help="Explicitly download/reuse the planned published originals")
    args = parser.parse_args(argv)
    if args.report.exists() or not 1 <= args.api_workers <= 4 or args.max_bytes < 1 or (not args.dataset and not args.materials):
        parser.error("Choose a new report path, positive byte budget, 1–4 API workers and dataset or explicit materials")
    references, skipped, index_sha = dataset_assets(args.dataset) if args.dataset else ({}, [], None)
    requested = [asset_id(value) for group in args.materials or [] for value in group.split(",")]
    if len(requested) != len(set(requested)):
        parser.error("Duplicate asset IDs")
    materials = sorted(requested or references)
    if args.dataset and any(material not in references for material in materials):
        parser.error("Requested dataset material lacks complete verified Poly Haven provenance")
    if not materials:
        parser.error("No eligible Poly Haven materials")
    report = {"schema": "texture-studio-published-resolution-v1", "started_utc": utc_now(), "mode": "download" if args.download else "plan", "resolution": args.resolution, "destination": str(args.destination.absolute()), "original_sources_modified": False, "active_dataset_index_modified": False, "resize_or_gamma_conversion": False, "dataset_index_sha256_before": index_sha, "skipped_other_or_unverified_providers": skipped, "maximum_total_published_bytes": args.max_bytes, "materials": [], "errors": []}
    def fetch(material: str) -> dict:
        try:
            payload, provenance = fetch_metadata(material, args.api_cache)
            return make_plan(material, payload, provenance, references.get(material, {}), args.resolution)
        except Exception as error:
            return {"material_id": material, "error": str(error)}
    with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        for item in pool.map(fetch, materials):
            (report["errors"] if "error" in item else report["materials"]).append(item)
    report["planned_material_count"] = len(report["materials"])
    report["planned_png_count"] = len(report["materials"]) * len(ROLES)
    report["total_published_bytes"] = sum(item["published_bytes"] for item in report["materials"])
    report["within_requested_byte_budget"] = report["total_published_bytes"] <= args.max_bytes
    report["status"] = "plan_complete" if not report["errors"] else "incomplete"
    if args.download and not report["errors"]:
        try:
            if not report["within_requested_byte_budget"]:
                raise ValueError("Published byte total exceeds explicit disk/download budget; no PNG downloaded")
            missing_bytes = 0
            for item in report["materials"]:
                for entry in item["maps"].values():
                    path = args.destination / item["material_id"] / entry["filename"]
                    if path.exists() or path.is_symlink():
                        verify_map(path, entry, args.resolution)
                    else:
                        missing_bytes += entry["published_bytes"]
            ancestor = args.destination.absolute()
            while not ancestor.exists():
                ancestor = ancestor.parent
            report["incremental_download_bytes"] = missing_bytes
            report["disk_free_before_bytes"] = shutil.disk_usage(ancestor).free
            if report["disk_free_before_bytes"] < missing_bytes:
                raise ValueError("Insufficient free disk for planned original map bytes; no PNG downloaded")
            report["downloaded_materials"] = []
            for item in report["materials"]:
                print(f"Downloading/verifying published {args.resolution}: {item['material_id']}", flush=True)
                report["downloaded_materials"].append(download_asset(item, args.destination))
            report["status"] = "download_complete_verified"
        except Exception as error:
            report["errors"].append({"error": str(error)})
            report["status"] = "incomplete"
    report["dataset_index_sha256_after"] = hashlib.sha256((args.dataset / "dataset.json").read_bytes()).hexdigest() if args.dataset else None
    report["dataset_index_unchanged"] = report["dataset_index_sha256_before"] == report["dataset_index_sha256_after"]
    if not report["dataset_index_unchanged"]:
        report["errors"].append({"error": "Dataset index changed concurrently; planned source-region snapshot is stale"})
        report["status"] = "incomplete"
    report["finished_utc"] = utc_now()
    if not publish_bytes_once(args.report, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()):
        raise FileExistsError(f"Report appeared concurrently and was left untouched: {args.report}")
    print(json.dumps({"status": report["status"], "materials": report["planned_material_count"], "pngs": report["planned_png_count"], "published_bytes": report["total_published_bytes"], "report": str(args.report), "errors": report["errors"]}), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
