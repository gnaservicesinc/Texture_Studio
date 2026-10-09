#!/usr/bin/env python3
"""Download original ambientCG photogrammetry PNG map sets at a native size.

Planning is the default; --download obtains the publisher's package. Keep only
its original Color, Displacement, NormalGL and Roughness PNG members, preserving
their exact filenames and bytes. UInt16 displacement is required. This never
resizes images, changes transfer encoding, or edits a prepared dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from urllib.parse import quote
import zipfile

import audit_ambientcg_package as audit
from download_material_resolution import digest_file, png_header, publish_bytes_once

SCHEMA = "ipde-ambientcg-resolution-download-v1"
ROLE_SUFFIXES = {"input": "Color", "height": "Displacement", "normal": "NormalGL", "roughness": "Roughness"}
MAX_MEMBER_BYTES = 256 * 1024 * 1024


def asset_ids(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",")]
    if (not 1 <= len(result) <= 64 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", item) for item in result)
            or len({item.casefold() for item in result}) != len(result)):
        raise ValueError("Use unique official ambientCG asset IDs, separated by commas")
    return result


def public_plan(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def fetch_plan(asset_id: str, resolution: str) -> dict:
    api_url = "https://ambientcg.com/api/v2/full_json?include=downloadData&id=" + quote(asset_id, safe="")
    with tempfile.TemporaryDirectory(prefix="ambientcg-plan-") as directory:
        temporary = Path(directory)
        api_path, license_path = temporary / "api.json", temporary / "license.html"
        api_transport = audit.bounded_download(api_url, api_path, audit.MAX_METADATA_BYTES)
        license_transport = audit.bounded_download(audit.LICENSE_URL, license_path, audit.MAX_METADATA_BYTES)
        api_bytes, license_bytes = api_path.read_bytes(), license_path.read_bytes()
    license_text = " ".join(re.sub(r"<[^>]+>", " ", license_bytes.decode("utf-8")).split())
    if "Creative Commons CC0 1.0 Universal License" not in license_text:
        raise ValueError("Official ambientCG license snapshot does not establish CC0")
    payload = json.loads(api_bytes)
    if not isinstance(payload, dict) or not isinstance(payload.get("foundAssets"), list):
        raise ValueError("Official ambientCG metadata must contain a foundAssets list")
    asset, package = audit.choose_package(payload, asset_id, resolution)
    if asset.get("creationMethod") != "PBRPhotogrammetry":
        raise ValueError(f"{asset_id}: official metadata does not identify Surface Photogrammetry")
    creation = {"id": asset["creationMethod"], "name": asset.get("creationMethodName"),
                "description": asset.get("creationMethodDescription"), "asset_page_label": "Surface Photogrammetry",
                "scope": "This asset only; not inferred for other ambientCG materials"}
    return {"material_id": asset_id.lower(), "asset_id": asset_id, "provider": "ambientCG",
            "asset_url": "https://ambientcg.com/view?id=" + asset_id, "resolution": resolution.lower(),
            "license": "CC0-1.0", "license_url": audit.LICENSE_URL, "creation_method": creation,
            "package": {"filename": package["fileName"], "url": package["downloadLink"], "published_api_bytes": package["size"]},
            "api": {"api_url": api_url, "snapshot_sha256": hashlib.sha256(api_bytes).hexdigest(), "transport": api_transport},
            "license_evidence": {"snapshot_sha256": hashlib.sha256(license_bytes).hexdigest(), "retrieved_utc": license_transport["completed_utc"]},
            "physical_width_meters": None, "physical_height_meters": None,
            "provider_dimensions_raw": {key: asset.get(key) for key in ("dimensionX", "dimensionY", "dimensionZ")},
            "provider_dimension_unit": "Unverified; raw API fields retained without conversion or physical scale inference",
            "published_original_bytes": True, "numeric_values_transformed": False, "source_image_padding": False,
            "source_image_resizing": False, "required_height_sample_bits": 16, "training_split_assigned": False,
            "_api_bytes": api_bytes, "_license_bytes": license_bytes}


def stage_members(archive: Path, directory: Path, asset_id: str, resolution: str) -> dict[str, Path]:
    paths = {role: directory / f"{asset_id}_{resolution}-PNG_{suffix}.png" for role, suffix in ROLE_SUFFIXES.items()}
    with zipfile.ZipFile(archive) as package:
        members = {}
        for role, path in paths.items():
            found = [member for member in package.infolist() if member.filename == path.name]
            if len(found) != 1:
                raise ValueError(f"Missing or duplicate exact original ZIP member: {path.name}")
            member = found[0]
            if (member.is_dir() or member.flag_bits & 1 or not 0 < member.file_size <= MAX_MEMBER_BYTES
                    or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK):
                raise ValueError(f"Invalid bounded original ZIP member: {path.name}")
            members[role] = member
        if sum(member.file_size for member in members.values()) > audit.MAX_ARCHIVE_BYTES:
            raise ValueError("Selected uncompressed PNG members exceed the byte budget")
        directory.mkdir()
        for role, path in paths.items():
            count = 0
            with package.open(members[role]) as source, path.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > members[role].file_size:
                        raise ValueError("ZIP member exceeded its declared byte count")
                    target.write(chunk)
            if count != members[role].file_size:
                raise ValueError("ZIP member differs from its declared byte count")
    return paths


def verify_native_maps(paths: dict[str, Path], resolution: str) -> dict[str, dict]:
    records = {}
    for role, path in paths.items():
        header = png_header(path)
        if max(header["width"], header["height"]) != int(resolution[:-1]) * 1024:
            raise ValueError(f"Unexpected native {resolution} dimensions: {path.name}")
        if role == "height" and header["sample_bits"] != 16:
            raise ValueError(f"Published displacement is {header['sample_bits']}-bit; UInt16 required, no upconversion")
        if role in ("input", "normal") and header["channels"] not in (3, 4):
            raise ValueError(f"Published {role} map must be RGB(A)")
        records[role] = {**digest_file(path), "png_header": header, "actual_header_verified": True,
                         "original_precision_preserved": True, "numeric_values_transformed": False}
    if len({(entry["png_header"]["width"], entry["png_header"]["height"]) for entry in records.values()}) != 1:
        raise ValueError("Paired native PNG dimensions differ")
    return records


def verify_existing(path: Path, expected: dict) -> None:
    if path.is_symlink() or not path.is_file() or digest_file(path) != {key: expected[key] for key in ("bytes", "md5", "sha256")}:
        raise ValueError(f"Existing map differs from the official member; left untouched: {path}")


def publish_identical_metadata(path: Path, payload: bytes) -> None:
    if not publish_bytes_once(path, payload) and (path.is_symlink() or path.read_bytes() != payload):
        raise ValueError(f"Existing content-addressed metadata differs; left untouched: {path}")


def matching_manifest(previous: dict, plan: dict, records: dict) -> bool:
    return (previous.get("material_id") == plan["material_id"] and previous.get("asset_id") == plan["asset_id"]
            and previous.get("provider") == "ambientCG" and previous.get("resolution") == plan["resolution"]
            and {role: item.get("sha256") for role, item in previous.get("downloaded_maps", {}).items()}
            == {role: item["sha256"] for role, item in records.items()})


def download_asset(plan: dict, destination: Path, evidence: Path) -> dict:
    folders = {path.parent for path in destination.glob("*/*.png")
               if re.fullmatch(re.escape(plan["asset_id"]) + r"_\d+K-PNG_(?:Color|Displacement|NormalGL|Roughness)\.png", path.name, re.IGNORECASE)}
    if len(folders) > 1:
        raise ValueError("Asset exists in multiple source folders; avoid another copy")
    directory = next(iter(folders)) if folders else destination / plan["material_id"]
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError(f"Invalid destination material directory: {directory}")
    destination.mkdir(parents=True, exist_ok=True)
    owned = []
    with tempfile.TemporaryDirectory(prefix=".ambientcg-download-", dir=destination) as work:
        temporary = Path(work)
        archive = temporary / plan["package"]["filename"]
        transport = audit.bounded_download(plan["package"]["url"], archive, plan["package"]["published_api_bytes"], archive.name)
        paths = stage_members(archive, temporary / "maps", plan["asset_id"], plan["resolution"].upper())
        records = verify_native_maps(paths, plan["resolution"])
        # Verify the whole archive/member contract before publishing any image.
        staged_audit = audit.compare_package(archive, temporary / "maps", plan["asset_id"], plan["resolution"].upper(),
                                            plan["package"]["published_api_bytes"], transport["headers"].get("x-bz-content-sha1"))
        manifest = directory / f"material-source-{plan['resolution']}.json"
        if manifest.exists() or manifest.is_symlink():
            if manifest.is_symlink() or not matching_manifest(json.loads(manifest.read_text()), plan, records):
                raise ValueError(f"Existing provenance manifest differs; left untouched: {manifest}")
        for role, source in paths.items():
            target = directory / source.name
            if target.exists() or target.is_symlink():
                verify_existing(target, records[role])
        directory.mkdir(exist_ok=True)
        try:
            for role, source in paths.items():
                target = directory / source.name
                try:
                    os.link(source, target)
                    owned.append((target, audit.file_identity(target)))
                    action = "downloaded_verified"
                except FileExistsError:
                    verify_existing(target, records[role])
                    action = "reused_verified"
                records[role].update(path=str(target.resolve()), action=action, filename=target.name)
            verified = audit.compare_package(archive, directory, plan["asset_id"], plan["resolution"].upper(),
                                             plan["package"]["published_api_bytes"], transport["headers"].get("x-bz-content-sha1"))
            if verified["package"]["sha256"] != staged_audit["package"]["sha256"]:
                raise ValueError("Official archive changed during publication")
            asset_evidence = evidence / plan["material_id"]
            api_path = asset_evidence / f"official-metadata-{plan['api']['snapshot_sha256']}.json"
            license_path = asset_evidence / f"license-{plan['license_evidence']['snapshot_sha256']}.html"
            publish_identical_metadata(api_path, plan["_api_bytes"])
            publish_identical_metadata(license_path, plan["_license_bytes"])
            package_report = {"schema": audit.SCHEMA, "provider": "ambientCG", "material_id": plan["material_id"],
                              "asset_id": plan["asset_id"], "asset_url": plan["asset_url"], "completed_utc": audit.now(),
                              "api_provenance": {**plan["api"], "snapshot_path": str(api_path.resolve())},
                              "creation_method": plan["creation_method"], "physical_width_meters": plan["physical_width_meters"],
                              "physical_height_meters": plan["physical_height_meters"],
                              "provider_dimensions_raw": plan.get("provider_dimensions_raw"),
                              "provider_dimension_unit": plan.get("provider_dimension_unit"),
                              "license": {"spdx": "CC0-1.0", "url": audit.LICENSE_URL, **plan["license_evidence"],
                                          "snapshot_path": str(license_path.resolve()), "scope": "Official license applies to downloadable asset files"},
                              **verified, "source_pixels_modified": False, "source_files_modified": False,
                              "original_files_deleted": False, "verification_download_retained": False}
            package_report["package"].update(url=plan["package"]["url"], download_transport=transport,
                                              published_api_bytes=plan["package"]["published_api_bytes"])
            audit_path = asset_evidence / f"{plan['material_id']}-{plan['resolution']}-verified-package.json"
            report_bytes = (json.dumps(package_report, indent=2, allow_nan=False) + "\n").encode()
            if not publish_bytes_once(audit_path, report_bytes):
                old = json.loads(audit_path.read_text()) if not audit_path.is_symlink() else {}
                if (old.get("asset_id") != plan["asset_id"] or old.get("package", {}).get("sha256") != verified["package"]["sha256"]
                        or {entry["source_filename"]: entry["source_sha256"] for entry in old.get("files", [])}
                        != {entry["source_filename"]: entry["source_sha256"] for entry in verified["files"]}):
                    raise ValueError(f"Existing package audit differs; left untouched: {audit_path}")
            first = next(iter(records.values()))["png_header"]
            result = {**public_plan(plan), "downloaded_maps": records, "actual_native_pixel_dimensions": [first["width"], first["height"]],
                      "actual_headers_verified": True, "package_audit_path": str(audit_path.resolve()),
                      "status": "downloaded_or_reused_verified", "completed_utc": audit.now(), "temporary_archive_retained": False}
            payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
            if not publish_bytes_once(manifest, payload):
                if manifest.is_symlink() or not matching_manifest(json.loads(manifest.read_text()), plan, records):
                    raise ValueError(f"Concurrent provenance manifest differs; left untouched: {manifest}")
                result["existing_manifest_reused"] = True
            else:
                result["existing_manifest_reused"] = False
            result["manifest_path"] = str(manifest.resolve())
            return result
        except Exception:
            # Remove only images newly published by this attempt and untouched
            # since publication; never delete preexisting or concurrently edited maps.
            for path, identity in reversed(owned):
                if path.exists() and not path.is_symlink() and audit.file_identity(path) == identity:
                    path.unlink()
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, help="Official IDs, e.g. Wicker013,Snow013,Snow014,Snow015")
    parser.add_argument("--resolution", choices=("1K", "2K", "4K", "8K"), default="1K")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("out/material-training/ambientcg-native-1k"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--download", action="store_true", help="Download native packages and publish all four verified maps")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent materials (1 to 8); default 4")
    parser.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024, help="Maximum total published package bytes")
    args = parser.parse_args(argv)
    materials = asset_ids(args.assets)
    if not 1 <= args.workers <= 8 or not 0 < args.max_bytes <= len(materials) * audit.MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid worker count or total archive budget")
    destination = args.destination or Path(f"/opt/ipde/material-dataset/sources-{args.resolution.lower()}")
    report_path = args.report or args.evidence / "download-report.json"
    report = {"schema": SCHEMA, "resolution": args.resolution.lower(), "destination": str(destination.absolute()),
              "original_sources_modified": False, "active_dataset_index_modified": False, "resize_or_gamma_conversion": False,
              "download_workers": args.workers, "materials": [], "downloaded_materials": [], "errors": []}
    def plan_one(material: str) -> dict:
        try:
            return {"plan": fetch_plan(material, args.resolution)}
        except (OSError, ValueError, subprocess.TimeoutExpired, zipfile.BadZipFile) as error:
            return {"error": {"asset_id": material, "error": str(error)}}
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        planned = list(workers.map(plan_one, materials))
    plans = [entry["plan"] for entry in planned if "plan" in entry]
    report["errors"] = [entry["error"] for entry in planned if "error" in entry]
    report["materials"] = [public_plan(plan) for plan in plans]
    report["total_published_bytes"] = total = sum(plan["package"]["published_api_bytes"] for plan in plans)
    report["within_requested_byte_budget"] = total <= args.max_bytes
    if total > args.max_bytes:
        report["errors"].append({"error": "Packages exceed the requested total byte budget; no maps downloaded"})
    elif args.download:
        ancestor = destination.absolute()
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < total * 3 + 512 * 1024 * 1024:
            report["errors"].append({"error": "Insufficient free disk for bounded temporary packages and original PNGs"})
        else:
            def download_one(plan: dict) -> dict:
                try:
                    return {"result": download_asset(plan, destination, args.evidence)}
                except (OSError, ValueError, subprocess.TimeoutExpired, zipfile.BadZipFile) as error:
                    return {"error": {"asset_id": plan["asset_id"], "error": str(error)}}
            with ThreadPoolExecutor(max_workers=args.workers) as workers:
                results = list(workers.map(download_one, plans))
            report["downloaded_materials"] = [entry["result"] for entry in results if "result" in entry]
            report["errors"].extend(entry["error"] for entry in results if "error" in entry)
    report["status"] = "incomplete" if report["errors"] else ("downloaded_and_verified" if args.download else "planned")
    report["completed_utc"] = audit.now()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + report_path.name + "-", suffix=".report-tmp", dir=report_path.parent)
    temporary_report = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_report, report_path)
    finally:
        temporary_report.unlink(missing_ok=True)
    print(json.dumps({"status": report["status"], "verified_materials": len(report["downloaded_materials"]),
                      "published_package_bytes": total, "errors": report["errors"], "report": str(report_path.resolve())}, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired, zipfile.BadZipFile) as error:
        raise SystemExit(f"error: {error}")
