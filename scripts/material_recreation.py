#!/usr/bin/env python3
"""Save source/crop provenance and exact recoverable source-note data."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qs, urlparse

SCHEMA = "ipde-material-recreation-v1"
PACKAGE_AUDIT_SCHEMA = "ipde-material-package-audit-v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Dataset paths must be relative and contained")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError("Dataset path leaves its directory")
    return result


def source_note_entry(data: bytes, checksum: str, material: str, originals: dict,
                      source_paths: dict[str, str]) -> dict:
    """Reference a recorded PNG only after comparing the complete current bytes."""
    if sha256(data) != checksum:
        raise ValueError("Changed source note payload")
    if data.startswith(PNG_SIGNATURE):
        for key, original in sorted(originals.items()):
            source = original["source"]
            if (original["material_id"] != material or not source["filename"].lower().endswith(".png")
                    or source["file_sha256"] != checksum or source["file_bytes"] != len(data)):
                continue
            parent = source_paths.get(key)
            if not parent or not Path(parent).is_file():
                continue  # A missing original cannot establish a fresh byte match.
            parent_data = Path(parent).read_bytes()
            if len(parent_data) != len(data) or sha256(parent_data) != checksum or parent_data != data:
                raise ValueError(f"Changed parent for duplicate source note: {key}")
            return {"sha256": checksum, "file_bytes": len(data),
                    "parent_reference": {"source_key": key, "sha256": checksum, "file_bytes": len(data)},
                    "reference_basis": "Exact complete PNG note and recorded current parent bytes matched; original note filename retained"}
    return {"sha256": checksum, "file_bytes": len(data), "data_base64": base64.b64encode(data).decode(),
            "contains_png_image_payload": data.startswith(PNG_SIGNATURE)}


def validated_source_notes(recipe: dict) -> dict:
    """Validate new references and legacy embedded notes without network access."""
    if recipe.get("schema") != SCHEMA:
        raise ValueError("Unsupported recreation recipe")
    notes, sources = recipe.get("source_notes", {}), recipe.get("sources", {})
    if not isinstance(notes, dict) or not isinstance(sources, dict):
        raise ValueError("Source notes and parents must be dictionaries")
    result = {}
    for key, note in notes.items():
        if (not isinstance(key, str) or Path(key).is_absolute() or len(Path(key).parts) != 2
                or str(Path(key)) != key or ".." in Path(key).parts or "\0" in key or not isinstance(note, dict)
                or not isinstance(note.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", note["sha256"])):
            raise ValueError("Malformed source note identity or checksum")
        embedded, referenced = "data_base64" in note, "parent_reference" in note
        if embedded == referenced:
            raise ValueError("Source note needs exactly one embedded payload or parent reference")
        if referenced:
            reference = note["parent_reference"]
            if not isinstance(reference, dict) or set(reference) != {"source_key", "sha256", "file_bytes"}:
                raise ValueError("Malformed source note parent reference")
            parent_key = reference["source_key"]
            if (not isinstance(parent_key, str) or parent_key not in sources
                    or Path(parent_key).is_absolute() or len(Path(parent_key).parts) != 2
                    or str(Path(parent_key)) != parent_key or ".." in Path(parent_key).parts or "\0" in parent_key
                    or Path(parent_key).parts[:1] != Path(key).parts[:1]):
                raise ValueError("Unknown or different-material source note parent")
            parent_entry = sources[parent_key]
            parent = parent_entry.get("source", {}) if isinstance(parent_entry, dict) else {}
            size = reference["file_bytes"]
            if (not isinstance(parent, dict) or type(size) is not int or size <= 0
                    or type(note.get("file_bytes")) is not int or note["file_bytes"] != size
                    or reference["sha256"] != note["sha256"] or parent.get("file_sha256") != note["sha256"]
                    or type(parent.get("file_bytes")) is not int or parent["file_bytes"] != size
                    or not isinstance(parent.get("filename"), str) or not parent["filename"].lower().endswith(".png")
                    or Path(parent_key).parts[-1] != parent.get("filename")):
                raise ValueError("Source note reference differs from its recorded PNG parent")
            result[key] = {"sha256": note["sha256"], "file_bytes": size, "source_key": parent_key}
        else:
            try:
                data = base64.b64decode(note["data_base64"], validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid embedded source note payload") from error
            if sha256(data) != note["sha256"] or ("file_bytes" in note and (type(note["file_bytes"]) is not int or note["file_bytes"] != len(data))):
                raise ValueError("Embedded source note checksum or byte count differs")
            result[key] = {"sha256": note["sha256"], "file_bytes": len(data), "data": data}
    return result


def restore_source_notes(recipe: dict, sources: Path) -> dict:
    """Restore exact note names/bytes from verified parents or legacy payloads."""
    entries = validated_source_notes(recipe)
    pending, already_present = [], []
    # Verify every available parent and destination before publishing any note.
    for key, entry in entries.items():
        path = relative_path(sources, key)
        if "source_key" in entry:
            parent_path = relative_path(sources, entry["source_key"])
            data = parent_path.read_bytes()
            if not data.startswith(PNG_SIGNATURE) or len(data) != entry["file_bytes"] or sha256(data) != entry["sha256"]:
                raise ValueError(f"Available source note parent changed: {entry['source_key']}")
        else:
            data = entry["data"]
        if path.exists():
            existing = path.read_bytes()
            if len(existing) != len(data) or sha256(existing) != entry["sha256"]:
                raise ValueError(f"Existing source note differs; no files overwritten: {key}")
            already_present.append(key)
        else:
            pending.append((key, path, data))
    restored = []
    for key, path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".restore-source-note-", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)  # Atomic publication without replacing any file.
            restored.append(key)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"restored_source_notes": restored, "already_present_source_notes": already_present,
            "source_parent_pixels_modified": False, "network_used": False}


def download_evidence(source: dict, asset: str, audit: dict) -> dict | None:
    """Accept only prior byte-for-byte audit evidence, never a guessed asset URL."""
    for entry in audit.get("files", []):
        url = entry.get("published_url", "")
        parsed = urlparse(url)
        if (entry.get("asset_id") == asset and entry.get("filename") == source["filename"]
                and entry.get("published_exact_match") is True
                and entry.get("source_stable") is not False
                and entry.get("local_bytes") == source["file_bytes"] == entry.get("published_bytes")
                and parsed.scheme == "https" and parsed.hostname == "dl.polyhaven.org"
                and parsed.path.startswith("/file/ph-assets/Textures/")
                and Path(parsed.path).name == source["filename"]):
            official = entry.get("published_md5")
            actual = entry.get("local_md5")
            if not all(isinstance(v, str) and re.fullmatch(r"[a-fA-F0-9]{32}", v) for v in (official, actual)):
                continue
            if official.casefold() != actual.casefold():
                continue
            if Path(entry.get("path", "")).name != source["filename"]:
                continue
            # Bind the audited MD5 to the source whose SHA256 was captured during
            # preparation. A shared filename and byte count alone are insufficient.
            prepared_md5 = source.get("file_md5")
            if prepared_md5 is not None:
                if (not isinstance(prepared_md5, str)
                        or re.fullmatch(r"[a-fA-F0-9]{32}", prepared_md5) is None
                        or prepared_md5.casefold() != actual.casefold()):
                    continue
                identity_basis = "Matching full-file MD5 in preparation metadata"
            else:
                expected = entry.get("current_stat")
                try:
                    state = Path(source["path"]).stat()
                except (KeyError, OSError):
                    continue
                current = {"device": state.st_dev, "inode": state.st_ino,
                           "size": state.st_size, "mtime_ns": state.st_mtime_ns}
                if not isinstance(expected, dict) or current != expected:
                    continue
                identity_basis = "Current parent device/inode/size/mtime match the full-file audit"
            return {"url": url, "published_md5": official, "published_bytes": entry["published_bytes"],
                    "audit_time_utc": audit.get("snapshot_completed_utc") or audit.get("timestamp_utc"),
                    "api_provenance": entry.get("api_provenance", {"api_url": entry.get("api_url")}),
                    "source_identity_basis": identity_basis,
                    "verification_basis": "Previous exact full-file MD5 and byte-count audit; SHA256 from preparation metadata, no rehash during recipe export"}
    return None


def package_download_evidence(source: dict, asset: str, package_audits: list[dict]) -> dict | None:
    """Record an independently verified archive member, without inventing a file URL."""
    if (not re.fullmatch(r"[a-f0-9]{64}", source.get("file_sha256", ""))
            or not isinstance(source.get("file_bytes"), int) or source["file_bytes"] <= 0):
        return None
    for audit in package_audits:
        if (audit.get("schema") != PACKAGE_AUDIT_SCHEMA or audit.get("provider") != "ambientCG"
                or audit.get("material_id") != asset):
            continue
        package = audit.get("package", {})
        parsed = urlparse(package.get("url", ""))
        filename = package.get("filename", "")
        asset_id = audit.get("asset_id", "")
        license_info = audit.get("license", {})
        if (not re.fullmatch(r"[A-Za-z0-9]+_[1248]K-PNG\.zip", filename)
                or not filename.startswith(asset_id + "_") or asset_id.casefold() != asset.casefold()
                or parsed.scheme != "https" or parsed.hostname != "ambientcg.com"
                or parsed.path != "/get" or parse_qs(parsed.query) != {"file": [filename]}
                or not re.fullmatch(r"[a-f0-9]{64}", package.get("sha256", ""))
                or not isinstance(package.get("file_bytes"), int) or package["file_bytes"] <= 0
                or license_info.get("spdx") != "CC0-1.0"
                or license_info.get("url") != "https://docs.ambientcg.com/license/"
                or not re.fullmatch(r"[a-f0-9]{64}", license_info.get("snapshot_sha256", ""))):
            continue
        for entry in audit.get("files", []):
            member = entry.get("archive_member", "")
            if (entry.get("source_filename") != source["filename"]
                    or entry.get("exact_full_file_match") is not True
                    or entry.get("source_stable") is False
                    or entry.get("source_sha256") != source["file_sha256"]
                    or entry.get("member_sha256") != source["file_sha256"]
                    or entry.get("source_bytes") != source["file_bytes"]
                    or entry.get("member_bytes") != source["file_bytes"]
                    or not member or Path(member).is_absolute() or ".." in Path(member).parts
                    or not member.startswith(filename[:-4] + "_") or not member.endswith(".png")):
                continue
            return {"url": package["url"], "download_kind": "zip_archive_member",
                    "archive_filename": filename, "archive_sha256": package["sha256"],
                    "archive_bytes": package["file_bytes"], "archive_member": member,
                    "member_sha256": entry["member_sha256"], "member_bytes": entry["member_bytes"],
                    "restore_filename": source["filename"], "provider": "ambientCG", "asset_id": asset_id,
                    "asset_url": audit.get("asset_url"), "api_provenance": audit.get("api_provenance"),
                    "creation_method": audit.get("creation_method"), "license": license_info,
                    "audit_time_utc": audit.get("completed_utc"),
                    "verification_basis": "Downloaded official package; archive SHA256/byte count and each full member/source SHA256/byte count matched; archive member and local restoration filenames are explicitly recorded"}
    return None


def export_recipe(dataset: Path, audit_path: Path | None = None,
                  package_audit_paths: list[Path] | None = None) -> dict:
    dataset = dataset.resolve()
    raw_index = (dataset / "dataset.json").read_bytes()
    index = json.loads(raw_index)
    if index.get("schema_version") != 2 or not isinstance(index.get("samples"), list):
        raise ValueError("Requires the prepared version-2 dataset index")
    raw_audit = audit_path.read_bytes() if audit_path else None
    audit = json.loads(raw_audit) if raw_audit else {}
    package_audits = [(path, path.read_bytes()) for path in (package_audit_paths or [])]
    package_reports = [json.loads(data) for _, data in package_audits]
    package_audit_hashes = {}
    for path, data in package_audits:
        digest = sha256(data)
        if path.name in package_audit_hashes and package_audit_hashes[path.name] != digest:
            raise ValueError("Different package audits share a filename; use unique report filenames")
        package_audit_hashes[path.name] = digest
    originals = {}
    records = {}
    notes = {}
    source_paths = {}
    pending_notes = []
    for item in index["samples"]:
        folder = relative_path(dataset, item["path"])
        raw = (folder / "sample.json").read_bytes()
        record = json.loads(raw)
        identity = record["sample_id"]
        if identity in records:
            raise ValueError("Duplicate sample identity")
        for field in ("sample_id", "material_id", "split", "status"):
            if item[field] != record[field]:
                raise ValueError(f"Index/sample disagreement: {identity} {field}")
        records[identity] = {"metadata": record, "metadata_sha256": sha256(raw)}
        for details in record["map_metadata"].values():
            source = details["source"]
            key = record["material_id"] + "/" + source["filename"]
            essential = {field: source[field] for field in ("filename", "file_bytes", "file_sha256", "width", "height", "sample_bits", "channels", "suffix", "resolution_label")}
            if key in originals and originals[key]["source"] != essential:
                raise ValueError(f"Conflicting source records: {key}")
            originals[key] = {"material_id": record["material_id"], "source": essential,
                              "download": (download_evidence(source, record["material_id"], audit)
                                           or package_download_evidence(source, record["material_id"], package_reports))}
            if source.get("path"):
                source_paths[key] = source["path"]
        for note in record.get("source_notes", []):
            pending_notes.append((record["material_id"], Path(note["path"]).name, relative_path(folder, note["path"]), note["sha256"]))
    for material, filename, path, expected in pending_notes:
        data = path.read_bytes()
        if sha256(data) != expected:
            raise ValueError(f"Changed source note: {path}")
        key = material + "/" + filename
        if key in notes and (notes[key]["sha256"] != expected or notes[key]["file_bytes"] != len(data)):
            raise ValueError(f"Conflicting source note records: {key}")
        if key not in notes:
            notes[key] = source_note_entry(data, expected, material, originals, source_paths)
    script_root = Path(__file__).parent
    code = {name: sha256((script_root / name).read_bytes()) for name in ("material_dataset.py", "prepare-materials.sh") if (script_root / name).exists()}
    code_snapshot = {name: (script_root / name).read_text() for name in code}
    unresolved = [key for key, value in originals.items() if value["download"] is None]
    embedded_png_note_count = sum(note.get("contains_png_image_payload") is True for note in notes.values())
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_index": index, "dataset_index_sha256": sha256(raw_index), "sample_records": records,
            "sources": originals, "source_notes": notes, "preparation_code_sha256": code,
            "preparation_code_snapshot": code_snapshot,
            "source_audit_sha256": sha256(raw_audit) if raw_audit else None,
            "package_audit_sha256": package_audit_hashes,
            "material_count": len({item["material_id"] for item in index["samples"]}),
            "required_parent_bytes": sum(value["source"]["file_bytes"] for value in originals.values()),
            "unresolved_source_files": unresolved, "fully_redownloadable_from_recorded_urls": not unresolved,
            "source_pixels_modified": False, "image_payloads_copied": bool(embedded_png_note_count),
            "embedded_png_source_note_count": embedded_png_note_count,
            "source_note_storage": "Unique notes embedded unchanged; exact verified PNG duplicates reference a recorded same-material parent without duplicate payload bytes",
            "scope": "Selected generated samples, original parent identities, crop/split/encoding/normal transforms, notes and exact metadata hashes. Does not retain unindexed manual samples. " + ("Unmatched PNG source notes are embedded unchanged, including their image bytes." if embedded_png_note_count else "Ordinary parent/map image payloads are not embedded; exact PNG note duplicates reference verified parents."),
            "recovery_steps": ["Download parents sequentially using recorded URLs into a separate sources directory.",
                               "Download each shared ZIP package once, verify its archive SHA256/byte count, and extract only the recorded members under restore_filename; do not extract arbitrary ZIP paths.",
                               "Verify byte count, full SHA256 and published MD5 before using a download; provider files can change.",
                               "Obtain unresolved source files independently and verify their recorded SHA256.",
                               "Restore notes with restore-notes after verifying available parents; referenced notes retain their original filenames and exact bytes. Legacy embedded notes remain supported.",
                               "Use the recorded crop rectangles, transforms and original precision, with the recorded preparation code revision.",
                               "Verify decoded crop hashes against sample_records; PNG container bytes may differ across encoder versions."]}


def plan(recipe: dict) -> dict:
    if recipe.get("schema") != SCHEMA:
        raise ValueError("Unsupported recreation recipe")
    values = recipe["sources"]
    notes = validated_source_notes(recipe)
    embedded_png_note_count = sum(entry.get("data", b"").startswith(PNG_SIGNATURE) for entry in notes.values())
    downloads = {}
    for value in values.values():
        entry = value["download"]
        if entry is not None:
            size = entry.get("archive_bytes", entry.get("published_bytes"))
            if entry["url"] in downloads and downloads[entry["url"]] != size:
                raise ValueError("Conflicting sizes for a shared download URL")
            downloads[entry["url"]] = size
    return {"material_count": recipe["material_count"], "sample_count": len(recipe["sample_records"]),
            "original_file_count": len(values), "required_parent_bytes": recipe["required_parent_bytes"],
            "downloadable_files": sum(value["download"] is not None for value in values.values()),
            "unique_download_count": len(downloads),
            "recorded_download_bytes": sum(downloads.values()),
            "unresolved_source_files": recipe["unresolved_source_files"],
            "fully_redownloadable_from_recorded_urls": recipe["fully_redownloadable_from_recorded_urls"],
            "embedded_source_note_count": sum("data" in entry for entry in notes.values()),
            "embedded_source_note_bytes": sum(entry["file_bytes"] for entry in notes.values() if "data" in entry),
            "parent_reference_source_note_count": sum("source_key" in entry for entry in notes.values()),
            "parent_reference_source_note_bytes": sum(entry["file_bytes"] for entry in notes.values() if "source_key" in entry),
            "embedded_png_source_note_count": embedded_png_note_count,
            "image_payloads_copied": bool(embedded_png_note_count),
            "network_used": False, "source_pixels_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--dataset", type=Path, required=True)
    exporting.add_argument("--source-audit", type=Path)
    exporting.add_argument("--package-audit", type=Path, action="append", default=[],
                           help="Previously verified package/member provenance report; repeatable")
    exporting.add_argument("--output", type=Path, required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--recipe", type=Path, required=True)
    restoring = commands.add_parser("restore-notes", help="Restore source notes without downloading or changing parents")
    restoring.add_argument("--recipe", type=Path, required=True)
    restoring.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        recipe = export_recipe(args.dataset, args.source_audit, args.package_audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.output.suffix == ".gz" else open
        with opener(args.output, "xt", encoding="utf-8") as stream:
            json.dump(recipe, stream, indent=2)
            stream.write("\n")
        print(json.dumps({**plan(recipe), "recipe_bytes": args.output.stat().st_size}, indent=2))
    else:
        opener = gzip.open if args.recipe.suffix == ".gz" else open
        with opener(args.recipe, "rt", encoding="utf-8") as stream:
            recipe = json.load(stream)
        print(json.dumps(plan(recipe) if args.command == "plan" else restore_source_notes(recipe, args.sources), indent=2))


if __name__ == "__main__":
    main()
