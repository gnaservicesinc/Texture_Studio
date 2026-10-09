#!/usr/bin/env python3
"""JSON bridge for source-referenced material dataset curation."""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any


from material_pbrnxt import sha256 as digest


def checked_relative(root: Path, name: str) -> Path:
    relative = Path(name)
    result = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not result.is_relative_to(root.resolve()):
        raise ValueError("Dataset metadata path must stay inside its directory")
    return result


SCHEMA = "texture-studio-material-workbench-v1"
ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ".material-workbench-journal.json"


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, value: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def commit_metadata(dataset: Path, updates: list[tuple[Path, dict]]) -> None:
    changes = []
    for path, value in updates:
        content = json_bytes(value)
        changes.append({"path": str(path.resolve().relative_to(dataset.resolve())), "old_sha256": digest(path),
                        "new_sha256": hashlib.sha256(content).hexdigest(), "new_base64": base64.b64encode(content).decode()})
    atomic_bytes(dataset / JOURNAL, json_bytes({"schema": SCHEMA, "changes": changes}))
    recover_journal(dataset)


def canonical_dataset(path: Path) -> Path:
    if path.name == "dataset.json" and path.is_file():
        path = path.parent
    if path.name == "sources" and (path.parent / "dataset.json").is_file():
        path = path.parent
    return path.resolve()


def read_dataset(dataset: Path) -> tuple[dict, list[tuple[dict, Path, dict]]]:
    dataset = dataset.resolve()
    index = json.loads((dataset / "dataset.json").read_text())
    from material_native_size import ensure_source_index
    index = ensure_source_index(dataset, index)
    if index.get("schema_version") != 2 or not isinstance(index.get("samples"), list):
        raise ValueError("Expected a prepared schema-2 material dataset")
    records, seen = [], set()
    for item in index["samples"]:
        path = checked_relative(dataset, item["path"]) / "sample.json"
        sample = json.loads(path.read_text())
        if any(sample.get(key) != item.get(key) for key in ("sample_id", "material_id", "status", "split")):
            raise ValueError(f"Dataset/sample identity disagrees: {path}")
        if sample["sample_id"] in seen:
            raise ValueError("Duplicate dataset sample identity")
        seen.add(sample["sample_id"])
        records.append((item, path, sample))
    return index, records


def recover_journal(dataset: Path) -> None:
    journal = dataset / JOURNAL
    if not journal.exists():
        return
    changes = json.loads(journal.read_text())
    if changes.get("schema") != SCHEMA or set(changes) != {"schema", "changes"}:
        raise ValueError("Unrecognized material curation recovery journal")
    pending = []
    for change in changes["changes"]:
        path = checked_relative(dataset, change["path"])
        if path.name not in ("sample.json", "dataset.json"):
            raise ValueError("Curation recovery may only edit dataset/sample JSON")
        content = base64.b64decode(change["new_base64"], validate=True)
        if hashlib.sha256(content).hexdigest() != change["new_sha256"] or digest(path) not in (change["old_sha256"], change["new_sha256"]):
            raise ValueError("Curation recovery identity changed; original data remains untouched")
        pending.append((path, content))
    for path, content in pending:
        atomic_bytes(path, content)
    journal.unlink()
    sync_directory(journal.parent)


@contextlib.contextmanager
def dataset_lock(dataset: Path):
    dataset = dataset.resolve()
    with (dataset / ".material-workbench.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        recover_journal(dataset)
        yield


def training_active(dataset: Path) -> bool:
    # The native UI additionally disables editing while its child runs. This
    # check also protects against a trainer launched directly in a terminal.
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        if not any(name in command for name in ("material_model_workbench.py", "train_material_pbrnxt.py")):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            return True
        for i, token in enumerate(tokens):
            selected = tokens[i + 1] if token == "--dataset" and i + 1 < len(tokens) else token.partition("=")[2] if token.startswith("--dataset=") else None
            if selected:
                selected_path = Path(selected).resolve()
                if selected_path == dataset.resolve() or (selected_path.parent.name == ".training-data" and selected_path.parent.parent == dataset.resolve()):
                    return True
    return False


def dataset_info(args) -> dict:
    from material_dataset import png_image_header, resolve_map_path
    from material_native_size import SUPPORTED_SIZES, recover_original, source_review_records
    args.dataset = canonical_dataset(args.dataset)
    with dataset_lock(args.dataset):
        index, records = read_dataset(args.dataset)
        original_dataset = Path(index.get("native_size_preparation", {}).get("source_dataset_path", str(args.dataset)))
        review_size = getattr(args, "review_size", None)
        reviews_path = args.dataset / ".material-size-reviews.json"
        reviews = json.loads(reviews_path.read_text()) if review_size and reviews_path.is_file() else {}
        original_records = records
        if review_size and not index.get("native_size_preparation"):
            records = source_review_records(records, review_size)
        materials = {}
        for _item, path, sample in records:
            if review_size and not index.get("native_size_preparation"):
                sample = dict(sample)
                sample.update(reviews.get(f"{review_size}:{sample['sample_id']}", {}))
            maps = {}
            for role, filename in sample.get("maps", {}).items():
                details = sample.get("map_metadata", {}).get(role, {})
                map_path = resolve_map_path(path.parent, sample, role)
                source = details.get("source", {})
                original_path = Path(source["path"]) if isinstance(source.get("path"), str) else None
                if original_path is not None and not original_path.is_file():
                    recovered = recover_original(source, original_dataset)
                    if recovered is not None:
                        original_path = recovered
                        if details.get("storage") == "source_reference":
                            map_path = recovered
                try:
                    header = png_image_header(map_path)
                    actual = {"width": header["width"], "height": header["height"], "dimensions_verified": True}
                except (OSError, ValueError) as error:
                    # Let automatic preparation rebuild from original parents;
                    # loading a dataset with a repairable map is still useful.
                    actual = {"width": None, "height": None, "dimensions_verified": False,
                              "dimension_issue": str(error)}
                convention = ("directx" if role == "normal" and details.get("storage") == "source_reference"
                    and details.get("source", {}).get("suffix") == "nor_dx" else "opengl" if role == "normal" else None)
                maps[role] = {"path": str(map_path), "sha256": details.get("sample_sha256"),
                    "source_bits": details.get("sample_bits", details.get("source", {}).get("sample_bits")),
                    "encoding": details.get("encoding"), "source_normal_convention": convention,
                    "original_source_path": str(original_path) if original_path is not None else None,
                    "original_source_sha256": source.get("file_sha256"),
                    "original_source_width": source.get("width"), "original_source_height": source.get("height"),
                    "original_normal_convention": "directx" if source.get("suffix") == "nor_dx" else "opengl" if role == "normal" else None,
                    "crop_rectangle": sample.get("crop_rectangle_top_left_xywh"),
                    **actual}
            input_variants = []
            for variant in sample.get("input_variants", []):
                source = variant.get("source", {})
                if source.get("file_sha256") == sample.get("map_metadata", {}).get("input", {}).get("source", {}).get("file_sha256"):
                    input_variants.append(dict(maps["input"], variant_id=variant.get("variant_id", "default")))
                    continue
                filename = variant.get("filename", variant.get("path", source.get("path")))
                variant_path = Path(filename)
                if not variant_path.is_absolute():
                    variant_path = path.parent / variant_path
                original_path = Path(source["path"]) if isinstance(source.get("path"), str) else None
                if original_path is not None and not original_path.is_file():
                    recovered = recover_original(source, original_dataset)
                    if recovered is not None:
                        original_path = recovered
                        if variant.get("storage") == "source_reference":
                            variant_path = recovered
                try:
                    header = png_image_header(variant_path)
                    actual = {"width": header["width"], "height": header["height"], "dimensions_verified": True}
                except (OSError, ValueError) as error:
                    actual = {"width": None, "height": None, "dimensions_verified": False, "dimension_issue": str(error)}
                input_variants.append({"path": str(variant_path), "sha256": variant.get("sample_sha256"),
                    "variant_id": variant.get("variant_id", "default"), "encoding": variant.get("encoding"),
                    "source_bits": variant.get("sample_bits", source.get("sample_bits")),
                    "original_source_path": str(original_path) if original_path is not None else None,
                    "original_source_sha256": source.get("file_sha256"), "original_source_width": source.get("width"),
                    "original_source_height": source.get("height"), "crop_rectangle": sample.get("crop_rectangle_top_left_xywh"),
                    **actual})
            width, height = sample.get("sample_pixel_dimensions", [None, None])
            material = materials.setdefault(sample["material_id"], {"material_id": sample["material_id"],
                "asset_family_id": sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"])),
                "source_set_id": sample.get("source_set_id"), "samples": []})
            material["samples"].append({"sample_id": sample["sample_id"], "status": sample["status"], "split": sample["split"], "width": width, "height": height, "maps": maps, "crop_rectangle": sample.get("crop_rectangle_top_left_xywh"), "note": sample.get("curation_note"), "metadata_path": str(path), "input_variant_count": len(input_variants) or 1, "input_variants": input_variants,
                "asset_family_id": sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"])),
                "source_family_id": sample.get("source_family_id", sample.get("asset_family_id", sample["material_id"])),
                "source_set_id": sample.get("source_set_id"), "source_region_id": sample.get("source_region_id"),
                "available_targets": sample.get("available_targets", [role for role in ("height", "roughness", "normal") if role in sample.get("maps", {})]),
                "validation_scope": sample.get("validation_scope")})
        eligible_sizes = []
        for size in SUPPORTED_SIZES:
            for _entry, _path, sample in original_records:
                if sample["status"] in ("excluded", "rejected"):
                    continue
                all_sources = [d.get("source", {}) for d in sample.get("map_metadata", {}).values()] + [d.get("source", {}) for d in sample.get("input_variants", [])]
                if all_sources and all(min(source.get("width", 0), source.get("height", 0)) >= size for source in all_sources):
                    eligible_sizes.append(size)
                    break
        return {"dataset_path": str(args.dataset.resolve()), "index_sha256": digest(args.dataset / "dataset.json"),
            "supported_training_sizes": eligible_sizes,
            "split_strategy": index.get("split_strategy"), "validation_scope": index.get("validation_scope"),
            "cross_size_validation_notice": index.get("native_size_preparation", {}).get("cross_size_validation_notice"),
            "automatic_validation": index.get("automatic_validation"), "materials": [materials[key] for key in sorted(materials)]}


def cleanup_size(args) -> dict:
    from material_native_size import cleanup_prepared_dataset
    dataset = canonical_dataset(args.dataset)
    if training_active(dataset):
        raise ValueError("Stop active material training before removing its temporary data")
    return cleanup_prepared_dataset(dataset)


def remove_missing(args) -> dict:
    """Remove a listed material only when its original source is truly absent."""
    from material_dataset import resolve_map_path
    from material_native_size import recover_original
    dataset = canonical_dataset(args.dataset)
    with dataset_lock(dataset):
        index, records = read_dataset(dataset)
        if args.expected_index_sha256 and digest(dataset / "dataset.json") != args.expected_index_sha256:
            raise ValueError("Dataset changed since selection; reload it before removing a missing source")
        if getattr(args, "review_size", None) and not index.get("native_size_preparation"):
            from material_native_size import source_review_records
            records = source_review_records(records, args.review_size)
        selected = next(((entry, path, record) for entry, path, record in records if record["sample_id"] == args.sample), None)
        if selected is None:
            return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": False}
        entry, metadata_path, record = selected
        paths = {role: resolve_map_path(metadata_path.parent, record, role) for role in record["maps"]}
        roles = [role for role, path in paths.items() if path.absolute() == args.path.absolute()]
        if not roles:
            variants = []
            for descriptor in record.get("input_variants", []):
                variant_path = resolve_map_path(metadata_path.parent,
                    {"maps": {"input": descriptor.get("path") or descriptor.get("filename")},
                     "map_metadata": {"input": descriptor}}, "input")
                if variant_path.absolute() == args.path.absolute():
                    variants.append(descriptor)
            if len(variants) != 1:
                raise ValueError("Missing map path differs from the selected listed material")
            variant = variants[0]
            if args.path.is_file():
                return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": False}
            binding = index.get("native_size_preparation", {})
            original = Path(binding.get("source_dataset_path", str(dataset)))
            recovered = recover_original(variant["source"], original)
            if recovered is not None:
                return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": False,
                        "recoverable": True, "source_path": str(recovered)}
            # An unavailable alternate color does not discard intact geometry
            # or other registered colors. Update the actual source record when
            # the selected inspector entry is a virtual native crop.
            stored = json.loads(metadata_path.read_text())
            stored["input_variants"] = [descriptor for descriptor in stored.get("input_variants", [])
                                        if descriptor.get("variant_id") != variant.get("variant_id")]
            if not binding:
                source = variant["source"]
                tombstones = index.setdefault("removed_missing_color_variants", [])
                tombstone = {"source_set_id": record.get("source_set_id"), "material_id": record["material_id"],
                             "variant_id": variant["variant_id"], "path": source["path"], "file_sha256": source["file_sha256"]}
                if tombstone not in tombstones:
                    tombstones.append(tombstone)
            commit_metadata(dataset, [(metadata_path, stored), (dataset / "dataset.json", index)])
            return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": True,
                    "removed_variant_id": variant["variant_id"], "source_images_modified": False,
                    "index_sha256": digest(dataset / "dataset.json")}
        if len(roles) != 1:
            raise ValueError("Missing map path differs from the selected listed material")
        role = roles[0]
        if paths[role].is_file():
            return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": False}
        binding = index.get("native_size_preparation", {})
        original = Path(binding.get("source_dataset_path", str(dataset)))
        recovered = recover_original(record["map_metadata"][role]["source"], original)
        if recovered is not None:
            return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": False,
                    "recoverable": True, "source_path": str(recovered)}
        index["samples"] = [item for item in index["samples"] if item["sample_id"] != record.get("source_record_sample_id", args.sample)]
        if not binding:
            source = record["map_metadata"][role]["source"]
            tombstones = index.setdefault("removed_missing_sources", [])
            tombstone = {"source_set_id": record.get("source_set_id"), "material_id": record["material_id"],
                         "path": source["path"], "file_sha256": source["file_sha256"]}
            if tombstone not in tombstones:
                tombstones.append(tombstone)
        atomic_bytes(dataset / "dataset.json", json_bytes(index))
        return {"dataset_path": str(dataset), "sample_id": args.sample, "removed": True,
                "source_images_modified": False, "index_sha256": digest(dataset / "dataset.json")}


def prepare_size(args) -> dict:
    from material_native_size import PREPARATION_SCHEMA, prepare_from_records
    supplied = canonical_dataset(args.dataset)
    if training_active(supplied):
        raise ValueError("Stop active material training before changing its training resolution")
    with dataset_lock(supplied):
        supplied_index, _records = read_dataset(supplied)
        lineage = supplied_index.get("native_size_preparation", {})
        original = supplied
        if lineage:
            if lineage.get("schema") != PREPARATION_SCHEMA or not Path(lineage.get("source_dataset_path", "")).is_absolute():
                raise ValueError("Native crop dataset has an invalid original-dataset lineage")
            original = canonical_dataset(Path(lineage["source_dataset_path"]))
            if original == supplied:
                raise ValueError("Native crop dataset has circular original-dataset lineage")
    # Always acquire the original before a derivative. Cached curation uses
    # its own lock, so inverse locking could otherwise deadlock a size switch.
    with contextlib.ExitStack() as stack:
        for dataset in sorted({original, supplied}, key=str):
            stack.enter_context(dataset_lock(dataset))
        supplied_index, supplied_records = read_dataset(supplied)
        if args.expected_index_sha256 and digest(supplied / "dataset.json") != args.expected_index_sha256:
            raise ValueError("Dataset changed since selection; reload it before preparing native crops")
        current_lineage = supplied_index.get("native_size_preparation", {})
        if current_lineage != lineage:
            raise ValueError("Native dataset lineage changed; reload it before preparing crops")
        options = dict(automatic_validation=getattr(args, "automatic_validation", False),
                       validation_material=getattr(args, "material", None), target=getattr(args, "target", "height"))
        if original == supplied:
            prepared, information = prepare_from_records(original, supplied_records, args.size, **options)
        else:
            _original_index, original_records = read_dataset(original)
            prepared, information = prepare_from_records(original, original_records, args.size, review_records=supplied_records, **options)
    result = dataset_info(argparse.Namespace(dataset=prepared))
    result["preparation"] = information
    return result


def curate(args) -> dict:
    args.dataset = canonical_dataset(args.dataset)
    if training_active(args.dataset):
        raise ValueError("Stop active material training before editing its dataset")
    with dataset_lock(args.dataset):
        index, records = read_dataset(args.dataset)
        index_path = args.dataset / "dataset.json"
        previous = digest(index_path)
        if args.expected_index_sha256 and previous != args.expected_index_sha256:
            raise ValueError("Dataset changed since selection; reload it before curation")
        review_size = getattr(args, "review_size", None)
        if review_size and not index.get("native_size_preparation"):
            from material_native_size import source_review_records
            records = source_review_records(records, review_size)
        matches = [(item, path, sample) for item, path, sample in records if sample["sample_id"] == args.sample]
        if len(matches) != 1:
            raise ValueError("Select one exact indexed sample ID")
        item, path, sample = matches[0]
        review_size = getattr(args, "review_size", None)
        if review_size and not index.get("native_size_preparation"):
            reviews_path = args.dataset / ".material-size-reviews.json"
            reviews = json.loads(reviews_path.read_text()) if reviews_path.is_file() else {}
            key = f"{review_size}:{sample['sample_id']}"
            review = dict(reviews.get(key, {}))
            review.update(status=args.status, review_status={"approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"}[args.status],
                          split=args.split or review.get("split", sample["split"]))
            if args.note is not None:
                review["curation_note"] = args.note
            reviews[key] = review
            atomic_bytes(reviews_path, json_bytes(reviews))
            return {"dataset_path": str(args.dataset), "sample_id": args.sample, "status": args.status,
                    "split": review["split"], "review_size": review_size, "index_sha256": previous,
                    "metadata_sha256": digest(reviews_path), "source_bytes_modified": False}
        sample["status"] = item["status"] = args.status
        changed_records = [(item, path, sample)]
        if args.split:
            family = sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"]))
            siblings = [(entry, metadata_path, record) for entry, metadata_path, record in records
                if record.get("asset_family_id", record.get("source_family_id", record["material_id"])) == family]
            source_sets = {record.get("source_set_id", record["material_id"]) for _entry, _path, record in siblings}
            # Every resolution/color sibling must share its split. A single
            # native 8K source can instead reserve nonoverlapping known regions.
            changed_records = siblings if len(source_sets) > 1 or not index.get("native_size_preparation") else [(item, path, sample)]
            for sibling_entry, _sibling_path, sibling in changed_records:
                sibling["split"] = sibling_entry["split"] = args.split
            from material_pbrnxt_data import rectangles_overlap
            training = [record for _entry, _path, record in siblings if record["split"] == "train"]
            checks = [record for _entry, _path, record in siblings if record["split"] == "validation"]
            if any(rectangles_overlap(check["crop_rectangle_top_left_xywh"], train["crop_rectangle_top_left_xywh"])
                   for check in checks for train in training):
                raise ValueError("Validation cannot overlap training pixels from the same source family")
        if args.note is not None:
            sample["curation_note"] = args.note
        sample["review_status"] = {"approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"}[args.status]
        metadata_changes = [(changed_path, changed_sample) for _entry, changed_path, changed_sample in changed_records]
        commit_metadata(args.dataset, [*metadata_changes, (index_path, index)])
        return {"dataset_path": str(args.dataset.resolve()), "sample_id": args.sample, "status": args.status, "split": sample["split"], "index_sha256": digest(index_path), "previous_index_sha256": previous, "metadata_sha256": digest(path), "source_bytes_modified": False}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("dataset", "prepare-size", "cleanup-size", "remove-missing", "curate"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", type=Path, required=True)
        if name in ("dataset", "curate", "remove-missing"):
            sub.add_argument("--review-size", type=int, choices=(256, 512, 1024, 2048))
        if name == "remove-missing":
            sub.add_argument("--sample", required=True)
            sub.add_argument("--path", type=Path, required=True)
            sub.add_argument("--expected-index-sha256")
        if name == "prepare-size":
            sub.add_argument("--size", type=int, choices=(256, 512, 1024, 2048), required=True)
            sub.add_argument("--expected-index-sha256")
            sub.add_argument("--automatic-validation", action="store_true")
            sub.add_argument("--target", choices=("height", "roughness", "normal"), default="height")
            sub.add_argument("--material")
        if name == "curate":
            sub.add_argument("--sample", required=True)
            sub.add_argument("--status", choices=("approved", "excluded", "unreviewed"), required=True)
            sub.add_argument("--split", choices=("train", "validation"))
            sub.add_argument("--note")
            sub.add_argument("--expected-index-sha256")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = {"dataset": dataset_info, "prepare-size": prepare_size, "cleanup-size": cleanup_size,
                      "remove-missing": remove_missing, "curate": curate}[args.command](args)
        print(json.dumps(dict(result, schema=SCHEMA, protocol_schema=SCHEMA, command=args.command, ok=True), allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        details = {"code": type(error).__name__, "message": str(error)}
        if hasattr(error, "details"):
            details["details"] = error.details
        print(json.dumps({"schema": SCHEMA, "command": args.command, "ok": False, "error": details}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
