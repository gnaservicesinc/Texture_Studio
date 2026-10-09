#!/usr/bin/env python3
"""JSON bridge for source-referenced material dataset curation."""
from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime, timezone
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
import uuid

# Resolve bundled sibling modules even when Python disables ambient import paths.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def digest(path: Path) -> str:
    """Metadata management must not require an installed training framework."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_relative(root: Path, name: str) -> Path:
    relative = Path(name)
    result = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not result.is_relative_to(root.resolve()):
        raise ValueError("Dataset metadata path must stay inside its directory")
    return result


SCHEMA = "texture-studio-material-workbench-v1"
ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ".material-workbench-journal.json"
MANAGEMENT_SCHEMA = "texture-studio-dataset-management-v1"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_digest(path: Path) -> str | None:
    return digest(path) if path.is_file() else None


def review_digest(dataset: Path) -> str:
    # A concrete absence fingerprint lets the first review reject a stale
    # selection from another window which created the file in the meantime.
    return metadata_digest(dataset / ".material-size-reviews.json") or hashlib.sha256(b"").hexdigest()


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
        changes.append({"path": str(path.resolve().relative_to(dataset.resolve())), "old_sha256": metadata_digest(path),
                        "new_sha256": hashlib.sha256(content).hexdigest(), "new_base64": base64.b64encode(content).decode()})
    atomic_bytes(dataset / JOURNAL, json_bytes({"schema": SCHEMA, "changes": changes}))
    recover_journal(dataset)


def canonical_dataset(path: Path) -> Path:
    if path.name == "dataset.json" and path.is_file():
        path = path.parent
    if path.name == "sources" and (path.parent / "dataset.json").is_file():
        path = path.parent
    if path.is_file():
        raise ValueError("Select a dataset folder or its dataset.json file")
    return path.resolve()


def review_records_with_fallback(records, size):
    """Keep originals that cannot produce the selected grid in the inspector."""
    from material_native_size import source_review_records
    reviewed = source_review_records(records, size)
    represented = {sample.get("source_record_sample_id", sample["sample_id"])
                   for _entry, _path, sample in reviewed}
    return reviewed + [record for record in records if record[2]["sample_id"] not in represented]


def read_dataset(dataset: Path) -> tuple[dict, list[tuple[dict, Path, dict]]]:
    dataset = dataset.resolve()
    if not (dataset / "dataset.json").is_file():
        raise ValueError("This folder does not contain a material dataset. Create a dataset or import material maps into one.")
    index = json.loads((dataset / "dataset.json").read_text())
    if index.get("schema_version") != 2 or not isinstance(index.get("samples"), list):
        raise ValueError("Expected a prepared schema-2 material dataset")
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
        if path.name not in ("sample.json", "dataset.json", ".material-size-reviews.json"):
            raise ValueError("Curation recovery may only edit dataset/sample JSON")
        content = base64.b64decode(change["new_base64"], validate=True)
        if hashlib.sha256(content).hexdigest() != change["new_sha256"] or metadata_digest(path) not in (change["old_sha256"], change["new_sha256"]):
            raise ValueError("Curation recovery identity changed; original data remains untouched")
        pending.append((path, content))
    for path, content in pending:
        atomic_bytes(path, content)
    journal.unlink()
    sync_directory(journal.parent)


@contextlib.contextmanager
def dataset_lock(dataset: Path):
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError("The dataset folder is missing. Create a dataset or choose an existing dataset folder.")
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
        if not any(name in command for name in ("material_model_workbench.py", "train_material_pbrnxt.py", "material_workbench.py")):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            return True
        if "material_workbench.py" in command and "prepare-size" not in tokens:
            continue
        for i, token in enumerate(tokens):
            selected = tokens[i + 1] if token == "--dataset" and i + 1 < len(tokens) else token.partition("=")[2] if token.startswith("--dataset=") else None
            if selected:
                selected_path = canonical_dataset(Path(selected))
                if selected_path == dataset.resolve() or (selected_path.parent.name == ".training-data" and selected_path.parent.parent == dataset.resolve()):
                    return True
    return False


def dataset_info(args) -> dict:
    from material_dataset import png_image_header, resolve_map_path
    from material_native_size import SUPPORTED_SIZES, current_review, recover_original, dataset_region_plan
    args.dataset = canonical_dataset(args.dataset)
    with dataset_lock(args.dataset):
        index, records = read_dataset(args.dataset)
        original_dataset = Path(index.get("native_size_preparation", {}).get("source_dataset_path", str(args.dataset)))
        metadata = index
        if index.get("native_size_preparation") and (original_dataset / "dataset.json").is_file():
            metadata = json.loads((original_dataset / "dataset.json").read_text())
        review_size = getattr(args, "review_size", None) or metadata.get("training_size") or getattr(args, "default_review_size", None)
        reviews_path = args.dataset / ".material-size-reviews.json"
        reviews = json.loads(reviews_path.read_text()) if review_size and reviews_path.is_file() else {}
        original_records = records
        plans, resolution_plans = {}, {}
        if review_size and not index.get("native_size_preparation"):
            resolution_plans = {str(size): {target: dataset_region_plan(records, size, reviews, target)
                                for target in ("height", "roughness", "normal")} for size in SUPPORTED_SIZES}
            plans = resolution_plans[str(review_size)]
        selected_plan = plans.get(getattr(args, "target", "height"), {})
        if review_size and not index.get("native_size_preparation"):
            records = review_records_with_fallback(records, review_size)
        materials = {}
        for _item, path, sample in records:
            if review_size and not index.get("native_size_preparation"):
                sample = dict(sample)
                sample.update(current_review(reviews, f"{review_size}:{sample['sample_id']}", sample))
                assignment = selected_plan.get("assignments", {}).get(sample["sample_id"], {})
                sample.update({key: assignment[key] for key in ("split", "split_assignment", "validation_scope") if key in assignment})
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
                "name": sample.get("name", sample["material_id"].replace("_", " ")),
                "asset_family_id": sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"])),
                "source_set_id": sample.get("source_set_id"), "samples": []})
            material["samples"].append({"sample_id": sample["sample_id"], "status": sample["status"], "split": sample["split"], "split_assignment": sample.get("split_assignment"), "width": width, "height": height, "maps": maps, "crop_rectangle": sample.get("crop_rectangle_top_left_xywh"), "note": sample.get("curation_note"), "metadata_path": str(path), "input_variant_count": len(input_variants) or 1, "input_variants": input_variants,
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
            "name": metadata.get("name", original_dataset.name), "description": metadata.get("description", ""),
            "dataset_id": metadata.get("dataset_id"), "created_utc": metadata.get("created_utc"),
            "updated_utc": metadata.get("updated_utc"), "dataset_management": metadata.get("dataset_management"),
            "original_dataset_path": str(original_dataset.resolve()), "review_sha256": review_digest(original_dataset),
            "material_count": len(materials), "sample_count": len(records),
            "training_size": metadata.get("training_size"), "review_size": review_size,
            "source_set_count": len(original_records),
            "training_plans": {key: {k: v for k, v in plan.items() if k != "assignments"} for key, plan in plans.items()},
            "resolution_plans": {size: {key: {k: v for k, v in plan.items() if k != "assignments"}
                                 for key, plan in targets.items()} for size, targets in resolution_plans.items()},
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
            records = review_records_with_fallback(records, args.review_size)
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
        expected_review = getattr(args, "expected_review_sha256", None)
        if expected_review and review_digest(original) != expected_review:
            raise ValueError("Dataset reviews changed since selection; reload it before preparing crops")
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
        original_records = records
        index_path = args.dataset / "dataset.json"
        previous = digest(index_path)
        if args.expected_index_sha256 and previous != args.expected_index_sha256:
            raise ValueError("Dataset changed since selection; reload it before curation")
        expected_review = getattr(args, "expected_review_sha256", None)
        review_dataset = Path(index.get("native_size_preparation", {}).get("source_dataset_path", str(args.dataset)))
        if expected_review and review_digest(review_dataset) != expected_review:
            raise ValueError("Dataset reviews changed since selection; reload it before curation")
        review_size = getattr(args, "review_size", None)
        if review_size and not index.get("native_size_preparation"):
            records = review_records_with_fallback(records, review_size)
        matches = [(item, path, sample) for item, path, sample in records if sample["sample_id"] == args.sample]
        if len(matches) != 1:
            raise ValueError("Select one exact indexed sample ID")
        item, path, sample = matches[0]
        if args.split == "automatic":
            if index.get("native_size_preparation"):
                raise ValueError("Open the source dataset to restore automatic splits")
            family = sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"]))
            siblings = [record for record in original_records if record[2].get("asset_family_id", record[2].get("source_family_id", record[2]["material_id"])) == family]
            material_ids = {record[2]["material_id"] for record in siblings}
            updates = []
            for entry, metadata_path, sibling in siblings:
                sibling.pop("split_assignment", None)
                sibling["split"] = entry["split"] = "train"
                updates.append((metadata_path, sibling))
            reviews_path = args.dataset / ".material-size-reviews.json"
            if reviews_path.is_file():
                reviews = json.loads(reviews_path.read_text())
                for key, review in reviews.items():
                    grid, _, identity = key.partition(":")
                    if (not review_size or grid == str(review_size)) and any(identity.startswith(m + "_") for m in material_ids):
                        review.pop("split_assignment", None)
                        review.pop("split", None)
                updates.append((reviews_path, reviews))
            commit_metadata(args.dataset, [*updates, (index_path, index)])
            return {"dataset_path": str(args.dataset), "sample_id": args.sample, "split": "automatic", "source_bytes_modified": False}
        if index.get("native_size_preparation"):
            from material_native_size import current_review
            original_reviews_path = review_dataset / ".material-size-reviews.json"
            original_reviews = json.loads(original_reviews_path.read_text()) if original_reviews_path.is_file() else {}
            for _entry, _path, record in records:
                record.setdefault("source_review_snapshot", current_review(original_reviews, f"{index['crop_size']}:{record['sample_id']}", record))
        review_size = getattr(args, "review_size", None)
        fallback_review_key = None
        if review_size and not index.get("native_size_preparation") and "source_record_sample_id" not in sample:
            # A map below the current grid is shown as its complete original.
            # Editing it applies to the source, so a later eligible grid keeps
            # the assignment instead of losing it in an unusable-size sidecar.
            fallback_review_key = f"{review_size}:{sample['sample_id']}"
            records = original_records
            item, path, sample = next(record for record in records if record[2]["sample_id"] == sample["sample_id"])
            review_size = None
        if review_size and not index.get("native_size_preparation"):
            from material_native_size import current_review, source_binding
            reviews_path = args.dataset / ".material-size-reviews.json"
            reviews = json.loads(reviews_path.read_text()) if reviews_path.is_file() else {}
            key = f"{review_size}:{sample['sample_id']}"
            review = dict(current_review(reviews, key, sample))
            review.update(status=args.status, review_status={"approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"}[args.status],
                          split=args.split or review.get("split", sample["split"]))
            review["source_binding_sha256"] = source_binding(sample)
            if args.split:
                review["split_assignment"] = "manual"
            if args.note is not None:
                review["curation_note"] = args.note
            reviews[key] = review
            if args.split:
                family = sample.get("asset_family_id", sample.get("source_family_id", sample["material_id"]))
                siblings = [record for _entry, _path, record in records
                    if record.get("asset_family_id", record.get("source_family_id", record["material_id"])) == family]
                source_sets = {record.get("source_set_id", record["material_id"]) for record in siblings}
                if len(source_sets) > 1:
                    for sibling in siblings:
                        sibling_key = f"{review_size}:{sibling['sample_id']}"
                        sibling_review = dict(current_review(reviews, sibling_key, sibling))
                        sibling_review.update(split=args.split, split_assignment="manual", source_binding_sha256=source_binding(sibling))
                        reviews[sibling_key] = sibling_review
                from material_dataset import rectangles_overlap
                training = [record for record in siblings if current_review(reviews, f"{review_size}:{record['sample_id']}", record).get("split", record["split"]) == "train"]
                checks = [record for record in siblings if current_review(reviews, f"{review_size}:{record['sample_id']}", record).get("split", record["split"]) == "validation"]
                if any(rectangles_overlap(check["crop_rectangle_top_left_xywh"], train["crop_rectangle_top_left_xywh"])
                       for check in checks for train in training):
                    raise ValueError("Validation cannot overlap training pixels from the same source family")
            atomic_bytes(reviews_path, json_bytes(reviews))
            return {"dataset_path": str(args.dataset), "sample_id": args.sample, "status": args.status,
                    "split": review["split"], "review_size": review_size, "index_sha256": previous,
                    "metadata_sha256": digest(reviews_path), "review_sha256": review_digest(args.dataset), "source_bytes_modified": False}
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
                sibling["split_assignment"] = "manual"
            from material_dataset import rectangles_overlap
            training = [record for _entry, _path, record in siblings if record["split"] == "train"]
            checks = [record for _entry, _path, record in siblings if record["split"] == "validation"]
            if any(rectangles_overlap(check["crop_rectangle_top_left_xywh"], train["crop_rectangle_top_left_xywh"])
                   for check in checks for train in training):
                raise ValueError("Validation cannot overlap training pixels from the same source family")
        if args.note is not None:
            sample["curation_note"] = args.note
        sample["review_status"] = {"approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"}[args.status]
        metadata_changes = [(changed_path, changed_sample) for _entry, changed_path, changed_sample in changed_records]
        reviews_path = args.dataset / ".material-size-reviews.json"
        if fallback_review_key and reviews_path.is_file():
            reviews = json.loads(reviews_path.read_text())
            if fallback_review_key in reviews:
                reviews.pop(fallback_review_key)
                metadata_changes.append((reviews_path, reviews))
        commit_metadata(args.dataset, [*metadata_changes, (index_path, index)])
        return {"dataset_path": str(args.dataset.resolve()), "sample_id": args.sample, "status": args.status, "split": sample["split"], "index_sha256": digest(index_path), "previous_index_sha256": previous, "metadata_sha256": digest(path), "source_bytes_modified": False}


def checked_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200 or any(ord(c) < 32 for c in value):
        raise ValueError("Enter a name with 1–200 characters and no control characters")
    return value.strip()


def checked_description(value: str | None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or "\0" in value or len(value) > 100000:
        raise ValueError("Dataset description must be text without null characters")
    return value


def check_index(dataset: Path, args) -> None:
    expected = getattr(args, "expected_index_sha256", None)
    if expected and metadata_digest(dataset / "dataset.json") != expected:
        raise ValueError("Dataset changed since selection; reload it before editing")


def editable_index(dataset: Path, args) -> tuple[dict, list]:
    check_index(dataset, args)
    index, records = read_dataset(dataset)
    check_index(dataset, args)
    if index.get("native_size_preparation"):
        raise ValueError("Open the original dataset to manage its information and materials")
    return index, records


def check_idle(dataset: Path) -> None:
    if training_active(dataset):
        raise ValueError("Stop active material training or preparation before editing its dataset")


def refreshed(dataset: Path, *, review_size=None, target="height", **details) -> dict:
    return dict(dataset_info(argparse.Namespace(dataset=dataset, review_size=review_size, target=target)), source_bytes_modified=False,
                source_images_modified=False, **details)


def checked_training_size(value: int) -> int:
    from material_native_size import SUPPORTED_SIZES
    if type(value) is not int or value not in SUPPORTED_SIZES:
        raise ValueError("Choose a rendering and training resolution of 256, 512, 1024 or 2048 pixels")
    return value


def create_dataset(args) -> dict:
    """Create a small owned metadata directory without creating image copies."""
    name = checked_name(args.name)
    description = checked_description(getattr(args, "description", ""))
    configured_size = getattr(args, "training_size", None)
    size = checked_training_size(1024 if configured_size is None else configured_size)
    dataset = args.dataset.expanduser().absolute()
    if dataset.is_symlink() or dataset.name in ("dataset.json", "sources", ".training-data"):
        raise ValueError("Choose a new dataset folder, not an existing metadata or source folder")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        dataset.mkdir()
        created = True
    except FileExistsError:
        if not dataset.is_dir() or any(dataset.iterdir()):
            raise ValueError("That folder already contains files. Choose a new or empty dataset folder.")
    try:
        with dataset_lock(dataset):
            # Another creator may have acquired the same directory lock first.
            if any(path.name != ".material-workbench.lock" for path in dataset.iterdir()):
                raise ValueError("That folder already contains files. Choose a new or empty dataset folder.")
            now = timestamp()
            from material_dataset import GENERATOR
            from material_native_size import SOURCE_STORAGE
            index = {"schema_version": 2, "generator": GENERATOR, "dataset_id": str(uuid.uuid4()),
                "name": name, "description": description, "created_utc": now, "updated_utc": now,
                "dataset_management": {"schema": MANAGEMENT_SCHEMA, "owns_directory": True},
                "storage_policy": SOURCE_STORAGE, "source_images_modified": False,
                "original_sources_required_for_training": True, "split_strategy": "whole-material-v1",
                "validation_scope": "held_out_materials", "training_size": size, "samples": []}
            atomic_bytes(dataset / "dataset.json", json_bytes(index))
    except BaseException:
        if created and not (dataset / "dataset.json").exists():
            # Only our empty scaffolding can be removed after a creation failure.
            remaining = list(dataset.iterdir())
            if all(path.name == ".material-workbench.lock" for path in remaining):
                for path in remaining:
                    path.unlink()
                dataset.rmdir()
        raise
    return refreshed(dataset, created=True, target=getattr(args, "target", "height"))


def edit_dataset(args) -> dict:
    dataset = canonical_dataset(args.dataset)
    check_idle(dataset)
    with dataset_lock(dataset):
        index, _records = editable_index(dataset, args)
        if getattr(args, "name", None) is not None:
            index["name"] = checked_name(args.name)
        if getattr(args, "description", None) is not None:
            index["description"] = checked_description(args.description)
        if getattr(args, "training_size", None) is not None:
            index["training_size"] = checked_training_size(args.training_size)
        index["updated_utc"] = timestamp()
        commit_metadata(dataset, [(dataset / "dataset.json", index)])
    return refreshed(dataset, review_size=getattr(args, "review_size", None), target=getattr(args, "target", "height"))


def material_identifier(name: str) -> str:
    from material_dataset import slugify
    try:
        return slugify(name)
    except ValueError:
        # Keep Unicode display names usable while retaining canonical path IDs.
        return "material_" + hashlib.sha256(name.encode()).hexdigest()[:16]


def source_material(name: str, paths: dict[str, Path], normal_convention="opengl") -> dict:
    from material_dataset import source_summary
    name = checked_name(name)
    if "input" not in paths or not any(role in paths for role in ("height", "normal", "roughness")):
        raise ValueError("Choose a color/input map and at least one height, normal, or roughness map")
    family = material_identifier(name)
    maps = {}
    for role, path in paths.items():
        path = path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"The {role} source map is missing: {path}")
        source = source_summary(path)
        if role == "input" and source["channels"] not in (3, 4):
            raise ValueError("Color/input maps must contain native RGB or RGBA channels")
        if role == "normal" and source["channels"] not in (3, 4):
            raise ValueError("Normal maps must contain native RGB or RGBA channels")
        suffix = {"input": "diff", "height": "disp", "roughness": "rough", "normal": "nor_dx" if normal_convention == "directx" else "nor_gl"}[role]
        maps[role] = dict(source, path=str(path), suffix=suffix, source_family_id=family, asset_family_id=family)
    dimensions = {(source["width"], source["height"]) for source in maps.values()}
    if len(dimensions) != 1:
        raise ValueError("All maps in a material must have exactly the same native dimensions; no resizing is applied")
    width, height = next(iter(dimensions))
    resolution = f"{width // 1024}k" if width == height and width >= 1024 and width % 1024 == 0 else f"{width}x{height}"
    for source in maps.values():
        source["resolution_label"] = resolution
    maps["input"]["variant_id"] = "color_default"
    return {"name": name, "material_id": f"{family}_{resolution}", "source_family_id": family,
        "source_set_id": f"{family}_{width}x{height}", "source_directory": str(Path(maps["input"]["path"]).parent),
        "common_pixel_dimensions": [width, height], "resolution_label": resolution, "maps": maps,
        "input_variants": [maps["input"]], "source_files": list(maps.values()), "warnings": [], "ignored_files": [], "problems": []}


def source_record(material: dict) -> dict:
    from material_dataset import GENERATOR, describe_encoding
    width, height = material["common_pixel_dimensions"]
    sources = [*material["maps"].values(), *material["input_variants"]]
    if any((source.get("width"), source.get("height")) != (width, height) for source in sources):
        raise ValueError("All maps and color variants must have exactly the same native dimensions; no resizing is applied")
    if any(source["channels"] not in (3, 4) for source in material["input_variants"]):
        raise ValueError("Diffuse maps must contain native RGB or RGBA channels")
    if "normal" in material["maps"] and material["maps"]["normal"]["channels"] not in (3, 4):
        raise ValueError("Normal maps must contain native RGB or RGBA channels")
    sample = {"schema_version": 2, "generator": GENERATOR, "sample_id": material["material_id"] + "_full",
        "name": material.get("name", material["source_family_id"]), "material_id": material["material_id"],
        "asset_family_id": material["source_family_id"], "source_family_id": material["source_family_id"],
        "source_set_id": material["source_set_id"], "source_directory": material["source_directory"],
        "source_resolution_label": material["resolution_label"], "sample_pixel_dimensions": [width, height],
        "source_pixel_dimensions": [width, height], "crop_rectangle_top_left_xywh": [0, 0, width, height],
        "status": "unreviewed", "split": "train", "review_status": "unreviewed", "maps": {},
        "map_metadata": {}, "input_variants": [], "source_precision_verified": True, "crop_values_verified": True,
        "source_files": material.get("source_files", list(material["maps"].values())),
        "source_discovery_warnings": material.get("warnings", []), "ignored_source_files": material.get("ignored_files", [])}
    for role, source in material["maps"].items():
        sample["maps"][role] = source["path"]
        sample["map_metadata"][role] = {"source": source, "storage": "source_reference", "filename": source["path"],
            "sample_sha256": source["file_sha256"], "sample_bits": source["sample_bits"], "channels": source["channels"],
            "encoding": describe_encoding(role, source), "transforms": [], "exact_source_crop": True}
    for source in material["input_variants"]:
        sample["input_variants"].append(dict(sample["map_metadata"]["input"], source=source,
            filename=source["path"], path=source["path"], sample_sha256=source["file_sha256"],
            sample_bits=source["sample_bits"], channels=source["channels"], variant_id=source["variant_id"],
            source_pixel_dimensions=[width, height], crop_rectangle_top_left_xywh=[0, 0, width, height]))
    sample["available_targets"] = [role for role in ("height", "roughness", "normal") if role in sample["maps"]
                                  and (role != "height" or sample["map_metadata"][role]["sample_bits"] >= 16)]
    return sample


def source_set_identity(record: dict) -> tuple:
    """Use file bindings, so aliases cannot register originals twice by name."""
    def binding(source):
        return str(Path(source.get("path", "")).resolve()), source.get("file_sha256")
    colors = {binding(record["map_metadata"]["input"].get("source", {}))} if "input" in record["map_metadata"] else set()
    colors.update(binding(variant.get("source", {})) for variant in record.get("input_variants", []))
    return (tuple(sorted((role, *binding(details.get("source", {})))
                         for role, details in record["map_metadata"].items())),
            tuple(sorted(colors)))


def register_materials(dataset: Path, args, materials: list[dict]) -> dict:
    """Publish references to originals as one recoverable metadata transaction."""
    if not materials:
        raise ValueError("No material maps found. Choose a folder containing paired color and height, normal, or roughness PNG maps.")
    check_idle(dataset)
    with dataset_lock(dataset):
        index, records = editable_index(dataset, args)
        expected_review = getattr(args, "expected_review_sha256", None)
        if expected_review and review_digest(dataset) != expected_review:
            raise ValueError("Dataset reviews changed since preview. Scan the folder again.")
        size = getattr(args, "training_size", None)
        size_changed = size is not None and index.get("training_size") != checked_training_size(size)
        if size is not None:
            index["training_size"] = size
        current = {sample["material_id"]: sample for _item, _path, sample in records}
        existing_sources = {source_set_identity(sample) for sample in current.values()}
        pending, duplicates = {}, 0
        for material in materials:
            if material.get("problems"):
                raise ValueError(f"Choose one source map per role for {material['material_id']}: {'; '.join(material['problems'])}")
            sample = source_record(material)
            source_identity = source_set_identity(sample)
            if source_identity in existing_sources:
                duplicates += 1
                continue
            prior = current.get(sample["material_id"]) or pending.get(sample["material_id"])
            if prior:
                raise ValueError(f"A different material named {sample['name']} already exists at this size. Choose another material name.")
            pending[sample["material_id"]] = sample
            existing_sources.add(source_identity)
        updates = []
        from material_native_size import CURATION_FIELDS, source_binding
        reviews_path = dataset / ".material-size-reviews.json"
        reviews = json.loads(reviews_path.read_text()) if reviews_path.is_file() else {}
        previous_reviews = dict(reviews)
        for sample in pending.values():
            folder = checked_relative(dataset, "samples/" + sample["sample_id"])
            prior_path = folder / "sample.json"
            previous_sample = json.loads(prior_path.read_text()) if prior_path.is_file() else None
            if previous_sample and source_binding(previous_sample) == source_binding(sample):
                # Restoring exact originals can restore their scientific review.
                sample.update({key: previous_sample[key] for key in CURATION_FIELDS if key in previous_sample})
            else:
                # A reused name is not proof that the pixels were reviewed.
                prefix = sample["material_id"] + "_"
                reviews = {key: value for key, value in reviews.items() if not key.partition(":")[2].startswith(prefix)}
            folder.mkdir(parents=True, exist_ok=True)
            updates.append((folder / "sample.json", sample))
            index["samples"].append({key: sample[key] for key in ("sample_id", "material_id", "status", "split")} | {"path": "samples/" + sample["sample_id"]})
        if pending or size_changed:
            # Explicitly importing a removed set restores its membership.
            index["removed_materials"] = [item for item in index.get("removed_materials", [])
                                         if item.get("material_id") not in pending]
            index["updated_utc"] = timestamp()
            if reviews != previous_reviews:
                updates.append((reviews_path, reviews))
            commit_metadata(dataset, [*updates, (dataset / "dataset.json", index)])
    return refreshed(dataset, review_size=getattr(args, "review_size", None), target=getattr(args, "target", "height"), added_material_count=len(pending), duplicate_material_count=duplicates)


def add_material(args) -> dict:
    dataset = canonical_dataset(args.dataset)
    paths = {role: getattr(args, role) for role in ("input", "height", "roughness", "normal") if getattr(args, role, None) is not None}
    return register_materials(dataset, args, [source_material(args.name, paths, getattr(args, "normal_convention", "opengl"))])


def import_inventory(folder: Path, dataset: Path) -> list[list]:
    """Inspect nested originals, excluding generated datasets and hidden caches."""
    inventory = []
    for directory, children, files in os.walk(folder):
        root = Path(directory)
        children[:] = sorted(name for name in children if not name.startswith(".")
            and (root / name).resolve() != dataset.resolve()
            and not (root / name / "dataset.json").is_file())
        for name in sorted(files):
            path = root / name
            if name.startswith(".") or path.suffix.casefold() != ".png":
                continue
            state = path.stat()
            inventory.append([str(path), str(path.resolve()), state.st_size, state.st_mtime_ns, state.st_ctime_ns])
    return inventory


def discover_import_materials(folder: Path, dataset: Path, source_cache: dict | None = None,
                              *, allow_invalid: bool = False) -> tuple[list[dict], list[str]]:
    from material_dataset import discover_source_sets, parse_source_filename, source_summary
    if not folder.is_dir():
        raise ValueError("Choose an existing folder of material maps")
    materials, warnings = [], []
    if (folder / "dataset.json").is_file():
        if folder == dataset:
            raise ValueError("This dataset is already open. Choose a different folder to import.")
        with dataset_lock(folder):
            _index, records = read_dataset(folder)
            for _entry, _path, sample in records:
                maps = {}
                for role, details in sample["map_metadata"].items():
                    if role not in ("input", "height", "normal", "roughness"):
                        continue
                    source = dict(details["source"])
                    path = Path(source["path"])
                    actual = source_summary(path)
                    if actual["file_sha256"] != source["file_sha256"]:
                        raise ValueError("An imported original source changed since it was registered")
                    maps[role] = {**source, **actual}
                variants = []
                for details in sample.get("input_variants", []):
                    source = dict(details["source"])
                    actual = source_summary(Path(source["path"]))
                    if actual["file_sha256"] != source["file_sha256"]:
                        raise ValueError("An imported color source changed since it was registered")
                    variants.append({**source, **actual, "variant_id": details["variant_id"]})
                if not variants:
                    variants = [dict(maps["input"], variant_id="color_default")]
                materials.append({"name": sample.get("name", sample["material_id"]), "material_id": sample["material_id"],
                    "source_family_id": sample.get("source_family_id", sample["material_id"]),
                    "source_set_id": sample.get("source_set_id", sample["material_id"]),
                    "source_directory": str(Path(maps["input"]["path"]).parent),
                    "common_pixel_dimensions": [maps["input"]["width"], maps["input"]["height"]],
                    "resolution_label": sample.get("source_resolution_label", "native"), "maps": maps, "input_variants": variants})
    else:
        # Accept the asset folder itself, a provider's sources root, or an
        # enclosing download folder. Only recognized paired map sets register.
        directories = sorted({Path(item[0]).parent for item in import_inventory(folder, dataset)}, key=str)
        provider_directories = [path for path in directories if any(parse_source_filename(item.name) for item in path.iterdir() if item.is_file())]
        materials.extend(material for material in discover_source_sets(folder, source_cache, source_directories=provider_directories,
                         issues=warnings if allow_invalid else None, progress=lambda path, i, total: print(f"IPDE_PROGRESS:Inspecting material folder {i}/{total}: {path.name}", file=sys.stderr, flush=True))
                         if "input" in material["maps"] and any(role in material["maps"] for role in ("height", "normal", "roughness")))
        aliases = {"input": ("input", "color", "colour", "albedo", "basecolor", "base_color", "diffuse"),
                   "height": ("height", "displacement", "disp"), "roughness": ("roughness", "rough"),
                   "normal": ("normal", "normalgl", "normal_gl", "normaldx", "normal_dx")}
        for directory in directories:
            if directory in provider_directories:
                continue
            maps, convention, ambiguous = {}, "opengl", False
            for path in directory.iterdir():
                if not path.is_file() or path.suffix.casefold() != ".png":
                    continue
                role = next((role for role, names in aliases.items() if path.stem.casefold() in names), None)
                if role:
                    if role in maps:
                        message = f"Multiple {role} maps in {directory}; add this material and choose its maps explicitly"
                        if not allow_invalid:
                            raise ValueError(message)
                        warnings.append("Skipped: " + message)
                        ambiguous = True
                    maps[role] = path
                    if role == "normal" and path.stem.casefold() in ("normaldx", "normal_dx"):
                        convention = "directx"
            if not ambiguous and "input" in maps and any(role in maps for role in ("height", "roughness", "normal")):
                try:
                    materials.append(source_material(directory.name, maps, convention))
                except (OSError, ValueError) as error:
                    if not allow_invalid:
                        raise
                    warnings.append(f"Skipped {directory.name}: {error}")
    for material in materials:
        warnings.extend(f"{material['material_id']}: {warning}" for warning in material.get("warnings", []))
    return materials, warnings



def scan_folder(args) -> dict:
    from material_native_size import SUPPORTED_SIZES, dataset_region_plan
    dataset, folder = canonical_dataset(args.dataset), args.folder.expanduser().resolve()
    with dataset_lock(dataset):
        index, records = editable_index(dataset, args)
        index_sha256 = digest(dataset / "dataset.json")
        reviews_path = dataset / ".material-size-reviews.json"
        reviews = json.loads(reviews_path.read_text()) if reviews_path.is_file() else {}
        reviews_sha256 = review_digest(dataset)
    inventory = import_inventory(folder, dataset)
    source_cache = {source["path"]: source for _entry, _path, record in records
                    for source in [*record.get("source_files", []),
                                   *(d.get("source", {}) for d in record.get("map_metadata", {}).values()),
                                   *(d.get("source", {}) for d in record.get("input_variants", []))] if source.get("path")}
    materials, warnings = discover_import_materials(folder, dataset, source_cache, allow_invalid=True)
    if inventory != import_inventory(folder, dataset):
        raise ValueError("The source folder changed while scanning. Scan it again before importing.")
    identities = {source_set_identity(record[2]) for record in records}
    names = {record[2]["material_id"] for record in records}
    accepted, duplicates = [], 0
    for material in materials:
        if material.get("problems"):
            warnings.append(f"Skipped {material['material_id']}: {'; '.join(material['problems'])}")
            continue
        try:
            sample = source_record(material)
        except ValueError as error:
            warnings.append(f"Skipped {material['material_id']}: {error}")
            continue
        identity = source_set_identity(sample)
        if identity in identities:
            duplicates += 1
            continue
        if sample["material_id"] in names:
            warnings.append(f"Skipped {sample['material_id']}: a different map set already uses this name. Add it manually with a different name.")
            continue
        names.add(sample["material_id"])
        identities.add(identity)
        accepted.append(material)
    recognized = {str(Path(source["path"]).resolve()) for material in materials
                  for source in material.get("source_files", material["maps"].values())}
    ignored = sum(item[1] not in recognized for item in inventory)
    if ignored:
        warnings.append(f"{ignored} PNG files are not part of a recognized paired diffuse and surface map set.")
    if not materials:
        warnings.append("No paired PNG material maps were recognized. Choose a folder of named diffuse, displacement, roughness or normal maps, or add maps manually.")
    combined = records + [({}, dataset / "dataset.json", source_record(material)) for material in accepted]
    plans = {str(size): {target: {k: v for k, v in dataset_region_plan(combined, size, reviews, target).items() if k != "assignments"}
             for target in ("height", "roughness", "normal")} for size in SUPPORTED_SIZES}
    # The verified source metadata is reused only while every original's file
    # identity and stat still match. Changing resolution never rehashes the GBs.
    plan = {"schema": "texture-studio-folder-import-v1", "dataset_path": str(dataset), "folder_path": str(folder),
            "index_sha256": index_sha256, "review_sha256": reviews_sha256, "inventory": inventory, "materials": accepted}
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(args.plan, json_bytes(plan))
    return {"folder_path": str(folder), "plan_path": str(args.plan), "plan_sha256": digest(args.plan),
            "index_sha256": index_sha256, "source_set_count": len(materials), "added_material_count": len(accepted),
            "duplicate_material_count": duplicates, "ignored_file_count": ignored,
            "warnings": warnings, "plans": plans}


def import_folder(args) -> dict:
    dataset, folder = canonical_dataset(args.dataset), args.folder.expanduser().resolve()
    check_idle(dataset)
    plan_path = getattr(args, "plan", None)
    if plan_path:
        if digest(plan_path) != getattr(args, "expected_plan_sha256", None):
            raise ValueError("The import preview changed. Scan the folder again.")
        plan = json.loads(plan_path.read_text())
        if (plan.get("schema") != "texture-studio-folder-import-v1" or plan.get("folder_path") != str(folder)
                or plan.get("dataset_path") != str(dataset) or plan.get("index_sha256") != digest(dataset / "dataset.json")
                or plan.get("review_sha256") != review_digest(dataset)
                or plan.get("inventory") != import_inventory(folder, dataset)):
            raise ValueError("The dataset or original folder changed since preview. Scan the folder again.")
        materials = plan["materials"]
        args.expected_review_sha256 = plan["review_sha256"]
        if not getattr(args, "expected_index_sha256", None):
            args.expected_index_sha256 = plan["index_sha256"]
        for material in materials:
            for source in [*material["maps"].values(), *material["input_variants"]]:
                state = Path(source["path"]).stat()
                if source.get("source_stat") != {"size": state.st_size, "mtime_ns": state.st_mtime_ns, "ctime_ns": state.st_ctime_ns}:
                    raise ValueError("An original map changed since preview. Scan the folder again.")
    else:
        materials, _warnings = discover_import_materials(folder, dataset)
    return register_materials(dataset, args, materials)


def remove_material(args) -> dict:
    dataset = canonical_dataset(args.dataset)
    check_idle(dataset)
    with dataset_lock(dataset):
        index, records = editable_index(dataset, args)
        selected = [sample for _entry, _path, sample in records if sample["material_id"] == args.material]
        if not selected:
            raise ValueError("Select a material in the open dataset before removing it")
        index["samples"] = [item for item in index["samples"] if item["material_id"] != args.material]
        tombstones = index.setdefault("removed_materials", [])
        for sample in selected:
            removed = {"material_id": sample["material_id"], "source_set_id": sample.get("source_set_id")}
            if removed not in tombstones:
                tombstones.append(removed)
        index["updated_utc"] = timestamp()
        commit_metadata(dataset, [(dataset / "dataset.json", index)])
    return refreshed(dataset, review_size=getattr(args, "review_size", None), target=getattr(args, "target", "height"), removed_material_id=args.material, removed_sample_count=len(selected))


def validate_delete(args) -> dict:
    """Give the UI a conservative, original-preserving macOS Trash scope."""
    dataset = canonical_dataset(args.dataset)
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
    if dataset in protected or dataset.is_relative_to(ROOT / "scripts") or dataset.is_relative_to(ROOT / "src"):
        raise ValueError("This is a protected workspace folder, not a removable dataset")
    check_idle(dataset)
    with dataset_lock(dataset):
        index, records = editable_index(dataset, args)
        for stage in (dataset / ".training-data").glob(".preparing-*"):
            marker = stage / ".native-crop-preparation.json"
            if not marker.is_file():
                continue
            try:
                ownership = json.loads(marker.read_text())
                if ownership.get("source_dataset_path") != str(dataset) or type(ownership.get("pid")) is not int:
                    continue
                os.kill(ownership["pid"], 0)
            except (OSError, ValueError):
                continue
            raise ValueError("Finish or cancel active dataset preparation before deleting the dataset")
        owned = index.get("dataset_management", {})
        safe = owned.get("schema") == MANAGEMENT_SCHEMA and owned.get("owns_directory") is True
        for _entry, _metadata, sample in records:
            for details in [*sample.get("map_metadata", {}).values(), *sample.get("input_variants", [])]:
                source = details.get("source", {})
                if not source.get("path") or Path(source["path"]).resolve().is_relative_to(dataset):
                    safe = False
        # Ownership markers alone cannot authorize moving unrelated/new files.
        permitted = {"dataset.json", ".material-workbench.lock", ".material-size-reviews.json", ".DS_Store"}
        for path in dataset.rglob("*"):
            relative = path.relative_to(dataset)
            if path.is_symlink():
                safe = False
            if path.is_file() and not (str(relative) in permitted or (len(relative.parts) == 3 and relative.parts[0] == "samples" and relative.parts[-1] == "sample.json")):
                safe = False
        return {"dataset_path": str(dataset), "index_sha256": digest(dataset / "dataset.json"),
            "name": index.get("name", dataset.name), "safe_to_trash_folder": safe,
            "trash_paths": [str(dataset if safe else dataset / "dataset.json")],
            "original_sources_preserved": True, "source_bytes_modified": False,
            "deletion_scope": "owned_dataset_folder" if safe else "dataset_membership_only"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("dataset", "prepare-size", "cleanup-size", "remove-missing", "curate", "create-dataset", "edit-dataset", "add-material", "import-folder", "scan-folder", "remove-material", "validate-delete"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", type=Path, required=True)
        if name in ("dataset", "curate", "remove-missing", "create-dataset", "edit-dataset", "add-material", "import-folder", "remove-material"):
            sub.add_argument("--review-size", type=int, choices=(256, 512, 1024, 2048))
        if name in ("create-dataset", "edit-dataset"):
            sub.add_argument("--name", required=name == "create-dataset")
            sub.add_argument("--description", default="" if name == "create-dataset" else None)
            sub.add_argument("--training-size", type=int, choices=(256, 512, 1024, 2048))
        if name in ("edit-dataset", "add-material", "import-folder", "scan-folder", "remove-material", "validate-delete"):
            sub.add_argument("--expected-index-sha256")
        if name == "add-material":
            sub.add_argument("--name", required=True)
            sub.add_argument("--input", type=Path, required=True)
            for role in ("height", "normal", "roughness"):
                sub.add_argument("--" + role, type=Path)
            sub.add_argument("--normal-convention", choices=("opengl", "directx"), default="opengl")
        if name in ("import-folder", "scan-folder"):
            sub.add_argument("--folder", type=Path, required=True)
            sub.add_argument("--plan", type=Path, required=name == "scan-folder")
        if name == "import-folder":
            sub.add_argument("--expected-plan-sha256")
            sub.add_argument("--training-size", type=int, choices=(256, 512, 1024, 2048))
        if name in ("dataset", "create-dataset", "edit-dataset", "add-material", "import-folder", "remove-material"):
            sub.add_argument("--target", choices=("height", "roughness", "normal"), default="height")
        if name == "dataset":
            sub.add_argument("--default-review-size", type=int, choices=(256, 512, 1024, 2048))
        if name == "remove-material":
            sub.add_argument("--material", required=True)
        if name == "remove-missing":
            sub.add_argument("--sample", required=True)
            sub.add_argument("--path", type=Path, required=True)
            sub.add_argument("--expected-index-sha256")
        if name == "prepare-size":
            sub.add_argument("--size", type=int, choices=(256, 512, 1024, 2048), required=True)
            sub.add_argument("--expected-index-sha256")
            sub.add_argument("--expected-review-sha256")
            sub.add_argument("--automatic-validation", action="store_true")
            sub.add_argument("--target", choices=("height", "roughness", "normal"), default="height")
            sub.add_argument("--material")
        if name == "curate":
            sub.add_argument("--sample", required=True)
            sub.add_argument("--status", choices=("approved", "excluded", "unreviewed"), required=True)
            sub.add_argument("--split", choices=("train", "validation", "automatic"))
            sub.add_argument("--note")
            sub.add_argument("--expected-index-sha256")
            sub.add_argument("--expected-review-sha256")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = {"dataset": dataset_info, "prepare-size": prepare_size, "cleanup-size": cleanup_size,
                      "remove-missing": remove_missing, "curate": curate,
                      "create-dataset": create_dataset, "edit-dataset": edit_dataset,
                      "add-material": add_material, "import-folder": import_folder, "scan-folder": scan_folder,
                      "remove-material": remove_material, "validate-delete": validate_delete}[args.command](args)
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
