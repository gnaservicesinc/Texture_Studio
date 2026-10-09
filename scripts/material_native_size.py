"""Prepare exact native training crops from distinct original resolution sets.

Unchanged maps refer directly to their original files. Each changed native crop
is saved once in the final temporary training dataset. Numeric source codes are
copied exactly, without resampling, gamma, range stretching, or denoising.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from material_dataset import (GENERATOR, MAP_NAMES, describe_encoding, discover_source_sets, file_sha256,
    parse_source_filename, png_image_header, read_png, resolve_map_path, slugify,
    source_summary, write_json, write_png)

PREPARATION_SCHEMA = "texture-studio-native-crops-v1"
SOURCE_STORAGE = "original-source-references-v1"
STAGING_ROOT = ".training-data"
PREPARING_MARKER = ".native-crop-preparation.json"
CURATION_FIELDS = ("status", "split", "split_assignment", "review_status", "curation_note", "source_binding_sha256")
NOTICE = "Checks group source families; disjoint 8K regions measure known-material quality, not unseen-material generalization."
SUPPORTED_SIZES = (256, 512, 1024, 2048)


def source_binding(sample: dict) -> str:
    """Bind reviews to original map bytes, independently of crop size/path."""
    identities = {"maps": {role: details.get("source", {}).get("file_sha256")
                           for role, details in sample.get("map_metadata", {}).items()},
                  "colors": sorted({details.get("source", {}).get("file_sha256", "")
                                    for details in sample.get("input_variants", [])})}
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def current_review(reviews: dict, key: str, sample: dict) -> dict:
    review = reviews.get(key, {})
    return review if not review.get("source_binding_sha256") or review["source_binding_sha256"] == source_binding(sample) else {}


def saved_review(sample: dict) -> dict:
    return {key: sample.get(key) for key in CURATION_FIELDS} | {"source_binding_sha256": source_binding(sample)}


class NativePreparationError(ValueError):
    def __init__(self, message: str, problems: list[dict]):
        super().__init__(message)
        self.details = {"problems": problems, "source_images_modified": False}


def snapshot(dataset: Path, records: list[tuple[dict, Path, dict]]) -> dict:
    if not records:
        raise ValueError("Dataset has no original material maps")
    return {"dataset_path": str(dataset), "index_sha256": file_sha256(dataset / "dataset.json"),
            "sample_metadata_sha256": {str(path.relative_to(dataset)): file_sha256(path)
                                       for _item, path, _sample in records}}


def ensure_snapshot(dataset: Path, proof: dict) -> None:
    for name, checksum in {"dataset.json": proof["index_sha256"], **proof["sample_metadata_sha256"]}.items():
        if file_sha256(dataset / name) != checksum:
            raise ValueError("Dataset review metadata changed during preparation; reload and retry")


def _source_maps(sources: Path, requested: set[str] | None = None,
                 source_cache: dict[str, dict] | None = None) -> list[dict]:
    """Keep each native resolution/color set registered to its own originals."""
    materials = []
    for material in discover_source_sets(sources, source_cache):
        if requested is not None and not ({material["material_id"], material["source_family_id"]} & requested):
            continue
        maps = material["maps"]
        # Incomplete older sets remain usable for their actual output targets.
        # A normal/bump map never fabricates a missing displacement map.
        if "input" in maps and any(role in maps for role in ("height", "normal", "roughness")):
            if material["problems"]:
                raise ValueError(f"Ambiguous originals for {material['material_id']}: {'; '.join(material['problems'])}")
            materials.append(material)
    return materials


def _source_inventory(sources: Path) -> str:
    entries = []
    for folder in sorted(item for item in sources.iterdir() if item.is_dir()):
        for path in sorted(folder.iterdir()):
            sidecar = path.name.startswith("material-source") and path.suffix.casefold() == ".json"
            if not path.is_file() or (parse_source_filename(path.name) is None and not sidecar):
                continue
            state = path.stat()
            entries.append([str(path.relative_to(sources)), str(path.resolve()), state.st_size,
                            state.st_mtime_ns, state.st_ctime_ns])
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def ensure_source_index(dataset: Path, index: dict) -> dict:
    """Replace obsolete crop indexes with small manifests pointing at originals.

    This is the current data contract, not compatibility support for development
    caches. Existing review notes and exclusions remain recorded in the manifest.
    No image bytes are created, removed, or modified here.
    """
    if index.get("native_size_preparation"):
        return index
    sources = dataset / "sources"
    if not sources.is_dir():
        return index
    inventory = _source_inventory(sources)
    if (index.get("storage_policy") == SOURCE_STORAGE
            and index.get("source_discovery") == "registered-resolution-and-color-sets-v2"
            and index.get("source_inventory_sha256") == inventory):
        return index
    prior = {}
    source_cache = {}
    for entry in index.get("samples", []):
        record = dict(entry)
        path = dataset / entry["path"] / "sample.json"
        if path.is_file():
            record.update(json.loads(path.read_text()))
        for details in [*record.get("map_metadata", {}).values(), *record.get("input_variants", [])]:
            source = details.get("source", {})
            if source.get("path"):
                source_cache[source["path"]] = source
        for source in record.get("source_files", []):
            if source.get("path"):
                source_cache[source["path"]] = source
        prior.setdefault(entry["material_id"], []).append(record)
    originals = _source_maps(sources, source_cache=source_cache)
    inspected_sources = {source["path"]: source["file_sha256"] for material in originals
                         for source in material.get("source_files", [])}
    tombstones = index.get("removed_missing_sources", [])
    suppressed = {(entry.get("source_set_id"), entry.get("material_id")) for entry in tombstones
        if inspected_sources.get(entry.get("path")) != entry.get("file_sha256")}
    originals = [material for material in originals if not any(
        set_id == material["source_set_id"] or material_id == material["material_id"]
        for set_id, material_id in suppressed)]
    removed_materials = index.get("removed_materials", [])
    originals = [material for material in originals if not any(
        item.get("material_id") == material["material_id"] or (item.get("source_set_id")
            and item["source_set_id"] == material["source_set_id"]) for item in removed_materials)]
    color_tombstones = index.get("removed_missing_color_variants", [])
    for material in originals:
        material["input_variants"] = [variant for variant in material["input_variants"]
            if not any((entry.get("source_set_id") == material["source_set_id"]
                        or entry.get("material_id") == material["material_id"])
                       and entry.get("variant_id") == variant["variant_id"]
                       and entry.get("file_sha256") != variant["file_sha256"]
                       for entry in color_tombstones)]
    records = []
    matched_prior = set()
    for material in originals:
        identity = material["material_id"] + "_full"
        decisions = prior.get(material["material_id"], [])
        source_dimensions = {(m["width"], m["height"]) for m in material["maps"].values()}
        if len(source_dimensions) != 1:
            continue
        width, height = next(iter(source_dimensions))
        if not decisions:
            # Preserve review metadata for the same original grid when adding
            # resolution suffixes; new resolution sets begin unreviewed.
            decisions = [record for records in prior.values() for record in records
                if record.get("source_pixel_dimensions") == [width, height]
                and (record.get("source_family_id", record.get("material_id")) == material["source_family_id"]
                    or any(details.get("source", {}).get("filename") in {s["filename"] for s in material["source_files"]}
                           for details in record.get("map_metadata", {}).values()))]
        full = next((s for s in decisions if s.get("crop_rectangle_top_left_xywh") == [0, 0, width, height]), None)
        if full:
            matched_prior.add(full["sample_id"])
        # A crop approval cannot approve previously unseen parts of a full map.
        excluded = any(s.get("status") in ("excluded", "rejected") for s in decisions)
        sample = {"schema_version": 2, "generator": GENERATOR, "sample_id": identity,
                  "material_id": material["material_id"], "source_directory": material["source_directory"],
                  "source_family_id": material["source_family_id"], "asset_family_id": material["source_family_id"],
                  "source_set_id": material["source_set_id"], "source_resolution_label": material["resolution_label"],
                  "sample_pixel_dimensions": [width, height], "source_pixel_dimensions": [width, height],
                  "crop_rectangle_top_left_xywh": [0, 0, width, height], "maps": {}, "map_metadata": {},
                  "source_precision_verified": True, "crop_values_verified": True,
                  "split": full.get("split", "train") if full else "train",
                  "status": full.get("status", "unreviewed") if full else "excluded" if excluded else "unreviewed",
                  "review_status": full.get("review_status", "unreviewed") if full else "unreviewed",
                  "curation_note": full.get("curation_note") if full else "\n".join(s["curation_note"] for s in decisions if s.get("curation_note")) or None,
                  "prior_crop_decisions": [{k: s.get(k) for k in ("sample_id", "status", "split", "curation_note", "crop_rectangle_top_left_xywh")} for s in decisions]}
        sample["input_variants"] = []
        for role, source in material["maps"].items():
            sample["maps"][role] = source["path"]
            sample["map_metadata"][role] = {"source": source, "storage": "source_reference",
                "filename": source["path"], "sample_sha256": source["file_sha256"],
                "sample_bits": source["sample_bits"], "channels": source["channels"],
                "encoding": describe_encoding(role, source),
                "transforms": [], "exact_source_crop": True}
        for source in material["input_variants"]:
            sample["input_variants"].append({"variant_id": source["variant_id"], "path": source["path"],
                "filename": source["path"], "source": source, "storage": "source_reference",
                "sample_sha256": source["file_sha256"], "sample_bits": source["sample_bits"],
                "channels": source["channels"], "encoding": describe_encoding("input", source),
                "source_pixel_dimensions": [width, height], "crop_rectangle_top_left_xywh": [0, 0, width, height],
                "transforms": [], "exact_source_crop": True})
        if full and index.get("storage_policy") == SOURCE_STORAGE:
            # Existing source checksums are immutable evidence. A missing or
            # changed source must still fail inspection/training against that
            # evidence; a discovery refresh cannot silently accept new bytes.
            for role, details in full.get("map_metadata", {}).items():
                if details.get("storage") == "source_reference":
                    retained = copy.deepcopy(details)
                    discovered = sample["map_metadata"].get(role, {}).get("source", {})
                    if discovered.get("file_sha256") == retained["source"].get("file_sha256"):
                        retained["source"] = {**retained["source"], **discovered}
                        retained["filename"] = retained["source"]["path"]
                    sample["map_metadata"][role] = retained
                    sample["maps"][role] = retained["source"]["path"]
            variants = {v["variant_id"]: v for v in sample["input_variants"]}
            for variant in full.get("input_variants", []):
                retained = copy.deepcopy(variant)
                discovered = variants.get(variant["variant_id"], {})
                if discovered.get("sample_sha256") == retained.get("sample_sha256"):
                    retained.update(source={**retained["source"], **discovered["source"]},
                                    path=discovered["path"], filename=discovered["filename"])
                variants[variant["variant_id"]] = retained
            if not full.get("input_variants") and "input" in full.get("map_metadata", {}):
                details = copy.deepcopy(full["map_metadata"]["input"])
                suffix = details.get("source", {}).get("suffix", "diff")
                from material_dataset import color_variant_id
                details.update(variant_id=color_variant_id(suffix), path=full["maps"]["input"])
                variants[details["variant_id"]] = details
            sample["input_variants"] = sorted(variants.values(), key=lambda v: v["variant_id"])
        if full:
            sample = {**copy.deepcopy(full), **sample}
        sample["available_targets"] = [role for role in ("height", "roughness", "normal")
            if role in sample["map_metadata"] and (role != "height" or sample["map_metadata"][role]["source"]["sample_bits"] >= 16)]
        sample["source_discovery_warnings"] = material["warnings"]
        sample["source_files"] = material["source_files"]
        sample["ignored_source_files"] = material["ignored_files"]
        folder = dataset / "samples" / identity
        folder.mkdir(parents=True, exist_ok=True)
        metadata_path = folder / "sample.json"
        if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != sample:
            write_json(metadata_path, sample)
        records.append(sample)
    if index.get("storage_policy") == SOURCE_STORAGE:
        for decisions in prior.values():
            for record in decisions:
                if record["sample_id"] in matched_prior:
                    continue
                if any(details.get("storage") != "source_reference" for details in record.get("map_metadata", {}).values()):
                    continue
                # Keep unavailable originals in the inspection list until the
                # explicit missing-source action removes their bound entry.
                records.append(record)
    if not records:
        if tombstones or removed_materials or not index.get("samples"):
            return index
        raise ValueError("No original registered material maps remain in the source folder")
    prior_order = {entry["sample_id"]: number for number, entry in enumerate(index.get("samples", []))}
    records.sort(key=lambda record: (prior_order.get(record["sample_id"], len(prior_order)), record["sample_id"]))
    rebuilt = {**index, "schema_version": 2, "generator": GENERATOR, "storage_policy": SOURCE_STORAGE,
               "source_discovery": "registered-resolution-and-color-sets-v2", "source_inventory_sha256": inventory,
               "source_directory": str(sources.resolve()), "source_images_modified": False,
               "original_sources_required_for_training": True, "split_strategy": "whole-material-v1",
               "validation_scope": "held_out_materials", "samples": [{k: s[k] for k in ("sample_id", "material_id", "split", "status")} | {"path": "samples/" + s["sample_id"]} for s in records]}
    if tombstones:
        rebuilt["removed_missing_sources"] = tombstones
    if color_tombstones:
        rebuilt["removed_missing_color_variants"] = color_tombstones
    if (index.get("source_discovery") == rebuilt["source_discovery"]
            and index.get("samples") == rebuilt["samples"]):
        # Source changes alone do not mutate review/index identity. This is
        # especially important for the checksum-bound missing-source action.
        return index
    write_json(dataset / "dataset.json", rebuilt)
    return rebuilt


def recover_original(source: dict, dataset: Path) -> Path | None:
    """Find an intact recorded original after the source directory moved."""
    expected = source.get("file_sha256")
    current = Path(source.get("path", ""))
    if current.is_file() and file_sha256(current) == expected:
        return current
    roots = [dataset / "sources", Path(source.get("path", "")).parent]
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(root.rglob(source.get("filename") or current.name))
    for path in dict.fromkeys(candidates):
        if path.is_file() and file_sha256(path) == expected:
            return path.resolve()
    return None


def maps_have_native_dimensions(path: Path, sample: dict, size: int) -> bool:
    if sample.get("sample_pixel_dimensions") != [size, size] or not sample.get("maps"):
        return False
    try:
        return all([png_image_header(resolve_map_path(path.parent, sample, role))[axis]
                    for axis in ("width", "height")] == [size, size] for role in sample["maps"])
    except (OSError, ValueError, KeyError):
        return False


def crop_layout(width: int, height: int, size: int) -> list[tuple[str, list[int]]]:
    """Keep native pixels: one center crop, or three distinct 8K-style corners."""
    if min(width, height) < size:
        raise ValueError("Source cannot supply the selected original training detail")
    if (width, height) == (size, size):
        return [("full", [0, 0, size, size])]
    if min(width, height) >= 8192:
        # The unused bottom-left fourth corner leaves one complete region spare.
        return [("top_left", [0, 0, size, size]),
                ("top_right", [width - size, 0, size, size]),
                ("bottom_right", [width - size, height - size, size, size])]
    return [("center", [(width - size) // 2, (height - size) // 2, size, size])]


def source_review_records(records: list[tuple[dict, Path, dict]], size: int) -> list[tuple[dict, Path, dict]]:
    """Recreate crop identities for inspection after temporary pixels are purged."""
    result = []
    for entry, path, original in records:
        width, height = original["source_pixel_dimensions"]
        if min(width, height) < size:
            continue
        for region, rectangle in crop_layout(width, height, size):
            sample = copy.deepcopy(original)
            sample.update(source_record_sample_id=original["sample_id"],
                          sample_id=original["material_id"] + "_" + region,
                          sample_pixel_dimensions=[size, size], crop_rectangle_top_left_xywh=rectangle,
                          source_region_id=region,
                          source_normalized_rectangle_xywh=[rectangle[0] / width, rectangle[1] / height,
                                                            size / width, size / height])
            result.append((dict(entry, sample_id=sample["sample_id"]), path, sample))
    return result


def _verified_source(source: dict, dataset: Path, material_id: str, role: str) -> dict:
    source = copy.deepcopy(source)
    path = recover_original(source, dataset)
    if path is None:
        raise NativePreparationError("The original map is missing or changed; reconnect the recorded source.",
            [{"code": "source_missing" if not Path(source["path"]).is_file() else "source_changed",
              "material_id": material_id, "role": role, "path": source["path"],
              "action": "Restore the original source map; derived images cannot replace it."}])
    source["path"] = str(path)
    header = png_image_header(path)
    if any(header[key] != source[key] for key in ("width", "height", "channels", "sample_bits")):
        raise ValueError("Original source dimensions or precision disagree with metadata")
    return source


def _materials(records: list[tuple[dict, Path, dict]], dataset: Path, size: int | None = None) -> dict:
    materials = {}
    verified_sources = {}
    def verified(source, material_id, role):
        path = Path(source["path"])
        try:
            state = path.stat()
            key = (str(path.resolve()), source["file_sha256"], state.st_size, state.st_mtime_ns, state.st_ctime_ns)
        except OSError:
            return _verified_source(source, dataset, material_id, role)
        if key in verified_sources:
            return dict(copy.deepcopy(source), path=verified_sources[key])
        result = _verified_source(source, dataset, material_id, role)
        verified_sources[key] = result["path"]
        return result
    for _entry, metadata_path, sample in records:
        material_id = sample["material_id"]
        sources = [d.get("source", {}) for d in sample.get("map_metadata", {}).values()] + [d.get("source", {}) for d in sample.get("input_variants", [])]
        if size is not None and any(min(source.get("width", 0), source.get("height", 0)) < size for source in sources):
            continue
        if material_id != slugify(material_id):
            raise ValueError("Material identity must be canonical and contain no path components")
        if material_id in materials:
            continue
        material = {"sample": sample, "path": metadata_path, "maps": {}, "input_variants": []}
        for role, details in sample["map_metadata"].items():
            if role not in MAP_NAMES:
                continue
            source = verified(details["source"], material_id, role)
            material["maps"][role] = {**details, "source": source}
        dimensions = {(m["source"]["width"], m["source"]["height"]) for m in material["maps"].values()}
        if len(dimensions) != 1:
            raise ValueError(f"Original paired maps are not registered: {material_id}")
        material["dimensions"] = next(iter(dimensions))
        variants = sample.get("input_variants") or [dict(material["maps"]["input"], variant_id="default")]
        seen = set()
        for details in variants:
            source = verified(details["source"], material_id, "input")
            if (source["width"], source["height"]) != material["dimensions"]:
                raise ValueError("Color variations must use exactly the same source geometry")
            identity = source["file_sha256"]
            if identity in seen:
                continue
            seen.add(identity)
            material["input_variants"].append({**details, "source": source})
        material["family"] = sample.get("asset_family_id", sample.get("source_family_id", material_id))
        materials[material_id] = material
    return materials


def _review_key(size: int, sample: dict) -> str:
    return f"{size}:{sample['sample_id']}"


def _prepared_map(details: dict, role: str, rectangle: list[int], folder: Path, filename: str,
                  original_data: tuple[np.ndarray, dict] | None = None) -> dict:
    """Write one lossless native crop; unchanged original maps stay references."""
    source = details["source"]
    original_path = Path(source["path"])
    x, y, width, height = rectangle
    changed = rectangle != [0, 0, source["width"], source["height"]]
    transform_normal = role == "normal" and source.get("suffix") in ("nor_dx", "normal_dx")
    output_details = copy.deepcopy(details)
    transforms = []
    if changed:
        original, header = original_data if original_data is not None else read_png(original_path)
        if header["file_sha256"] != source["file_sha256"]:
            raise ValueError("Original source changed during native crop preparation")
        values = np.ascontiguousarray(original[y:y + height, x:x + width])
        if transform_normal:
            values = values.copy()
            values[..., 1] = np.iinfo(values.dtype).max - values[..., 1]
        output = folder / filename
        write_png(output, values, header, compression=3)
        returned, returned_header = read_png(output)
        if returned.dtype != values.dtype or not np.array_equal(returned, values):
            raise ValueError("Prepared map lossless export changed native numeric codes")
        path = output.name
        transforms.append({"type": "native_crop", "rectangle_top_left_xywh": rectangle,
                           "algorithm": "exact_native_integer_codes", "range_normalization": False,
                           "gamma_applied": False})
        output_details.update(storage="prepared_crop", sample_sha256=returned_header["file_sha256"],
                              sample_bits=returned_header["sample_bits"], channels=returned_header["channels"])
    else:
        path = str(original_path)
        output_details.update(storage="source_reference", sample_sha256=source["file_sha256"],
                              sample_bits=source["sample_bits"], channels=source["channels"])
    if transform_normal:
        transforms.append({"type": "directx_to_opengl", "component": "G", "applied_in_memory": not changed})
    output_details.update(path=path, filename=path, source=source, transforms=transforms,
                          source_pixel_dimensions=[source["width"], source["height"]],
                          crop_rectangle_top_left_xywh=list(rectangle),
                          exact_source_crop=True,
                          encoding=details.get("encoding", "source_srgb_assumed" if role == "input" else "linear_data"))
    return output_details


def prepare_from_records(dataset: Path, records: list[tuple[dict, Path, dict]], size: int,
                         automatic_validation: bool = False, validation_material: str | None = None,
                         review_records: list | None = None, target: str = "height") -> tuple[Path, dict]:
    if type(size) is not int or size not in SUPPORTED_SIZES:
        raise ValueError("Choose training map size 256, 512, 1024, or 2048")
    if target not in ("height", "roughness", "normal"):
        raise ValueError("Select a height, roughness, or normal training target")
    dataset = dataset.resolve()
    proof = snapshot(dataset, records)
    selected_records = [record for record in records if record[2]["material_id"] == validation_material] if validation_material else records
    if validation_material and not selected_records:
        raise ValueError("Selected training material is absent from the source dataset")
    materials = _materials(selected_records, dataset, size)
    if not materials:
        raise ValueError("No source set can supply the selected original training detail")
    review_path = dataset / ".material-size-reviews.json"
    reviews = json.loads(review_path.read_text()) if review_path.is_file() else {}
    for _entry, _path, sample in review_records or []:
        previous_size = sample.get("sample_pixel_dimensions", [size])[0]
        key = _review_key(previous_size, sample)
        current = current_review(reviews, key, sample)
        if (("source_review_snapshot" not in sample and not current)
                or ("source_review_snapshot" in sample and current == sample["source_review_snapshot"])):
            reviews[key] = saved_review(sample)
    if validation_material and validation_material not in materials:
        raise ValueError("Selected training material is absent from the source dataset")
    eligible_regions = {}
    for material_id, material in materials.items():
        eligible_regions[material_id] = [region for region, _rectangle in crop_layout(*material["dimensions"], size)
            if current_review(reviews, f"{size}:{material_id}_{region}", material["sample"]).get("status", material["sample"]["status"])
            not in ("excluded", "rejected")]
    target_eligible = {material_id for material_id, material in materials.items()
        if target in material["maps"] and (target != "height" or material["maps"][target]["source"]["sample_bits"] == 16)
        and target in material["sample"].get("available_targets", [target])}
    included = [material_id for material_id in sorted(materials) if eligible_regions[material_id] and material_id in target_eligible]
    if validation_material and validation_material not in target_eligible:
        raise ValueError(f"Selected source set does not supply a supported {target} training target")
    families = {}
    for material_id in included:
        families.setdefault(materials[material_id]["family"], []).append(material_id)
    source_sets_by_family = {}
    for _entry, _path, record in records:
        family = record.get("asset_family_id", record.get("source_family_id", record["material_id"]))
        source_sets_by_family.setdefault(family, set()).add(record.get("source_set_id", record["material_id"]))
    manual_family_splits = {}
    for material_id, material in materials.items():
        family = material["family"]
        if len(source_sets_by_family.get(family, [])) < 2:
            continue
        decisions = [current_review(reviews, f"{size}:{material_id}_{region}", material["sample"])
                     for region, _rectangle in crop_layout(*material["dimensions"], size)]
        decisions.append(material["sample"])
        explicit = {review["split"] for review in decisions if review.get("split_assignment") == "manual"}
        if len(explicit) > 1:
            raise ValueError("Resolution/color siblings in a source family must share one manual training/validation assignment")
        if explicit:
            chosen = explicit.pop()
            if family in manual_family_splits and manual_family_splits[family] != chosen:
                raise ValueError("Resolution/color siblings in a source family must share one manual training/validation assignment")
            manual_family_splits[family] = chosen
    check_families = set()
    if automatic_validation:
        if validation_material:
            family = materials[validation_material]["family"]
            if len(families.get(family, [])) == 1 and len(eligible_regions[validation_material]) > 1:
                check_families = {family}
        elif len(families) > 1:
            count = max(1, int(len(families) * .05))
            check_families = set(sorted(families, key=lambda m: hashlib.sha256(m.encode()).digest())[:count])
        elif len(families) == 1 and len(families[next(iter(families))]) == 1:
            sole = families[next(iter(families))][0]
            if len(eligible_regions[sole]) > 1:
                check_families = set(families)
    regional_families = {family for family in check_families if len(families.get(family, [])) == 1
        and len(eligible_regions[families[family][0]]) > 1}
    regional_check_regions = {family: eligible_regions[families[family][0]][-1] for family in regional_families}
    # A full lower-resolution sibling overlaps every native high-resolution
    # region. Such families must be held out in full, never split by resolution.
    signature = hashlib.sha256(json.dumps({"schema": PREPARATION_SCHEMA, "size": size,
        "snapshot": proof, "checks": sorted(check_families), "regional_checks": regional_check_regions,
        "automatic_validation": automatic_validation, "validation_material": validation_material, "target": target,
        "reviews": reviews}, sort_keys=True).encode()).hexdigest()
    root = dataset / STAGING_ROOT
    root.mkdir(exist_ok=True)
    if root.is_symlink():
        raise ValueError("Training data directory must not be a symbolic link")
    destination = root / f"{size}-{signature[:20]}"
    for previous in sorted(root.iterdir()):
        if previous.is_symlink() or not previous.is_dir():
            continue
        marker = previous / PREPARING_MARKER
        if previous.name.startswith(".preparing-") and marker.is_file():
            try:
                ownership = json.loads(marker.read_text())
                owned = (ownership.get("schema") == PREPARATION_SCHEMA and ownership.get("source_dataset_path") == str(dataset)
                         and type(ownership.get("pid")) is int and ownership["pid"] > 0)
                if owned:
                    try:
                        os.kill(ownership["pid"], 0)
                    except ProcessLookupError:
                        shutil.rmtree(previous)
                        continue
                    except PermissionError:
                        pass
            except (OSError, ValueError, AttributeError):
                pass
        if previous != destination and (previous / "dataset.json").is_file():
            cleanup_prepared_dataset(previous)
    if review_path.is_file():
        reviews.update(json.loads(review_path.read_text()))
    root.mkdir(exist_ok=True)
    if destination.exists():
        try:
            validate_cached(destination, size, proof)
            return destination, _information(dataset, destination, size, proof, reused=True)
        except (OSError, ValueError, KeyError):
            cleanup_prepared_dataset(destination)
            reviews = json.loads(review_path.read_text()) if review_path.is_file() else reviews
            root.mkdir(exist_ok=True)
    required = 0
    for material in materials.values():
        for _region, rectangle in crop_layout(*material["dimensions"], size):
            if rectangle == [0, 0, *material["dimensions"]]:
                continue
            details = [d for role, d in material["maps"].items() if role != "input"] + material["input_variants"]
            required += size * size * sum(d["source"]["channels"] * (d["source"]["sample_bits"] // 8) for d in details)
    if shutil.disk_usage(root).free < required + 64 * 1024**2:
        try:
            root.rmdir()
        except OSError:
            pass
        raise NativePreparationError("Insufficient storage for the selected native training crops.",
            [{"code": "insufficient_disk_space", "required_bytes": required + 64 * 1024**2,
              "path": str(root), "action": "Free space or choose a smaller trainable size."}])
    stage = Path(tempfile.mkdtemp(prefix=".preparing-", dir=root))
    prepared = []
    try:
        write_json(stage / PREPARING_MARKER, {"schema": PREPARATION_SCHEMA,
                                            "source_dataset_path": str(dataset), "pid": os.getpid()})
        for material_id in sorted(materials):
            material = materials[material_id]
            width, height = material["dimensions"]
            if min(width, height) < size:
                raise ValueError(f"{material_id} cannot supply {size}×{size} original training detail")
            material_samples = []
            for region, rectangle in crop_layout(width, height, size):
                sample = copy.deepcopy(material["sample"])
                identity = material_id + "_" + region
                family = material["family"]
                review = current_review(reviews, f"{size}:{identity}", sample)
                split = review.get("split", sample.get("split", "train"))
                manual = review.get("split_assignment") == "manual" or sample.get("split_assignment") == "manual"
                if family in manual_family_splits:
                    split = manual_family_splits[family]
                    manual = True
                elif review.get("split_assignment") != "manual" and sample.get("split_assignment") == "manual":
                    split = sample["split"]
                if automatic_validation and not manual:
                    split = ("validation" if family in check_families and
                             (family not in regional_families or region == regional_check_regions[family]) else "train")
                sample.update(sample_id=identity, maps={}, map_metadata={}, input_variants=[], extra_maps={}, source_notes=[],
                    split=split, source_region_role=split, source_region_id=region,
                    asset_family_id=family, source_family_id=family,
                    source_set_id=sample.get("source_set_id", f"{family}_{width}x{height}"),
                    split_strategy="source-family-native-regions-v1",
                    validation_scope="known_disjoint_regions" if family in regional_families else "held_out_source_families",
                    sample_pixel_dimensions=[size, size], source_pixel_dimensions=[width, height],
                    crop_rectangle_top_left_xywh=rectangle,
                    source_normalized_rectangle_xywh=[rectangle[0] / width, rectangle[1] / height, size / width, size / height],
                    native_size_preparation={"schema": PREPARATION_SCHEMA, "source_dataset_path": str(dataset),
                        "target_resized": False, "target_cropped": rectangle != [0, 0, width, height]},
                    source_precision_verified=True, crop_values_verified=True)
                sample.update(review)
                sample["source_review_snapshot"] = copy.deepcopy(review)
                sample["split"] = split
                if manual:
                    sample["split_assignment"] = "manual"
                sample["source_binding_sha256"] = source_binding(material["sample"])
                folder = stage / "samples" / identity
                folder.mkdir(parents=True)
                material_samples.append((sample, folder, rectangle))
            primary_sha = material["maps"]["input"]["source"]["file_sha256"]
            map_jobs = [("input", details, MAP_NAMES["input"] if details["source"]["file_sha256"] == primary_sha
                         else f"input-variant-{number + 1}.png") for number, details in enumerate(material["input_variants"])]
            map_jobs.extend((role, details, MAP_NAMES[role]) for role, details in material["maps"].items() if role != "input")
            for role, details, filename in map_jobs:
                # Decode one original at a time, once for all native regions.
                # Three 8K crops never require three full map decodes or keeping
                # every source/color variant resident together.
                changed = any(rectangle != [0, 0, width, height] for _sample, _folder, rectangle in material_samples)
                original_data = read_png(Path(details["source"]["path"])) if changed else None
                for sample, folder, rectangle in material_samples:
                    returned = _prepared_map(details, role, rectangle, folder, filename, original_data)
                    if role == "input":
                        sample["input_variants"].append(returned)
                        if details["source"]["file_sha256"] == primary_sha:
                            sample["maps"]["input"] = returned["filename"]
                            sample["map_metadata"]["input"] = returned
                    else:
                        sample["maps"][role], sample["map_metadata"][role] = returned["filename"], returned
                del original_data
            for sample, folder, _rectangle in material_samples:
                if "input" not in sample["maps"]:
                    raise ValueError("Canonical color input is absent from the color variant set")
                write_json(folder / "sample.json", sample)
                prepared.append(sample)
        # Report the split actually applied after explicit user assignments.
        check_families = {sample["asset_family_id"] for sample in prepared
                          if sample["split"] == "validation" and sample["status"] not in ("excluded", "rejected")}
        train_families = {sample["asset_family_id"] for sample in prepared
                          if sample["split"] == "train" and sample["status"] not in ("excluded", "rejected")}
        regional_families = check_families & train_families
        from material_dataset import rectangles_overlap
        for family in regional_families:
            training = [sample for sample in prepared if sample["asset_family_id"] == family and sample["split"] == "train"]
            validation = [sample for sample in prepared if sample["asset_family_id"] == family and sample["split"] == "validation"]
            if any(rectangles_overlap(check["crop_rectangle_top_left_xywh"], train["crop_rectangle_top_left_xywh"])
                   for check in validation for train in training):
                raise ValueError("Validation cannot overlap training pixels from the same source family")
        regional_check_regions = {family: next(sample["source_region_id"] for sample in prepared
                                              if sample["asset_family_id"] == family and sample["split"] == "validation")
                                  for family in regional_families}
        for sample in prepared:
            sample["validation_scope"] = "known_disjoint_regions" if sample["asset_family_id"] in regional_families else "held_out_source_families"
            # Final scope depends on actual user-selected splits, so publish it
            # after reviewing the complete family rather than before it.
            write_json(stage / "samples" / sample["sample_id"] / "sample.json", sample)
        automatic = {"policy": "source-family-native-regions-v1", "fraction": .05, "target": target,
                     "source_family_ids": sorted(check_families), "regional_family_ids": sorted(regional_families),
                     "regional_check_regions": regional_check_regions,
                     "material_ids": sorted(m for m in materials if materials[m]["family"] in check_families),
                     "quick_fit_material_id": validation_material}
        derived = {"schema_version": 2, "generator": GENERATOR, "crop_size": size,
            "training_pixel_dimensions": [size, size], "split_strategy": "source-family-native-regions-v1",
            "validation_scope": "held_out_source_families_and_disjoint_known_regions" if regional_families else "held_out_source_families",
            "original_sources_required_for_training": True, "source_images_modified": False,
            "native_size_preparation": {"schema": PREPARATION_SCHEMA, "ephemeral": True,
                "source_dataset_path": str(dataset), "source_snapshot": proof, "target_resized": False,
                "target_cropped": any(s["native_size_preparation"]["target_cropped"] for s in prepared),
                "estimated_staging_bytes": required, "automatic_validation": automatic}, "automatic_validation": automatic,
            "samples": [{k: s[k] for k in ("sample_id", "material_id", "split", "status", "asset_family_id", "source_set_id")} |
                        {"path": "samples/" + s["sample_id"]} for s in prepared]}
        write_json(stage / "dataset.json", derived)
        validate_cached(stage, size, proof)
        ensure_snapshot(dataset, proof)
        if destination.exists():
            raise ValueError("Concurrent training dataset publication")
        (stage / PREPARING_MARKER).unlink()
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        try:
            root.rmdir()
        except OSError:
            pass
    return destination, _information(dataset, destination, size, proof, reused=False)


def _information(dataset: Path, prepared: Path, size: int, proof: dict, reused: bool) -> dict:
    index = json.loads((prepared / "dataset.json").read_text())
    return {"source_dataset_path": str(dataset), "prepared_dataset_path": str(prepared),
        "source_index_sha256": proof["index_sha256"], "crop_size": size, "reused": reused,
        "target_resized": False, "target_cropped": index["native_size_preparation"]["target_cropped"],
        "estimated_staging_bytes": index["native_size_preparation"].get("estimated_staging_bytes", 0),
        "original_dataset_modified": False, "split_lineage_changed": True,
        "cross_size_validation_notice": NOTICE, "ephemeral": True,
        "automatic_validation": index["automatic_validation"], "original_parents_required": True}


def validate_cached(dataset: Path, size: int, proof: dict) -> None:
    index = json.loads((dataset / "dataset.json").read_text())
    binding = index["native_size_preparation"]
    if binding.get("schema") != PREPARATION_SCHEMA or binding.get("source_snapshot") != proof or index.get("crop_size") != size:
        raise ValueError("Prepared training size/source identity differs")
    for item in index["samples"]:
        folder = dataset / item["path"]
        sample = json.loads((folder / "sample.json").read_text())
        if any(sample.get(k) != item.get(k) for k in ("sample_id", "material_id", "status", "split")) or not maps_have_native_dimensions(folder / "sample.json", sample, size):
            raise ValueError("Prepared map dimensions differ from the selected training grid")
        checked_paths = {}
        for role, details in sample["map_metadata"].items():
            path = resolve_map_path(folder, sample, role)
            checked_paths[path] = file_sha256(path)
            if checked_paths[path] != details["sample_sha256"]:
                raise ValueError("Prepared image bytes changed")
        for details in sample.get("input_variants", []):
            path = Path(details["filename"])
            if not path.is_absolute():
                path = folder / path
            if path not in checked_paths:
                checked_paths[path] = file_sha256(path)
            if checked_paths[path] != details["sample_sha256"]:
                raise ValueError("Prepared color variant bytes changed")
            header = png_image_header(path)
            if (header["width"], header["height"]) != (size, size):
                raise ValueError("Prepared color variant dimensions differ from the selected training grid")


def cleanup_prepared_dataset(dataset: Path) -> dict:
    """Purge only our temporary training files, preserving per-crop review metadata."""
    dataset = dataset.absolute()
    index_path = dataset / "dataset.json"
    if not index_path.is_file():
        return {"dataset_path": str(dataset), "removed": False}
    index = json.loads(index_path.read_text())
    binding = index.get("native_size_preparation", {})
    if binding.get("schema") != PREPARATION_SCHEMA or binding.get("ephemeral") is not True:
        return {"dataset_path": str(dataset), "removed": False}
    original = Path(binding["source_dataset_path"]).resolve()
    if dataset.parent != original / STAGING_ROOT or dataset.is_symlink() or dataset.parent.is_symlink():
        raise ValueError("Temporary training cleanup path differs from its owned source location")
    reviews_path = original / ".material-size-reviews.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.is_file() else {}
    for entry in index["samples"]:
        folder = dataset / entry["path"]
        if folder.is_symlink() or not folder.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Temporary sample path escapes owned training data")
        sample = json.loads((folder / "sample.json").read_text())
        key = _review_key(index["crop_size"], sample)
        current = current_review(reviews, key, sample)
        if (("source_review_snapshot" in sample and current != sample["source_review_snapshot"])
                or ("source_review_snapshot" not in sample and current)):
            # Source editing is authoritative when it changed after this cache
            # was prepared. Cleanup must not replace a newer user decision with
            # its old cached approval/note or automatic split assignment.
            continue
        reviews[key] = saved_review(sample)
    write_json(reviews_path, reviews)
    shutil.rmtree(dataset)
    try:
        dataset.parent.rmdir()
    except OSError:
        pass
    return {"dataset_path": str(dataset), "source_dataset_path": str(original), "removed": True,
            "source_images_modified": False, "reviews_preserved": True}
