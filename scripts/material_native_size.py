"""Prepare separately versioned native crops from an indexed material dataset.

This changes the sampled footprint, never the resolution of source pixels.
Size-specific validation is disjoint inside its own dataset; a previous size's
training footprint may overlap it and is recorded rather than called unseen.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from material_dataset import (GENERATOR, MAP_NAMES, REGION_SPLIT_STRATEGY,
    REGION_VALIDATION_SCOPE, file_sha256, heldout_regions, pixel_sha256,
    contained_path, read_png, rectangles_overlap, slugify, transformed_crop, write_json,
    write_png)

PREPARATION_SCHEMA = "texture-studio-native-material-size-v1"
NOTICE = ("Validation uses disjoint regions within this size-specific dataset. "
    "Earlier training at another crop size may have seen overlapping source "
    "pixels; cross-size warm starts do not establish unseen-region quality.")
CURATION_FIELDS = ("status", "split", "review_status", "source_region_role", "curation_note")


class NativePreparationError(ValueError):
    def __init__(self, message: str, problems: list[dict]):
        super().__init__(message)
        self.details = {"problems": problems, "original_dataset_modified": False,
                        "target_resized": False}


def snapshot(dataset: Path, records: list[tuple[dict, Path, dict]]) -> dict:
    if not records:
        raise ValueError("Dataset has no indexed material crops")
    for _item, path, sample in records:
        if json.loads(path.read_text()) != sample:
            raise ValueError("Dataset review metadata changed before preparation; reload and retry")
    return {"dataset_path": str(dataset), "index_sha256": file_sha256(dataset / "dataset.json"),
            "sample_metadata_sha256": {str(path.relative_to(dataset)): file_sha256(path)
                                       for _item, path, _sample in records}}


def ensure_snapshot(dataset: Path, proof: dict) -> None:
    paths = {"dataset.json": proof["index_sha256"], **proof["sample_metadata_sha256"]}
    for name, checksum in paths.items():
        if file_sha256(dataset / name) != checksum:
            raise ValueError("Dataset review metadata changed during preparation; reload and retry")


def problems_for_sources(records: list[tuple[dict, Path, dict]], size: int,
                         verify_files: bool = True) -> tuple[dict, list[dict]]:
    """Bind materials to recorded parent identities, without source discovery."""
    materials, problems, checked = {}, [], {}
    for _entry, _path, sample in records:
        material_id = sample["material_id"]
        if not isinstance(material_id, str) or not material_id or material_id != slugify(material_id):
            raise ValueError("Material identity must be canonical and contain no path components")
        if not isinstance(sample.get("sample_id"), str) or sample["sample_id"] != slugify(sample["sample_id"]):
            raise ValueError("Sample identity must be canonical and contain no path components")
        material = materials.setdefault(material_id, {"samples": [], "maps": {}})
        material["samples"].append(sample)
        material.setdefault("records", []).append((_path, sample))
        for role in sample["maps"]:
            if role not in MAP_NAMES:
                raise ValueError("Native re-preparation supports the declared diffuse/height/normal/roughness roles")
            details = sample["map_metadata"][role]
            source = details.get("source", {})
            key = (source.get("path"), source.get("file_sha256"))
            previous = material["maps"].get(role)
            if previous and (previous["source"].get("path"), previous["source"].get("file_sha256")) != key:
                problems.append({"code": "ambiguous_parent", "material_id": material_id,
                    "role": role, "action": "Use a dataset whose material crops share the same original parent maps."})
            elif previous and (previous.get("transforms", []) != details.get("transforms", []) or previous.get("encoding") != details.get("encoding")):
                problems.append({"code": "inconsistent_transform", "material_id": material_id,
                    "role": role, "action": "Review inconsistent encoding or convention metadata before preparing another size."})
            else:
                material["maps"][role] = details
            if key in checked:
                continue
            checked[key] = True
            issue = {"material_id": material_id, "role": role, "path": source.get("path")}
            if not isinstance(key[0], str) or not Path(key[0]).is_absolute() or not isinstance(key[1], str) or len(key[1]) != 64:
                problems.append(dict(issue, code="missing_parent_identity", action="Restore original parent path and SHA256 metadata; resized crops cannot replace an original."))
                continue
            if not verify_files:
                continue
            parent = Path(key[0])
            if not parent.is_file():
                problems.append(dict(issue, code="source_missing", action="Reconnect or restore the original parent file at the recorded path, then retry."))
            elif file_sha256(parent) != key[1]:
                problems.append(dict(issue, code="source_changed", expected_sha256=key[1], action="Restore the byte-identical original parent; changed source pixels are not accepted."))
    for material_id, material in materials.items():
        try:
            if not {"input", "height"} <= material["maps"].keys():
                raise ValueError("Paired diffuse and original displacement are required")
            dimensions = {(item["source"]["width"], item["source"]["height"]) for item in material["maps"].values()}
            if len(dimensions) != 1:
                raise ValueError("Original paired maps have different dimensions; registration needs review")
            if material["maps"]["height"]["source"].get("sample_bits") != 16:
                raise ValueError("Original height must retain 16-bit integer samples")
            width, height = next(iter(dimensions))
            material["regions"] = heldout_regions(width, height, size)
            material["dimensions"] = [width, height]
            for item in material["maps"].values():
                if any(t.get("type") not in ("directx_to_opengl", "srgb_to_linear", "gamma_to_linear") for t in item.get("transforms", [])):
                    raise ValueError("Recorded source transform is unsupported for native re-preparation")
        except (ValueError, KeyError, TypeError) as error:
            problems.append({"code": "native_regions_unavailable", "material_id": material_id,
                "message": str(error), "action": f"Use a larger registered original that can provide disjoint {size}×{size} regions, or retain the current crop size."})
    return materials, problems


def curation_for_region(samples: list[dict], rectangle: list[int]) -> dict:
    overlapping = [s for s in samples if rectangles_overlap(s["crop_rectangle_top_left_xywh"], rectangle)]
    exact = [s for s in overlapping if s["crop_rectangle_top_left_xywh"] == rectangle]
    decisions = exact or overlapping
    statuses = {s["status"] for s in decisions}
    if statuses & {"excluded", "rejected"}:
        status = "excluded"
    elif decisions and statuses <= {"approved", "accepted"}:
        # Approval of several small crops is not approval of unseen pixels in
        # a larger footprint. Only exactly reviewed source bounds inherit it.
        status = "approved" if exact else "unreviewed"
    else:
        status = "unreviewed"
    notes = [{"sample_id": s["sample_id"], "rectangle": s["crop_rectangle_top_left_xywh"],
              "split": s["split"], "status": s["status"], "note": s.get("curation_note"),
              "review_status": s.get("review_status")} for s in samples]
    text = "\n".join(f"{s['sample_id']}: {s['curation_note']}" for s in samples if s.get("curation_note"))
    return {"status": status, "review_status": "inherited_exclusion" if status == "excluded" else "inherited_exact_crop_approval" if status == "approved" else "native_footprint_needs_review",
            "curation_note": text or None, "prior_crop_decisions": notes}


def validate_cached(dataset: Path, size: int, proof: dict) -> None:
    index = json.loads((dataset / "dataset.json").read_text())
    binding = index.get("native_size_preparation", {})
    if binding.get("schema") != PREPARATION_SCHEMA or binding.get("source_snapshot") != proof or index.get("crop_size") != size:
        raise ValueError("Native-size cache identity changed; existing files were preserved")
    if index.get("automatic_validation") != binding.get("automatic_validation"):
        raise ValueError("Automatic check policy differs from its preparation identity")
    artifacts = json.loads((dataset / "native-size-artifacts.json").read_text())
    if artifacts.get("schema") != PREPARATION_SCHEMA or artifacts.get("source_snapshot") != proof:
        raise ValueError("Native-size artifact proof disagrees with its original source dataset")
    from train_material_height import verify_region_split
    records = []
    for item in index["samples"]:
        from material_dataset import contained_path
        folder = contained_path(dataset, item["path"], "Cached native crop")
        sample = json.loads((folder / "sample.json").read_text())
        if any(sample.get(k) != item.get(k) for k in ("sample_id", "material_id", "split", "status")) or sample.get("sample_pixel_dimensions") != [size, size]:
            raise ValueError("Cached native sample/index identity changed inconsistently")
        immutable = {k:v for k,v in sample.items() if k not in CURATION_FIELDS}
        immutable_hash = hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
        if artifacts["sample_immutable_sha256"].get(sample["sample_id"]) != immutable_hash:
            raise ValueError("Cached native sample scientific metadata changed")
        for role, filename in sample["maps"].items():
            path = contained_path(folder, filename, "Cached native map")
            if file_sha256(path) != sample["map_metadata"][role]["sample_sha256"]:
                raise ValueError(f"Cached native crop bytes changed: {path}; existing files were preserved")
            if artifacts["map_sha256"].get(str(path.relative_to(dataset))) != sample["map_metadata"][role]["sample_sha256"]:
                raise ValueError("Cached native crop no longer matches its immutable preparation proof")
        records.append({"metadata": sample})
    if {r["metadata"]["sample_id"] for r in records} != set(artifacts["sample_immutable_sha256"]):
        raise ValueError("Cached native crop membership changed")
    verify_region_split(records, index)


def recover_cache_reviews(dataset: Path, proof: dict) -> tuple[dict, dict]:
    """Recover only bound review metadata, never corrupt cached image bytes."""
    try:
        from material_dataset import contained_path
        index = json.loads((dataset / "dataset.json").read_text())
        artifacts = json.loads((dataset / "native-size-artifacts.json").read_text())
        if artifacts.get("schema") != PREPARATION_SCHEMA or artifacts.get("source_snapshot") != proof:
            raise ValueError("Source proof is missing or different")
        reviews, records = {}, []
        for entry in index["samples"]:
            folder = contained_path(dataset, entry["path"], "Prior cache review")
            record = json.loads((folder / "sample.json").read_text())
            immutable = {k:v for k,v in record.items() if k not in CURATION_FIELDS}
            if (artifacts["sample_immutable_sha256"].get(record["sample_id"]) != hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()
                    or any(record.get(k) != entry.get(k) for k in ("sample_id", "material_id", "status", "split"))
                    or record["status"] not in ("approved", "accepted", "excluded", "rejected", "prepared", "unreviewed")
                    or record["split"] not in ("train", "validation") or record.get("source_region_role") != record["split"]):
                raise ValueError("Scientific identity or review assignment is inconsistent")
            reviews[record["sample_id"]] = {"immutable_sha256": hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest(),
                "fields": {k:record.get(k) for k in CURATION_FIELDS}}
            records.append((entry, folder / "sample.json", record))
        if set(reviews) != set(artifacts["sample_immutable_sha256"]):
            raise ValueError("Prior crop membership is inconsistent")
        return reviews, snapshot(dataset, records)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NativePreparationError("Existing native cache needs review before repair; its files remain preserved.", [{"code": "cache_review_unrecoverable", "path": str(dataset), "message": str(error), "action": "Recover or review the cache metadata before rebuilding it, so exclusions and notes cannot be lost."}]) from error


def available_memory() -> int:
    """Read reclaimable memory without adding a runtime dependency."""
    import re
    import subprocess
    if sys.platform == "darwin":
        try:
            text = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+) bytes", text)[1])
            fields = {name:int(value) for name, value in re.findall(r"^(Pages [^:]+):\s+(\d+)", text, re.MULTILINE)}
            return page * sum(fields.get(name, 0) for name in ("Pages free", "Pages inactive", "Pages speculative"))
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            return 2*1024**3
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 2*1024**3


def worker_count(materials: dict, size: int, requested: int | None = None) -> int:
    available = available_memory()
    peak = max((max(d["source"]["width"] * d["source"]["height"] * d["source"]["channels"] * (d["source"]["sample_bits"] // 8) for d in material["maps"].values()) * 4 + size*size*64 for material in materials.values()), default=512*1024**2)
    budget = min(12*1024**3, available//3)
    hardware = max(1, min(8, os.cpu_count() or 1, len(materials), max(1, budget//max(peak, 1))))
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("Crop worker count must be positive")
    return hardware if requested is None else min(hardware, requested)


def copy_native_map(source: Path, destination: Path) -> None:
    """Independent APFS clone when available; never hardlink original maps."""
    if sys.platform == "darwin":
        import ctypes
        import errno
        library = ctypes.CDLL(None, use_errno=True)
        clone = library.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
        error = ctypes.get_errno()
        if error not in (errno.EXDEV, errno.ENOTSUP, errno.EINVAL, errno.ENOSYS):
            raise OSError(error, os.strerror(error), destination)
    with source.open("rb") as original, destination.open("xb") as target:
        shutil.copyfileobj(original, target, 1024*1024)
    shutil.copystat(source, destination)


def prepare_from_records(dataset: Path, index: dict, records: list[tuple[dict, Path, dict]], size: int,
                         cache_lock=None, automatic_validation: bool = False, validation_material: str | None = None,
                         workers: int | None = None, review_records: list | None = None) -> tuple[Path, dict]:
    """Caller holds the authoritative dataset lock throughout preparation."""
    if type(size) is not int or size not in (1024, 2048):
        raise ValueError("Choose native crop size 1024 or 2048")
    proof = snapshot(dataset, records)
    info = {"source_dataset_path": str(dataset), "source_index_sha256": proof["index_sha256"],
            "crop_size": size, "target_resized": False, "original_dataset_modified": False}
    if not automatic_validation and all(s["sample_pixel_dimensions"] == [size, size] for _i, _p, s in records):
        return dataset, dict(info, prepared_dataset_path=str(dataset), reused=True,
            split_lineage_changed=False, cross_size_validation_notice=None)
    materials, problems = problems_for_sources(records, size, verify_files=False)
    if problems:
        raise NativePreparationError("Native crops could not be prepared from the recorded originals. See the affected material and recovery action.", problems)
    overlay_by_region = {}
    for _entry, path, sample in review_records or []:
        material = materials.get(sample["material_id"])
        if material is None: continue
        if any(role not in material["maps"] or details["source"]["file_sha256"] != material["maps"][role]["source"]["file_sha256"]
               for role, details in sample["map_metadata"].items()):
            raise ValueError("Reviewed crop parent identity differs from its original dataset")
        material["records"].append((path, sample))
        decisions = [dict(previous, material_id=sample["material_id"],
                          crop_rectangle_top_left_xywh=previous["rectangle"], curation_note=previous.get("note"))
                     for previous in sample.get("prior_crop_decisions", [])]
        decisions.append(sample)
        for decision in decisions:
            if decision.get("review_status") not in ("user_approved", "user_excluded", "unreviewed"):
                continue
            rectangle = decision["crop_rectangle_top_left_xywh"]
            material["samples"] = [previous for previous in material["samples"] if previous["crop_rectangle_top_left_xywh"] != rectangle]
            material["samples"].append(decision)
            overlay_by_region[(sample["material_id"], tuple(rectangle))] = {key:decision.get(key) for key in ("material_id", "sample_id", "crop_rectangle_top_left_xywh", "status", "curation_note", "review_status")}
    overlay = [overlay_by_region[key] for key in sorted(overlay_by_region)]
    validation = None
    if automatic_validation:
        from material_validation_policy import plan
        validation = plan(materials, size, validation_material)
        info["automatic_validation"] = validation
    if index.get("split_strategy") != REGION_SPLIT_STRATEGY:
        raise NativePreparationError("Automatic crop-size preparation currently requires explicit disjoint-region validation.", [{"code": "unsupported_split_policy", "action": "Prepare a separate region-split dataset explicitly; whole-material splits are not silently changed."}])
    key = hashlib.sha256(json.dumps({"schema": PREPARATION_SCHEMA, "size": size, "snapshot": proof, "automatic_validation": validation, "png_compression": 3, "review_overlay": overlay}, sort_keys=True).encode()).hexdigest()
    cache_root = dataset / ".native-sizes"
    if cache_root.is_symlink():
        raise ValueError("Native-size cache directory must not be a symbolic link")
    stem = f"{size}-{key[:20]}"
    destination = cache_root / stem
    info.update(prepared_dataset_path=str(destination), split_lineage_changed=True,
        cross_size_validation_notice=NOTICE)
    cache_problems, prior_reviews, prior_review_snapshots = [], {}, []
    candidates = [destination, *sorted(cache_root.glob(stem + "-repair-*"))]
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError("Native-size cache destination must not be a symbolic link")
        if not candidate.exists():
            continue
        with cache_lock(candidate) if cache_lock is not None else contextlib.nullcontext():
            try:
                validate_cached(candidate, size, proof)
            except (OSError, ValueError, KeyError, TypeError) as error:
                cache_problems.append({"path": str(candidate), "message": str(error)})
                recovered, review_snapshot = recover_cache_reviews(candidate, proof)
                prior_review_snapshots.append((candidate, review_snapshot))
                for identity, review in recovered.items():
                    if identity in prior_reviews and prior_reviews[identity] != review:
                        raise NativePreparationError("Preserved native-cache reviews disagree; resolve them before rebuilding.", [{"code": "conflicting_cache_reviews", "sample_id": identity, "action": "Review the preserved cache versions and reconcile their curation metadata."}])
                    prior_reviews[identity] = review
                continue
        ensure_snapshot(dataset, proof)
        return candidate, dict(info, prepared_dataset_path=str(candidate), reused=True,
            original_parents_required=False, preserved_invalid_caches=cache_problems)
    # A valid self-contained cache does not need source parents. Creation or
    # repair does, and never falls back to enlarging a smaller crop.
    reused_parents = set()
    for material in materials.values():
        material["existing"] = {}
        for region in material["regions"]:
            exact = next(((path, sample) for path, sample in material["records"]
                          if sample["crop_rectangle_top_left_xywh"] == region["rectangle"]), None)
            if exact:
                material["existing"][tuple(region["rectangle"])] = exact
        if len(material["existing"]) != len(material["regions"]):
            for details in material["maps"].values():
                parent = details["source"]
                if not Path(parent["path"]).is_file() or file_sha256(Path(parent["path"])) != parent["file_sha256"]:
                    raise NativePreparationError("Native crops need their recorded originals. Restore the affected parents and retry.", [{"code":"source_missing" if not Path(parent["path"]).is_file() else "source_changed", "material_id":material["samples"][0]["material_id"], "role":next(role for role, value in material["maps"].items() if value is details), "path":parent["path"], "action":"Reconnect the byte-identical original parent map."}])
                reused_parents.add(parent["path"])

    if destination.exists():
        generation = 1
        while destination.with_name(stem + f"-repair-{generation:03d}").exists() or destination.with_name(stem + f"-repair-{generation:03d}").is_symlink():
            generation += 1
        destination = destination.with_name(stem + f"-repair-{generation:03d}")
        info["prepared_dataset_path"] = str(destination)
    cache_root.mkdir(exist_ok=True)
    # PNG can approach uncompressed size on photographic data. Check before
    # allocating/decompressing parents and report the required free space.
    estimate = sum(size * size * len(m["regions"]) * sum(d["source"]["channels"] * (d["source"]["sample_bits"] // 8) for d in m["maps"].values()) for m in materials.values())
    required = estimate + max(64 * 1024 * 1024, estimate // 20)
    if shutil.disk_usage(cache_root).free < required:
        raise NativePreparationError("Insufficient free space for a separate native-size dataset.", [{"code": "insufficient_disk_space", "required_bytes": required, "path": str(cache_root), "action": "Free storage for the derived crops; the original dataset will remain intact."}])
    stage = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache_root))
    output_records = []
    try:
        cancellation = threading.Event()
        started = time.monotonic()
        count = worker_count(materials, size, workers)
        info.update(worker_count=count, png_compression=3)
        print(json.dumps({"event":"native_preparation_started", "workers":count, "png_compression":3, "materials":len(materials)}), file=sys.stderr, flush=True)
        def prepare_material(material_id):
            material = materials[material_id]
            if cancellation.is_set(): raise InterruptedError("Crop preparation stopped")
            print(json.dumps({"event": "prepare_native_material", "material_id": material_id, "crop_size": size}), file=sys.stderr, flush=True)
            prepared = []
            for region in material["regions"]:
                sample = copy.deepcopy(material["records"][0][1])
                identity = f"{material_id}_auto_{region['ordinal']:03d}"
                sample.update(sample_id=identity, sample_pixel_dimensions=[size, size],
                    crop_rectangle_top_left_xywh=region["rectangle"], source_pixel_dimensions=material["dimensions"],
                    maps={}, extra_maps={}, map_metadata={}, source_notes=[],
                    split=region["split"], source_region_role=region["split"], source_region_name=region["region"],
                    split_strategy=REGION_SPLIT_STRATEGY, validation_scope=REGION_VALIDATION_SCOPE,
                    preparation_signature=key + ":" + identity,
                    native_size_preparation={"schema": PREPARATION_SCHEMA, "source_dataset_path": str(dataset), "source_index_sha256": proof["index_sha256"], "target_resized": False, "cross_size_validation_notice": NOTICE},
                    **curation_for_region(material["samples"], region["rectangle"]))
                folder = stage / "samples" / identity
                folder.mkdir(parents=True)
                prepared.append((folder, sample))
            for folder, sample in prepared:
                exact = material["existing"].get(tuple(sample["crop_rectangle_top_left_xywh"]))
                if exact:
                    path, previous = exact
                    for role, previous_filename in previous["maps"].items():
                        filename = MAP_NAMES[role]
                        source_path = contained_path(path.parent, previous_filename, "Existing native map")
                        expected = previous["map_metadata"][role]["sample_sha256"]
                        if file_sha256(source_path) != expected:
                            raise ValueError("Existing native crop bytes changed before copying")
                        copy_native_map(source_path, folder / filename)
                        if file_sha256(folder / filename) != expected:
                            raise ValueError("Native crop copy failed its checksum")
                        sample["maps"][role] = filename
                        sample["map_metadata"][role] = copy.deepcopy(previous["map_metadata"][role])
                        sample["map_metadata"][role]["filename"] = filename
            pending = [(folder, sample) for folder, sample in prepared if not sample["maps"]]
            for role, details in material["maps"].items():
                if not pending: break
                if cancellation.is_set(): raise InterruptedError("Crop preparation stopped")
                parent = details["source"]
                decoded, header = read_png(parent["path"])
                if header["file_sha256"] != parent["file_sha256"] or [header["width"], header["height"]] != material["dimensions"] or header["sample_bits"] != parent["sample_bits"]:
                    raise ValueError(f"Original parent changed during preparation: {parent['path']}")
                transfer = next((t for t in details.get("transforms", []) if t["type"] in ("srgb_to_linear", "gamma_to_linear")), None)
                override = {"mode": transfer["type"], "exponent": transfer.get("exponent")} if transfer else None
                for folder, sample in pending:
                    if cancellation.is_set(): raise InterruptedError("Crop preparation stopped")
                    x, y, width, height = sample["crop_rectangle_top_left_xywh"]
                    original = decoded[y:y+height, x:x+width]
                    crop, transforms, metadata = transformed_crop(original, role, {**parent, "color_chunks": header.get("color_chunks", [])}, override)
                    filename = MAP_NAMES.get(role, role + ".png")
                    write_png(folder / filename, crop, metadata)
                    returned, returned_header = read_png(folder / filename)
                    if returned.dtype != crop.dtype or not np.array_equal(returned, crop) or list(returned.shape[:2]) != [size, size]:
                        raise ValueError("Native crop integer round trip changed values or dimensions")
                    sample["maps"][role] = filename
                    sample["map_metadata"][role] = dict(details, filename=filename,
                        source=copy.deepcopy(parent), sample_bits=header["sample_bits"], channels=crop.shape[2], sample_dtype=str(crop.dtype),
                        sample_sha256=returned_header["file_sha256"], decoded_pixel_sha256=pixel_sha256(crop),
                        source_crop_pixel_sha256=pixel_sha256(original), transforms=transforms,
                        exact_source_crop=not transforms, transformed_source_crop_verified=True,
                        png_color_metadata={k:v for k,v in returned_header.items() if k not in ("color_chunks", "file_sha256")})
                del decoded
            for folder, sample in prepared:
                # Notes remain in the immutable source dataset. Carry their
                # original identities without duplicating large binary notes.
                sample["prior_source_notes"] = material["records"][0][1].get("source_notes", [])
                if sample["sample_id"] in prior_reviews:
                    immutable = {k:v for k,v in sample.items() if k not in CURATION_FIELDS}
                    prior = prior_reviews[sample["sample_id"]]
                    if hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest() != prior["immutable_sha256"]:
                        raise ValueError("Repair crop identity differs from preserved reviewed footprint")
                    sample.update(prior["fields"])
                write_json(folder / "sample.json", sample)
            return [sample for _folder, sample in prepared]
        pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="native-crop")
        futures = []
        try:
            futures = [pool.submit(prepare_material, identity) for identity in sorted(materials)]
            for completed, future in enumerate(as_completed(futures), 1):
                output_records.extend(future.result())
                print(json.dumps({"event":"native_material_complete", "completed":completed, "total":len(materials), "elapsed_seconds":round(time.monotonic()-started, 2)}), file=sys.stderr, flush=True)
        finally:
            cancellation.set()
            for future in futures: future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        output_records.sort(key=lambda sample: sample["sample_id"])
        info["elapsed_seconds"] = round(time.monotonic()-started, 3)
        derived_index = dict(index, crop_size=size, samples=[{k: s[k] for k in ("sample_id", "material_id", "split", "status")} | {"path": "samples/" + s["sample_id"]} for s in output_records],
            native_size_preparation={"schema": PREPARATION_SCHEMA, "source_dataset_path": str(dataset), "source_snapshot": proof, "target_resized": False, "cross_size_validation_notice": NOTICE, "automatic_validation": validation},
            original_sources_required_for_training=False, source_images_modified=False)
        if validation: derived_index["automatic_validation"] = validation
        write_json(stage / "dataset.json", derived_index)
        write_json(stage / "native-size-artifacts.json", {"schema": PREPARATION_SCHEMA,
            "source_snapshot": proof,
            "sample_immutable_sha256": {s["sample_id"]: hashlib.sha256(json.dumps({k:v for k,v in s.items() if k not in CURATION_FIELDS}, sort_keys=True).encode()).hexdigest() for s in output_records},
            "map_sha256": {"samples/" + s["sample_id"] + "/" + filename: s["map_metadata"][role]["sample_sha256"] for s in output_records for role, filename in s["maps"].items()}})
        validate_cached(stage, size, proof)
        ensure_snapshot(dataset, proof)
        # Rehash every parent immediately before publication, including roles
        # decoded early, so an external source edit never gets published.
        seen = set()
        for material in materials.values():
            for details in material["maps"].values():
                parent = details["source"]
                if parent["path"] in reused_parents and parent["path"] not in seen and file_sha256(Path(parent["path"])) != parent["file_sha256"]:
                    raise ValueError("Original parent changed before native crop publication")
                seen.add(parent["path"])
        with contextlib.ExitStack() as review_locks:
            for candidate, review_snapshot in sorted(prior_review_snapshots, key=lambda item:str(item[0])):
                if cache_lock is not None:
                    review_locks.enter_context(cache_lock(candidate))
                ensure_snapshot(candidate, review_snapshot)
            if destination.exists() or destination.is_symlink():
                raise ValueError("Concurrent native-size cache publication; no existing files overwritten")
            for path in stage.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for folder in [*sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p:len(p.parts), reverse=True), stage]:
                descriptor = os.open(folder, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            os.rename(stage, destination)
            descriptor = os.open(cache_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination, dict(info, reused=False, estimated_uncompressed_bytes=estimate,
        original_parents_required=bool(reused_parents), preserved_invalid_caches=cache_problems)
