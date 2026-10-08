#!/usr/bin/env python3
"""Refine PBRnxt's pretrained displacement decoder using native UInt16 pairs.

The 2048-pixel dataset can supply real 1024-pixel training crops. Neither model
inputs nor height targets are padded, resized, gamma-corrected or range-stretched.
This is an experimental checkpoint until full-resolution relief is reviewed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from material_pbrnxt import REVISION, WEIGHTS_SHA256, load_complete_pretrained, obtain_pretrained
from material_pbrnxt_data import PairCache, crop_pair, digest, height_loss, select_pairs, write_display, write_exr

SCHEMA = "texture-studio-pbrnxt-height-refinement-v1"
ROOT = Path(__file__).resolve().parents[1]
REFINEMENT_SCOPES = {
    "final-height": ("ups.3.",),
    "height-decoder": ("gen.m_dec_3.", "gen.m_tail_3.", "ups.3."),
}
DECODER_PREFIXES = REFINEMENT_SCOPES["final-height"]


def scope_prefixes(scope: str) -> tuple[str, ...]:
    if not isinstance(scope, str) or scope not in REFINEMENT_SCOPES:
        raise ValueError("Refinement scope must be final-height or height-decoder")
    return REFINEMENT_SCOPES[scope]


def checkpoint_scope(saved: dict) -> str:
    """Old v1 deltas contain only ups.3.; new ones explicitly bind their scope."""
    scope = saved.get("refinement_scope", "final-height")
    prefixes = scope_prefixes(scope)
    if "refinement_scope" not in saved:
        if "trainable_prefixes" in saved or "scope" in saved.get("configuration", {}):
            raise ValueError("Checkpoint refinement scope metadata is incomplete")
    elif (saved.get("trainable_prefixes") != list(prefixes)
          or saved.get("configuration", {}).get("scope") != scope):
        raise ValueError("Checkpoint refinement scope and permitted prefixes disagree")
    return scope


def resolve_scope(command: str, requested: str | None, previous: dict | None) -> str:
    previous_scope = checkpoint_scope(previous) if previous is not None else None
    scope = requested or previous_scope or "final-height"
    scope_prefixes(scope)
    if previous_scope == "height-decoder" and scope != previous_scope:
        raise ValueError("A height-decoder checkpoint cannot be narrowed and lose its learned generator state; omit --scope to inherit it")
    if command == "evaluate" and previous_scope is not None and scope != previous_scope:
        raise ValueError("Evaluate uses the checkpoint's recorded refinement scope; omit --scope")
    return scope


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def configure_refinement(model: torch.nn.Module, scope: str = "final-height") -> list[torch.nn.Parameter]:
    # Eval disables stochastic upstream noise and drop-path, not autograd.
    # Every trainable tensor belongs to the existing pretrained height decoder.
    prefixes = scope_prefixes(scope)
    named = list(model.named_parameters())
    if any(not any(name.startswith(prefix) for name, _parameter in named) for prefix in prefixes):
        raise ValueError("Pretrained height decoder is missing; no random replacement head is permitted")
    model.eval()
    selected = []
    for name, parameter in named:
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            selected.append(parameter)
    if not selected:
        raise ValueError("Pretrained height decoder is missing; no random replacement head is permitted")
    return selected


def resource_plan(size: int, memory_gib: float, cache_gib: float, scope: str = "final-height", *, inference: bool = False) -> dict:
    scope_prefixes(scope)
    if size < 256 or size % 64:
        raise ValueError("Choose a native crop of at least256 pixels, divisible by64; no padding or resizing")
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    if not math.isfinite(memory_gib) or memory_gib <= 0 or memory_gib > physical - 4:
        operation = "Inference" if inference else "Training"
        raise ValueError(f"{operation} budget must fit this Mac ({physical:.0f}GiB total,4GiB for macOS)")
    if not math.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("CPU cache budget must be finite and nonnegative")
    if inference:
        # A no-gradient forward does not retain training activations or allocate
        # optimizer state. Do not reject it using a measured backward-pass peak,
        # or invent a lower peak before measuring this architecture at this grid.
        # The positive MPS allocation cap remains authoritative for a real probe.
        driver_budget = memory_gib - cache_gib - 2
        if driver_budget < 1:
            raise ValueError("Inference budget must leave at least 1 GiB for Metal after the CPU cache and 2 GiB runtime reserve")
        return {"physical_gib": physical, "budget_gib": memory_gib, "cpu_cache_gib": cache_gib,
                "driver_budget_gib": driver_budget, "runtime_reserve_gib": 2,
                "driver_estimate_gib": None, "combined_estimate_gib": None,
                "refinement_scope": None, "operation": "inference", "peak_measured": False,
                "estimate_basis": "No-gradient inference; peak memory is unmeasured. The positive Metal allocation cap is retained; a native probe can still exhaust its budget."}
    # Measured on the64GiB M2 Max: complete native1024 final-height-branch
    # forward/backward allocated18.67GiB driver memory. Reserve optimizer,
    # CPU decoded-pair cache and modest runtime overhead separately.
    # The generator height decoder also retains upstream decoder/fusion
    # activations for gradients. A real1024 forward/backward/AdamW probe used
    # 34.28GiB driver memory; allow additional allocator/workspace overhead.
    driver_estimate = 2 + (17 if scope == "final-height" else 34) * (size / 1024) ** 2
    combined = driver_estimate + cache_gib + 1
    if combined > memory_gib:
        raise ValueError(f"Native{size} training estimates{combined:.1f}GiB above the{memory_gib:.1f}GiB budget. "
                         "Choose a smaller real training crop; keep the dataset unchanged. No padding/resizing is used.")
    return {"physical_gib": physical, "budget_gib": memory_gib, "cpu_cache_gib": cache_gib,
            "driver_estimate_gib": driver_estimate, "combined_estimate_gib": combined,
            "refinement_scope": scope,
            "estimate_basis": ("Measured1024 M2 Max complete native final-height-branch forward/backward; estimate, not a peak guarantee"
                               if scope == "final-height" else
                               "Measured1024 M2 Max native height-decoder forward/backward/AdamW34.28GiB plus allocator reserve; estimate, not a peak guarantee")}


def validate_checkpoint_metadata(saved: dict) -> str:
    if (not isinstance(saved, dict) or saved.get("schema") != SCHEMA or saved.get("base_sha256") != WEIGHTS_SHA256
            or saved.get("base_revision") != REVISION or saved.get("image_padding") is not False
            or saved.get("image_resizing") is not False or saved.get("target") != "height"):
        raise ValueError("Checkpoint is not this pinned native PBRnxt displacement refinement")
    size, configuration, step = saved.get("training_crop_size"), saved.get("configuration"), saved.get("step")
    if (type(size) is not int or size < 256 or size % 64
            or not isinstance(configuration, dict) or type(configuration.get("size")) is not int
            or configuration["size"] != size or type(step) is not int or step < 0):
        raise ValueError("Checkpoint crop grid, configuration or completed step is invalid")
    if (type(saved.get("whole_maps", False)) is not bool or type(configuration.get("whole_maps", False)) is not bool
            or ("whole_maps" in saved) != ("whole_maps" in configuration)
            or saved.get("whole_maps", False) != configuration.get("whole_maps", False)):
        raise ValueError("Checkpoint whole-map geometry policy and configuration disagree")
    return checkpoint_scope(saved)


def read_refinement_checkpoint(checkpoint: Path) -> dict:
    # A training process can atomically replace checkpoint.latest.pt while our
    # comparisons run. Bind identity to the exact bytes loaded, not the later
    # contents of its path. This private field is never saved into learned state.
    snapshot = checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(saved)
    saved["_snapshot_sha256"] = hashlib.sha256(snapshot).hexdigest()
    return saved


def restore_refinement(model: torch.nn.Module, checkpoint: Path, *, saved: dict | None = None) -> dict:
    saved = read_refinement_checkpoint(checkpoint) if saved is None else saved
    scope = validate_checkpoint_metadata(saved)
    state = saved.get("height_decoder_state")
    prefixes = scope_prefixes(scope)
    expected = {name for name in model.state_dict() if name.startswith(prefixes)}
    if not expected or not isinstance(state, dict) or set(state) != expected:
        raise ValueError("Checkpoint height decoder differs from the pretrained architecture")
    original = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != original[name].shape
                or tensor.dtype != original[name].dtype):
            raise ValueError("Checkpoint contains invalid pretrained decoder tensors")
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint contains invalid pretrained decoder tensors")
        if name in parameter_names:
            if not tensor.is_floating_point():
                raise ValueError("Checkpoint contains invalid pretrained decoder tensors")
        elif not torch.equal(tensor, original[name].cpu()):
            # SCUNet's positional layout includes Int64 indices and Float32
            # coordinate tables. Neither buffer represents learned values.
            raise ValueError("Checkpoint contains changed pretrained decoder layout buffers")
    original.update(state)
    model.load_state_dict(original, strict=True)
    return saved


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, report: dict, args: argparse.Namespace) -> None:
    scope = getattr(args, "scope", None) or "final-height"
    prefixes = scope_prefixes(scope)
    saved = {"schema": SCHEMA, "target": "height", "model_name": "PBRnxt material displacement refinement",
             "base_revision": REVISION, "base_sha256": WEIGHTS_SHA256,
             "step": step, "experimental": True, "production_eligible": False,
             "training_crop_size": args.size, "image_padding": False, "image_resizing": False,
             "whole_maps": getattr(args, "whole_maps", False),
             "input_transfer": "sRGB diffuse codes", "target_transfer": "UInt16 linear numeric codes /65535",
             "height_units": "relative source values", "normal_convention": "OpenGL +Y",
             "refinement_scope": scope, "trainable_prefixes": list(prefixes),
             "provenance": model.provenance, "dataset_identity": report["dataset"],
             "configuration": {**{name: getattr(args, name) for name in ("size", "seed", "learning_rate", "weight_decay", "memory_gib", "cache_gib")},
                               "scope": scope, "whole_maps": getattr(args, "whole_maps", False)},
             "height_decoder_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                                      if name.startswith(prefixes)},
             "optimizer_state": optimizer.state_dict()}
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(saved, temporary)
    temporary.replace(path)


def predict(model: torch.nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    output = model.height(rgb)
    if output.shape != (rgb.shape[0], 1, *rgb.shape[-2:]) or not torch.isfinite(output).all():
        raise ValueError("Model returned invalid native height; export stopped")
    return output


def reference_provenance(pair: dict, rectangle: list[int]) -> dict:
    """Keep full original-source identity separate from the two crop levels.

    This copies preparation evidence, not guessed URLs or inferred precision.
    The exported reference is still the exact Float32 transfer of the paired
    UInt16 crop, while source_path opens its full original parent when present.
    """
    metadata = pair["metadata"]
    details = metadata.get("map_metadata", {}).get("height", {})
    source = details.get("original_source") or details.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("Reference original-source provenance must be an object")
    result = {"dataset_sample_crop_rectangle": list(rectangle)}
    sample_path = pair.get("paths", {}).get("height")
    if sample_path is not None:
        result["dataset_sample_path"] = str(Path(sample_path).resolve())
    sample_checksum = pair.get("sha256", {}).get("height")
    if isinstance(sample_checksum, str) and re.fullmatch(r"[a-f0-9]{64}", sample_checksum):
        result["dataset_sample_sha256"] = sample_checksum
    bits = source.get("sample_bits", details.get("sample_bits"))
    if type(bits) is int and bits in (8, 16):
        result["source_bits"] = bits
    path = source.get("path", source.get("source_path"))
    if isinstance(path, str) and Path(path).is_absolute():
        result["source_path"] = path
    checksum = source.get("file_sha256", source.get("sha256"))
    if isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum):
        result["source_sha256"] = checksum
    dimensions = ([source["width"], source["height"]]
                  if "width" in source and "height" in source else metadata.get("source_pixel_dimensions"))
    parent = metadata.get("crop_rectangle_top_left_xywh")
    if parent is not None:
        if (not isinstance(parent, list) or len(parent) != 4 or len(rectangle) != 4
                or any(type(value) is not int for value in (*parent, *rectangle))
                or min(parent[:2]) < 0 or min(rectangle[:2]) < 0
                or min(parent[2:]) < 1 or min(rectangle[2:]) < 1
                or rectangle[0] + rectangle[2] > parent[2]
                or rectangle[1] + rectangle[3] > parent[3]):
            raise ValueError("Reference crop provenance leaves its real dataset parent crop")
        original_rectangle = [parent[0] + rectangle[0], parent[1] + rectangle[1], *rectangle[2:]]
        if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2
                or any(type(value) is not int or value < 1 for value in dimensions)
                or original_rectangle[0] + original_rectangle[2] > dimensions[0]
                or original_rectangle[1] + original_rectangle[3] > dimensions[1]):
            raise ValueError("Reference crop provenance leaves its full original source")
        result.update(source_crop_rectangle=original_rectangle,
                      source_pixel_dimensions=list(dimensions),
                      dataset_parent_source_crop_rectangle=list(parent))
    elif (isinstance(dimensions, (list, tuple)) and len(dimensions) == 2
          and all(type(value) is int and value > 0 for value in dimensions)):
        result["source_pixel_dimensions"] = list(dimensions)
    # Preparation attaches these fields only after the entire original file
    # matches the official API MD5/byte count. Do not guess from a material ID.
    url = source.get("published_url")
    parsed = urlparse(url) if isinstance(url, str) else None
    material = metadata.get("material_id")
    file_md5, published_md5 = source.get("file_md5"), source.get("published_md5")
    if (source.get("provider") == "Poly Haven" and source.get("license") == "CC0-1.0"
            and isinstance(material, str)
            and source.get("published_api_url") == f"https://api.polyhaven.com/files/{material}"
            and isinstance(file_md5, str) and re.fullmatch(r"[a-fA-F0-9]{32}", file_md5)
            and isinstance(published_md5, str) and file_md5.lower() == published_md5.lower()
            and type(source.get("file_bytes")) is int and source["file_bytes"] > 0
            and source["file_bytes"] == source.get("published_bytes")
            and "source_sha256" in result and parsed is not None
            and parsed.scheme == "https" and parsed.hostname == "dl.polyhaven.org"
            and parsed.username is None and parsed.password is None and parsed.port in (None, 443)
            and not parsed.query and not parsed.fragment
            and parsed.path.startswith("/file/ph-assets/Textures/")
            and Path(parsed.path).name == source.get("filename")):
        result["source_url"] = url
    package_url = source.get("published_package_url")
    package = urlparse(package_url) if isinstance(package_url, str) else None
    package_checksum = source.get("published_package_sha256")
    if (source.get("provider") == "ambientCG" and source.get("license") == "CC0-1.0"
            and "source_sha256" in result and source.get("published_member_sha256") == result["source_sha256"]
            and type(source.get("file_bytes")) is int and source["file_bytes"] > 0
            and source.get("published_member_bytes") == source["file_bytes"]
            and isinstance(package_checksum, str) and re.fullmatch(r"[a-f0-9]{64}", package_checksum)
            and type(source.get("published_package_bytes")) is int and source["published_package_bytes"] > 0
            and isinstance(source.get("published_archive_member"), str)
            and not Path(source["published_archive_member"]).is_absolute()
            and ".." not in Path(source["published_archive_member"]).parts
            and package is not None and package.scheme == "https" and package.hostname == "ambientcg.com"
            and package.username is None and package.password is None and package.port in (None, 443)
            and package.path == "/get" and not package.fragment
            and set(parse_qs(package.query)) == {"file"}
            and len(parse_qs(package.query)["file"]) == 1
            and re.fullmatch(r"[A-Za-z0-9]+_[1248]K-PNG\.zip", parse_qs(package.query)["file"][0])):
        result["source_url"] = package_url
    return result


def export_comparison(model: torch.nn.Module, pairs: list[dict], cache: PairCache, folder: Path,
                      label: str, device: torch.device, size: int, seed: int, whole_maps: bool = False) -> list[dict]:
    records = []
    model.eval()
    for pair in pairs:
        sample = pair["metadata"]["sample_id"]
        if not isinstance(sample, str) or sample in ("", ".", "..") or Path(sample).name != sample:
            raise ValueError("Comparison sample must have one safe directory name")
        directory = folder / sample
        directory.mkdir(parents=True, exist_ok=True)
        rgb, target, rectangle = crop_pair(*cache.load(pair), size, random.Random(seed), whole_maps=whole_maps)
        reference_identity = reference_provenance(pair, rectangle)
        native_dimensions = [rgb.shape[-1], rgb.shape[-2]]
        with torch.no_grad():
            height = predict(model, rgb.to(device)).cpu()
        _, metrics = height_loss(height, target, margin=64)
        prediction, reference = height.numpy()[0, 0], target.numpy()[0, 0]
        source_rgb = rgb.numpy()[0].transpose(1, 2, 0)
        source_identity = {
            "sample": sample, "native_size": max(native_dimensions), "native_dimensions": native_dimensions,
            "whole_maps": whole_maps, "source_rectangle": rectangle,
            "source_sha256": pair["sha256"],
            "reference_provenance": reference_identity,
            "input_float32_sha256": hashlib.sha256(np.ascontiguousarray(source_rgb).tobytes()).hexdigest(),
            "height_float32_sha256": hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
        }
        shared_names = {"reference-height.exr", "reference-height.png", "source.png", "reference-relief-8x.png"}
        identity_path = directory / "source-identity.json"
        if identity_path.exists():
            try:
                shared = json.loads(identity_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("Comparison source identity cannot be read; use a new folder") from error
            if (not isinstance(shared, dict) or shared.get("identity") != source_identity
                    or not isinstance(shared.get("files"), dict) or set(shared["files"]) != shared_names):
                raise ValueError("Comparison source/crop differs from existing shared reference; use a new folder")
            if any(not (directory / name).is_file() or digest(directory / name) != shared["files"][name]
                   for name in shared_names):
                raise ValueError("Comparison source/reference files changed; use a new folder")
        else:
            if any((directory / name).exists() for name in shared_names):
                raise ValueError("Existing shared comparison files have no verified source identity; use a new folder")
            write_exr(directory / "reference-height.exr", reference)
            write_display(directory / "reference-height.png", reference)
            write_display(directory / "source.png", source_rgb)
            write_display(directory / "reference-relief-8x.png", .5 + 8 * (reference - reference.mean()))
            write_json(identity_path, {"identity": source_identity,
                                       "files": {name: digest(directory / name) for name in sorted(shared_names)}})
        write_exr(directory / f"{label}-height.exr", prediction)
        write_display(directory / f"{label}-height.png", prediction)
        # Same explicit relief contrast for every model and reference, useful
        # when source height offsets differ. Never applied to EXR/training data.
        write_display(directory / f"{label}-relief-8x.png", .5 + 8 * (prediction - prediction.mean()))
        record = {"sample": sample, "model_label": label, "native_size": max(native_dimensions),
                  "native_dimensions": native_dimensions, "maximum_edge": size, "whole_maps": whole_maps,
                  "source_rectangle": rectangle,
                  "height_min": float(prediction.min()), "height_max": float(prediction.max()),
                  "clipped_display_fraction": float(((prediction < 0) | (prediction > 1)).mean()),
                  "metrics": metrics, "height_exr": str((directory / f"{label}-height.exr").resolve()),
                  "reference_exr": str((directory / "reference-height.exr").resolve()),
                  "reference_provenance": reference_identity,
                  "source_png": str((directory / "source.png").resolve()),
                  "note": "Metrics diagnose fitting only. Inspect native relief, inversions, noise and grain visually."}
        records.append(record)
    return records


def write_review_manifest(report: dict, path: Path) -> None:
    """Openable by Material Review with durable labels and model identity."""
    groups = {}
    reference_identities = {}
    for record in report["comparison"]:
        provenance = dict(record.get("reference_provenance", {}))
        identity = (record["reference_exr"], provenance)
        if record["sample"] in reference_identities and reference_identities[record["sample"]] != identity:
            raise ValueError("Comparison variants have different original reference provenance")
        reference_identities[record["sample"]] = identity
        bits = provenance.get("source_bits")
        precision = f"Original UInt{bits} height codes" if type(bits) is int and bits in (8, 16) else "Original numeric height values"
        group = groups.setdefault(record["sample"], {"material_id": record["sample"],
            "diffuse": record["source_png"], "diffuse_encoding": "sRGB",
            "height": record["reference_exr"], "reference_height": record["reference_exr"],
            **provenance,
            "variants": [{"name": "target", "role": "target", "height": record["reference_exr"],
                          "sample_label": record["sample"], "map_type": "height", **provenance,
                          "detail": f"{precision} in Float32 EXR · no gamma or stretching · real source map, not a model output"}]})
        if type(bits) is int and bits in (8, 16):
            group["target_original_bits"] = bits
        label = record["model_label"]
        if label == "pretrained-base":
            checkpoint, checksum, step = None, WEIGHTS_SHA256, None
            title = "PBRnxt pretrained base · native scale"
        elif label == "starting-checkpoint":
            checkpoint, checksum, step = report["parent"]["path"], report["parent"]["sha256"], report["parent"]["step"]
            title = f"PBRnxt starting checkpoint · step{step}"
        else:
            checkpoint, checksum, step = report["checkpoint"], report["checkpoint_sha256"], report["saved_step"]
            title = f"PBRnxt refined · step{step}"
        variant = {"name": title, "role": "checkpoint" if checkpoint else "base",
                   "height": record["height_exr"], "sample_label": record["sample"],
                   "model_name": "PBRnxt complete pretrained material mapping",
                   "model_architecture": "SCUNet + all trained RRDB convolutions · experimental1:1 adaptation",
                   "checkpoint_sha256": checksum, "map_type": "height",
                   "detail": (f"Native{' × '.join(str(value) for value in record.get('native_dimensions', [record['native_size'], record['native_size']]))} pixels"
                              f" · real source crop{record['source_rectangle']} · no padding/resizing · experimental")}
        if checkpoint:
            variant.update(checkpoint=checkpoint, checkpoint_step=step)
        group["variants"].append(variant)
    write_json(path, {"schema": "texture-studio-material-quality-review-v1", "comparison_target": "height",
                      "materials": list(groups.values()), "automatic_model_promotion": False})


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("train", "refine", "evaluate", "evaluate-base"))
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/pbrnxt-base")
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--weights", type=Path)
    cli.add_argument("--download", action="store_true")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--scope", choices=tuple(REFINEMENT_SCOPES),
                     help="Train existing pretrained final height branch or also its generator height decoder. New training defaults to final-height; refine/evaluate inherit the checkpoint scope. No random heads.")
    cli.add_argument("--materials", nargs="+")
    cli.add_argument("--size", type=int, default=1024, help="Actual real-pixel training crop, separate from dataset crop size")
    cli.add_argument("--whole-maps", action="store_true", help="Use every real pixel of each paired map, including rectangular maps. --size is a maximum edge; no padding/resizing or square crops.")
    cli.add_argument("--updates", type=int, default=194)
    cli.add_argument("--max-minutes", type=float, default=60)
    cli.add_argument("--learning-rate", type=float, default=1e-5)
    cli.add_argument("--weight-decay", type=float, default=1e-4)
    cli.add_argument("--memory-gib", type=float, default=max(8, os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3 - 8))
    cli.add_argument("--cache-gib", type=float, default=4)
    cli.add_argument("--check-count", type=int, default=5)
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--device", choices=("mps", "cpu"), default="mps" if torch.backends.mps.is_available() else "cpu")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "evaluate-base" and (args.checkpoint or args.scope):
        raise ValueError("Evaluate-base inspects only the pretrained base; omit checkpoint and refinement scope")
    if (args.command == "refine" and not args.checkpoint) or (args.command == "train" and args.checkpoint):
        raise ValueError("Train starts from the pretrained base; refine requires a PBRnxt checkpoint")
    if args.command == "evaluate" and not args.checkpoint:
        raise ValueError("Evaluate requires a refinement checkpoint; train automatically exports its pretrained base")
    if args.command in ("train", "refine"):
        for name in ("learning_rate", "weight_decay", "max_minutes"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if args.learning_rate <= 0 or args.max_minutes <= 0 or args.updates < 1:
            raise ValueError("Update count, learning rate and time limit must be positive")
    if args.check_count < 1:
        raise ValueError("Inspection sample count must be positive")
    previous = read_refinement_checkpoint(args.checkpoint) if args.checkpoint else None
    args.scope = resolve_scope(args.command, args.scope, previous)
    is_training = args.command in ("train", "refine")
    plan = resource_plan(args.size, args.memory_gib, args.cache_gib, args.scope, inference=not is_training)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new run folder; existing experiments are not overwritten")
    training, checks, identity = select_pairs(args.dataset, args.materials)
    if args.whole_maps:
        if any(min(pair["dimensions"]) < 256 or any(value % 64 for value in pair["dimensions"])
               or max(pair["dimensions"]) > args.size for pair in training + checks):
            raise ValueError("Whole maps need real dimensions at least256, divisible by64 and within --size maximum edge; choose an appropriate published source, no padding/resizing")
    elif any(min(pair["dimensions"]) < args.size for pair in training + checks):
        raise ValueError("Requested crop exceeds real dataset pixels; choose a smaller size")
    # Exact same known-source regions compare base / starting checkpoint / result.
    # When checks were not prepared for a single-material fit, label training
    # inspections honestly; do not invent a validation split or reserve pixels.
    inspection = checks[:args.check_count] or training[:args.check_count]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.source_dir or args.weights:
        if not args.source_dir or not args.weights:
            raise ValueError("Supply both pinned source directory and weights")
        source, weights = args.source_dir, args.weights
    else:
        source, weights = obtain_pretrained(args.model_directory, args.download)
    if device.type == "mps":
        # Budget includes host cache; a hard driver cap prevents swapping the
        # whole Mac to finish an accidentally oversized native grid.
        driver_budget = max(1, args.memory_gib - args.cache_gib - 2) * 1024 ** 3
        torch.mps.set_per_process_memory_fraction(driver_budget / torch.mps.recommended_max_memory())
    model = load_complete_pretrained(source, weights, device=device)
    if not is_training:
        model.eval().requires_grad_(False)
        parameters = []
    else:
        parameters = configure_refinement(model, args.scope)
    cache = PairCache(int(args.cache_gib * 1024 ** 3))
    report = {"schema": SCHEMA, "status": "running", "operation": args.command,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "experimental": True, "production_eligible": False, "base": model.provenance,
              "dataset": identity, "inspection_crop_size": args.size, "inspection_size": args.size,
              "training_crop_size": (None if args.command == "evaluate-base" else
                                     previous["training_crop_size"] if args.command == "evaluate" else args.size),
              "training_performed": is_training, "whole_maps": args.whole_maps,
              "dataset_crop_sizes": sorted({max(pair["dimensions"]) for pair in training}),
              "dataset_sample_dimensions": [list(value) for value in sorted({tuple(pair["dimensions"]) for pair in training})],
              "image_padding": False, "image_resizing": False, "resources": plan,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "refinement_scope": None if args.command == "evaluate-base" else args.scope,
              "trainable_prefixes": list(scope_prefixes(args.scope)) if is_training else [],
              "training_policy": ("Inspect pretrained base only; no training or optimizer" if args.command == "evaluate-base" else
                                  "Inspect the recorded checkpoint; no training or optimizer" if args.command == "evaluate" else
                                  "Refine pretrained final RRDB height output branch; complete material generator and other output branches frozen"
                                  if args.scope == "final-height" else
                                  "Refine pretrained generator height decoder/tail and final RRDB height branch; shared encoder/body/fusion and other material branches frozen"),
              "boundary_policy": "Loss excludes64 outer pixels, but inputs/targets contain only real source pixels",
              "device": str(device), "steps": [], "comparison": []}
    print(json.dumps({"event": "configuration", "base": "PBRnxt86.76M complete pretrained material network, native-scale adaptation",
                      "training_crop": report["training_crop_size"], "inspection_crop": args.size,
                      "dataset_crop": report["dataset_crop_sizes"], "refinement_scope": args.scope,
                      "trainable_parameters": report["trainable_parameters"], "resources": plan}), flush=True)
    comparison_options = {"whole_maps": True} if args.whole_maps else {}
    report["comparison"] += export_comparison(model, inspection, cache, args.output / "comparison", "pretrained-base", device, args.size, args.seed, **comparison_options)
    if args.command == "evaluate-base":
        report.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "run.json", report)
        write_review_manifest(report, args.output / "review-manifest.json")
        print(json.dumps({"status": "completed", "operation": "evaluate-base", "inspection_size": args.size,
                          "whole_maps": args.whole_maps, "training_performed": False,
                          "trainable_parameters": 0, "production_eligible": False}), flush=True)
        return 0
    parent_step = 0
    if args.checkpoint:
        previous = restore_refinement(model, args.checkpoint, saved=previous)
        parent_step = previous["step"]
        report["parent"] = {"path": str(args.checkpoint.resolve()), "sha256": previous["_snapshot_sha256"], "step": parent_step,
                            "refinement_scope": checkpoint_scope(previous)}
        if args.command != "evaluate":
            report["comparison"] += export_comparison(model, inspection, cache, args.output / "comparison", "starting-checkpoint", device, args.size, args.seed, **comparison_options)
    write_json(args.output / "run.json", report)
    optimizer = scheduler = None
    if is_training:
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=args.learning_rate * .2)
    latest = args.output / "checkpoint.latest.pt"
    stop_requested = False

    def stop(_number, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stopping after this update and saving the checkpoint…", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    step = 0
    completed_updates = 0
    started_run = time.monotonic()
    rng = random.Random(args.seed)
    schedule = []
    while is_training and len(schedule) < args.updates:
        epoch = list(range(len(training)))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    try:
        if args.command != "evaluate":
            for step, index in enumerate(schedule[:args.updates], 1):
                rgb, target, rectangle = crop_pair(*cache.load(training[index]), args.size, rng, augment=True, whole_maps=args.whole_maps)
                rgb, target = rgb.to(device), target.to(device)
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                prediction = predict(model, rgb)
                loss, metrics = height_loss(prediction, target, margin=64)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
                if not torch.isfinite(gradient_norm):
                    raise ValueError("Nonfinite training gradient; stopped before changing weights")
                optimizer.step()
                scheduler.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                event = {"event": "update", "step": parent_step + step,
                         "sample": training[index]["metadata"]["sample_id"], "source_rectangle": rectangle,
                         "model_input_dimensions": [rgb.shape[-1], rgb.shape[-2]], "whole_maps": args.whole_maps,
                         "seconds": time.monotonic() - started, "metrics": metrics,
                         "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(gradient_norm),
                         "driver_gib": torch.mps.driver_allocated_memory() / 1024 ** 3 if device.type == "mps" else None}
                report["steps"].append(event)
                completed_updates = step
                print(json.dumps(event), flush=True)
                del rgb, target, prediction, loss
                if step % 20 == 0:
                    save_checkpoint(latest, model, optimizer, parent_step + step, report, args)
                    write_json(args.output / "run.json", report)
                if stop_requested or time.monotonic() - started_run >= args.max_minutes * 60:
                    break
            save_checkpoint(latest, model, optimizer, parent_step + completed_updates, report, args)
        report["comparison"] += export_comparison(model, inspection, cache, args.output / "comparison", "refined-checkpoint", device, args.size, args.seed, **comparison_options)
        report["status"] = "stopped" if stop_requested else "completed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["saved_step"] = parent_step + completed_updates
        if args.command == "evaluate":
            report["checkpoint"] = str(args.checkpoint.resolve())
            report["checkpoint_sha256"] = previous["_snapshot_sha256"]
        else:
            report["checkpoint"] = str(latest.resolve())
            report["checkpoint_sha256"] = digest(latest)
        write_json(args.output / "run.json", report)
        write_review_manifest(report, args.output / "review-manifest.json")
        print(json.dumps({"status": report["status"], "checkpoint": report["checkpoint"], "step": report["saved_step"],
                          "training_crop_size": report["training_crop_size"], "inspection_crop_size": args.size,
                          "production_eligible": False}), flush=True)
    except BaseException as error:
        report["status"], report["error"] = "failed", str(error)
        report["completed_updates"] = completed_updates
        if completed_updates:
            save_checkpoint(latest, model, optimizer, parent_step + completed_updates, report, args)
        write_json(args.output / "run.json", report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
