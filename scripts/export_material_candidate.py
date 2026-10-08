#!/usr/bin/env python3
"""Export checked frozen/LoRA height candidates for visual material review.

Metrics do not select a winner here. Original diffuse/roughness and native
height values stay separate; Blender displays every candidate at shared gain.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

from frozen_dino_height import CODE_REVISION, MODEL_REVISION, MODEL_SHA256, ConditionedHeightNet, extract_features, load_frozen_encoder
from probe_material_adapters import install_adapters
from train_material_height import checked_relative, choose_device, diffuse_encoding, digest, find_samples, load_pair, write_float_exr, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "texture-studio-four-material-adaptation-diagnostic-v1"
CYCLE_SCHEMA = "texture-studio-material-training-cycle-v1"


def candidate_head(payload: dict, device: torch.device) -> torch.nn.Module:
    """Construct only from the verified byte snapshot; never reload its path."""
    if payload["schema"] == CYCLE_SCHEMA:
        from material_training_cycle import MaterialMapHead
        head = MaterialMapHead(**payload["head_config"])
    else:
        head = ConditionedHeightNet(**payload["head_config"])
    head.load_state_dict(payload["head_state"], strict=True)
    return head.to(device).eval()


def candidate_snapshot(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Candidate checkpoint must contain a dictionary")
    if payload.get("encoder", {}).get("checkpoint_sha256") != MODEL_SHA256 or payload["encoder"].get("official_meta_code_revision") != CODE_REVISION:
        raise ValueError("Candidate encoder identity differs from pinned weights/source")
    if payload.get("schema") == CYCLE_SCHEMA:
        if payload.get("target") != "height":
            raise ValueError("Height visual comparison requires a height-trained cycle checkpoint")
        config = payload.get("head_config", {})
        if set(config) != {"base_channels", "feature_channels", "projection_channels", "target"} or config.get("feature_channels") != 768 or config.get("target") != "height":
            raise ValueError("Unsupported height-cycle head configuration")
        if payload.get("adapter_state"):
            raise ValueError("Frozen height-cycle candidate unexpectedly contains adapters")
    else:
        if payload.get("schema") != SCHEMA or payload.get("variant") not in ("frozen", "lora"):
            raise ValueError("Expected a recorded frozen/LoRA adaptation checkpoint")
        if payload.get("head_config") != {"base_channels": 12, "feature_channels": 768, "projection_channels": 12}:
            raise ValueError("Unsupported native head configuration")
    state = payload.get("head_state")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Candidate head must contain finite Float32 tensors")
    if payload.get("variant") == "frozen" and payload.get("adapter_state"):
        raise ValueError("Frozen candidate unexpectedly contains adapters")
    size = payload.get("encoder_size", 518)
    if not isinstance(size, int) or isinstance(size, bool) or not 28 <= size <= 518 or size % 14:
        raise ValueError("Candidate encoder size must be aligned to 14 within 28–518")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Candidate checkpoint needs its actual nonnegative step")
    # Validate all head keys/shapes before any large encoder allocation.
    del_head = candidate_head(payload, torch.device("cpu"))
    del del_head
    return payload, checksum


def checked_candidate(path: Path) -> dict:
    return candidate_snapshot(path)[0]


def verify_bindings(bindings: dict[str, str]) -> None:
    for name, checksum in bindings.items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Selected export file changed: {name}")


def sample_snapshot(sample: dict) -> tuple[bytes, dict[str, str], dict[str, Path]]:
    path = sample["metadata_path"].resolve()
    raw = path.read_bytes()
    if json.loads(raw) != sample["metadata"]:
        raise ValueError("Selected sample manifest changed during selection")
    bindings = {str(path): hashlib.sha256(raw).hexdigest()}
    paths = {}
    for role in ("input", "height", "roughness"):
        metadata = sample["metadata"]
        paths[role] = checked_relative(path.parent, metadata["maps"][role])
        checksum = metadata["map_metadata"][role]["sample_sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"Need an exact prepared {role} map checksum")
        bindings[str(paths[role])] = checksum
    if paths["input"] != sample["input_path"].resolve() or paths["height"] != sample["height_path"].resolve():
        raise ValueError("Selected sample paths differ from its frozen manifest")
    return raw, bindings, paths


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    candidates = []
    names = set()
    for entry in args.candidate:
        name, separator, raw_path = entry.partition("=")
        if not separator or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ValueError("Candidates require unique simple NAME=CHECKPOINT entries")
        names.add(name)
        path = Path(raw_path).resolve()
        payload, checksum = candidate_snapshot(path)
        candidates.append((name, path, payload, checksum))
    if names & {"target", "flat"}:
        raise ValueError("Target and flat candidates are generated automatically")
    samples = [sample for sample in find_samples(args.dataset, args.allow_unreviewed) if sample["metadata"]["material_id"] in set(args.material) and sample["metadata"]["split"] == args.split]
    if {sample["metadata"]["material_id"] for sample in samples} != set(args.material):
        raise ValueError("Every requested material needs samples in the chosen split")
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("Export output must be outside its dataset")
    bindings = {}
    for _, path, _, checksum in candidates:
        if str(path) in bindings and bindings[str(path)] != checksum:
            raise ValueError("Repeated checkpoint path changed between candidate snapshots")
        bindings[str(path)] = checksum
    snapshots, source_paths = {}, {}
    for sample in samples:
        identity = sample["metadata"]["sample_id"]
        if not identity or len(identity) > 96 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identity):
            raise ValueError("Sample IDs need simple file-safe names")
        if identity in snapshots:
            raise ValueError("Duplicate selected sample ID")
        raw, source_bindings, paths = sample_snapshot(sample)
        snapshots[identity] = raw
        source_paths[identity] = paths
        bindings.update(source_bindings)
    verify_bindings(bindings)
    index_hash = digest(args.dataset / "dataset.json")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_directory = args.output / "dataset-snapshot"
    snapshot_directory.mkdir()
    for identity, raw in snapshots.items():
        (snapshot_directory / (identity + ".json")).write_bytes(raw)
    device = choose_device(args.device)
    materials = []
    for sample in samples:
        metadata = sample["metadata"]
        paths = source_paths[metadata["sample_id"]]
        width, height = metadata["sample_pixel_dimensions"]
        flat_path = args.output / f"{metadata['sample_id']}.flat.height.float32.exr"
        write_float_exr(flat_path, np.full((height, width), 0.5, dtype=np.float32))
        materials.append({"material_id": metadata["sample_id"], "diffuse": str(paths["input"]), "diffuse_encoding": "linear" if diffuse_encoding(metadata) == "linear" else "sRGB", "roughness": str(paths["roughness"]), "physical_width_m": 1.0, "displacement_scale_m": 0.03, "height_midlevel": 0.5, "height_gain": 1.0, "height_offset": 0.0, "tiling_repeats": 1, "surface_mode": "geometric_displacement", "variants": [{"name": "flat", "height": str(flat_path.resolve()), "height_sha256": digest(flat_path)}, {"name": "target", "height": str(paths["height"]), "height_sha256": bindings[str(paths["height"])]}], "source_region": metadata["crop_rectangle_top_left_xywh"], "target_original_bits": metadata["map_metadata"]["height"]["sample_bits"], "sample_metadata_snapshot": "dataset-snapshot/" + metadata["sample_id"] + ".json", "sample_metadata_sha256": bindings[str(sample["metadata_path"].resolve())], "source_map_sha256": {role: bindings[str(path)] for role, path in paths.items()}})
    for name, checkpoint_path, payload, checksum in candidates:
        verify_bindings(bindings)
        encoder, _ = load_frozen_encoder(args.model_directory, args.code_directory, device)
        if payload.get("variant") == "lora":
            adapters = install_adapters(encoder)
            if set(adapters) != set(payload.get("adapter_state", {})):
                raise ValueError("Adapter layer identities differ")
            for key, module in adapters.items():
                state = payload["adapter_state"][key]
                if set(state) != {"lora_A", "lora_B"}:
                    raise ValueError("Unexpected adapter tensor names")
                for field, value in state.items():
                    parameter = getattr(module, field)
                    if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                        raise ValueError("Invalid adapter tensor")
                    parameter.copy_(value.to(device))
            encoder.requires_grad_(False).eval()
        head = candidate_head(payload, device)
        for sample, material in zip(samples, materials):
            source, _ = load_pair(sample, device)
            encoder_size = payload.get("encoder_size", 518)
            prediction = head(source, extract_features(encoder, source, encoder_size))
            values = prediction[0, 0].detach().float().cpu().numpy()
            output = args.output / f"{material['material_id']}.{name}.height.float32.exr"
            write_float_exr(output, values)
            material["variants"].append({"name": name, "height": str(output.resolve()), "height_sha256": digest(output), "checkpoint": str(checkpoint_path), "checkpoint_sha256": checksum, "checkpoint_step": payload["step"], "native_dimensions": list(values.shape[::-1]), "contextual_encoder_size": encoder_size})
            del source, prediction
        del head, encoder
        if device.type == "mps":
            torch.mps.empty_cache()
        verify_bindings(bindings)
    verify_bindings(bindings)
    report = {"schema": "texture-studio-material-quality-review-v1", "purpose": "Human visual review; no metric winner or automatic model promotion", "source_images_modified": False, "native_numeric_exports_resized": False, "selected_files_sha256": bindings, "source_files_verified_unchanged": True, "checkpoint_files_verified_unchanged": True, "dataset_index_sha256_at_selection": index_hash, "dataset_index_changed_during_export": digest(args.dataset / "dataset.json") != index_hash, "materials": materials}
    write_json(args.output / "review-manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=CHECKPOINT; repeat")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
    parser.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
    args = parser.parse_args()
    try:
        result = run(args)
        print(f"Exported {len(result['materials'])} native material groups: {args.output / 'review-manifest.json'}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"export_material_candidate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
