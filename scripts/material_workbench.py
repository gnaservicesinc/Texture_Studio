#!/usr/bin/env python3
"""Deterministic JSON bridge for native material curation and learned maps."""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import urllib.request

import numpy as np
import torch

from frozen_dino_height import ARCHITECTURE, CODE_HASHES, CODE_REVISION, CONFIG_SHA256, DIAGNOSTIC_SCHEMA, MODEL_BYTES, MODEL_REVISION, MODEL_SHA256, extract_features, load_frozen_encoder
from material_training_cycle import ADAPTATION_SCHEMA, SCHEMA as CYCLE_SCHEMA, MaterialMapHead
from material_fixed_adapters import POLICY as FIXED_ADAPTER_POLICY, validate_adapters
from train_material_height import checked_relative, choose_device, digest, linear_rgb, opengl_normal_from_height, write_float_exr

SCHEMA = "texture-studio-material-workbench-v1"
PACKAGE_SCHEMA = "texture-studio-material-inference-package-v1"
ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ".material-workbench-journal.json"
OWNED_ENCODER = ".texture-studio-owned-encoder.json"


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


def canonical_dataset(path: Path) -> Path:
    if path.name == "dataset.json" and path.is_file():
        path = path.parent
    return path.resolve()


def read_dataset(dataset: Path) -> tuple[dict, list[tuple[dict, Path, dict]]]:
    dataset = dataset.resolve()
    index = json.loads((dataset / "dataset.json").read_text())
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
        if not any(name in command for name in ("material_training_cycle.py", "train_material_height.py", "diagnose_material_")):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            return True
        for i, token in enumerate(tokens):
            selected = tokens[i + 1] if token == "--dataset" and i + 1 < len(tokens) else token.partition("=")[2] if token.startswith("--dataset=") else None
            if selected and Path(selected).resolve() == dataset.resolve():
                return True
        # Resume may inherit dataset from its checkpoint; block conservatively.
        if "resume" in tokens:
            return True
    return False


def dataset_info(args) -> dict:
    args.dataset = canonical_dataset(args.dataset)
    with dataset_lock(args.dataset):
        index, records = read_dataset(args.dataset)
        materials = {}
        for _item, path, sample in records:
            maps = {}
            for role, filename in sample.get("maps", {}).items():
                details = sample.get("map_metadata", {}).get(role, {})
                maps[role] = {"path": str(checked_relative(path.parent, filename)), "sha256": details.get("sample_sha256"), "source_bits": details.get("sample_bits", details.get("source", {}).get("sample_bits")), "encoding": details.get("encoding")}
            width, height = sample.get("sample_pixel_dimensions", [None, None])
            material = materials.setdefault(sample["material_id"], {"material_id": sample["material_id"], "samples": []})
            material["samples"].append({"sample_id": sample["sample_id"], "status": sample["status"], "split": sample["split"], "width": width, "height": height, "maps": maps, "crop_rectangle": sample.get("crop_rectangle_top_left_xywh"), "note": sample.get("curation_note"), "metadata_path": str(path)})
        return {"dataset_path": str(args.dataset.resolve()), "index_sha256": digest(args.dataset / "dataset.json"), "split_strategy": index.get("split_strategy"), "validation_scope": index.get("validation_scope"), "cross_size_validation_notice": index.get("native_size_preparation", {}).get("cross_size_validation_notice"), "automatic_validation": index.get("automatic_validation"), "materials": [materials[key] for key in sorted(materials)]}


def prepare_size(args) -> dict:
    from material_native_size import PREPARATION_SCHEMA, prepare_from_records
    supplied = canonical_dataset(args.dataset)
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
        cache_lock = lambda path: contextlib.nullcontext() if path in (original, supplied) else dataset_lock(path)
        options = dict(automatic_validation=getattr(args, "automatic_validation", False),
                       validation_material=getattr(args, "material", None), workers=getattr(args, "workers", None))
        if original == supplied:
            prepared, information = prepare_from_records(original, supplied_index, supplied_records, args.size, cache_lock, **options)
        else:
            original_index, original_records = read_dataset(original)
            prepared, information = prepare_from_records(original, original_index, original_records, args.size, cache_lock, review_records=supplied_records if options["automatic_validation"] else None, **options)
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
        matches = [(item, path, sample) for item, path, sample in records if sample["sample_id"] == args.sample]
        if len(matches) != 1:
            raise ValueError("Select one exact indexed sample ID")
        item, path, sample = matches[0]
        sample["status"] = item["status"] = args.status
        if args.split:
            sample["split"] = item["split"] = args.split
            if index.get("split_strategy") == "heldout-region-v1":
                sample["source_region_role"] = args.split
                from train_material_height import verify_region_split
                verify_region_split([{"metadata": record[2]} for record in records], index)
            else:
                material_splits = {}
                for _entry, _path, record in records:
                    material_splits.setdefault(record["material_id"], set()).add(record["split"])
                if any(len(splits) > 1 for splits in material_splits.values()):
                    raise ValueError("Whole-material validation forbids material identities crossing splits")
        if args.note is not None:
            sample["curation_note"] = args.note
        sample["review_status"] = {"approved": "user_approved", "excluded": "user_excluded", "unreviewed": "unreviewed"}[args.status]
        changes = []
        for changed_path, value in ((path, sample), (index_path, index)):
            raw = json_bytes(value)
            changes.append({"path": str(changed_path.resolve().relative_to(args.dataset.resolve())), "old_sha256": digest(changed_path), "new_sha256": hashlib.sha256(raw).hexdigest(), "new_base64": base64.b64encode(raw).decode()})
        atomic_bytes(args.dataset / JOURNAL, json_bytes({"schema": SCHEMA, "changes": changes}))
        recover_journal(args.dataset)
        return {"dataset_path": str(args.dataset.resolve()), "sample_id": args.sample, "status": args.status, "split": sample["split"], "index_sha256": digest(index_path), "previous_index_sha256": previous, "metadata_sha256": digest(path), "source_bytes_modified": False}


def checkpoint_snapshot(path: Path, expected: str | None = None) -> tuple[dict, str]:
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Material head checkpoint exceeds the 64 MiB native-head bound")
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected and expected != checksum:
        raise ValueError("Selected checkpoint SHA256 changed")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    if payload.get("schema") == PACKAGE_SCHEMA:
        payload = payload["inference_checkpoint"]
    schema = payload.get("schema")
    if schema == DIAGNOSTIC_SCHEMA:
        if payload.get("architecture") != ARCHITECTURE or payload.get("variant") != "frozen_features":
            raise ValueError("Unsupported frozen-feature checkpoint architecture/variant")
        payload = dict(payload, head_config=dict(payload["model_config"], target="height"), head_state=payload["model_state"], target="height")
    elif schema == ADAPTATION_SCHEMA:
        if payload.get("variant") not in ("frozen", "lora"):
            raise ValueError("Unsupported adaptation variant")
        payload = dict(payload, head_config=dict(payload["head_config"], target="height"), target="height")
    elif schema != CYCLE_SCHEMA:
        raise ValueError("Unsupported learned material checkpoint; no model fallback is permitted")
    if payload.get("variant", "frozen") not in ("frozen", "lora", "frozen_features"):
        raise ValueError("Unsupported declared encoder variant")
    config = payload.get("head_config", {})
    if set(config) != {"base_channels", "feature_channels", "projection_channels", "target"} or config.get("feature_channels") != 768 or config.get("target") not in ("height", "roughness", "normal") or payload.get("target") != config["target"]:
        raise ValueError("Unsupported material head configuration")
    if (type(config["base_channels"]) is not int or not 4 <= config["base_channels"] <= 64 or config["base_channels"] % 4
            or type(config["projection_channels"]) is not int or not 1 <= config["projection_channels"] <= 64
            or type(config["feature_channels"]) is not int):
        raise ValueError("Native head channel dimensions exceed bounded configuration limits")
    encoder = payload.get("encoder", {})
    if encoder.get("checkpoint_sha256") != MODEL_SHA256 or encoder.get("official_meta_code_revision") != CODE_REVISION:
        raise ValueError("Checkpoint encoder differs from the pinned DINOv2 Base dependency")
    state = payload.get("head_state")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Material head tensors must be finite Float32")
    model = MaterialMapHead(**config)
    model.load_state_dict(state, strict=True)
    size, step = payload.get("encoder_size"), payload.get("step")
    if size is None and schema == ADAPTATION_SCHEMA:
        run_path = path.parent.parent / "run.json"
        if not run_path.is_file():
            raise ValueError("Legacy adaptation checkpoint needs its matching run.json to resolve encoder size; no default is assumed")
        run_raw = run_path.read_bytes()
        run = json.loads(run_raw)
        dimensions = run.get("encoder_dimensions")
        if (run.get("schema") != ADAPTATION_SCHEMA or not isinstance(dimensions, list) or len(dimensions) != 2 or dimensions[0] != dimensions[1]
                or not payload.get("schedule_sha256") or run.get("schedule_sha256") != payload.get("schedule_sha256")
                or not payload.get("selected_files_sha256") or run.get("selected_files_sha256") != payload.get("selected_files_sha256")):
            raise ValueError("Legacy adaptation run.json does not match checkpoint schedule/source identity")
        size = dimensions[0]
        payload = dict(payload, encoder_size=size, encoder_size_provenance={"source": "matched_legacy_run_json", "path": str(run_path.resolve()), "sha256": hashlib.sha256(run_raw).hexdigest(), "schedule_sha256": payload["schedule_sha256"]})
    elif size is None:
        raise ValueError("Checkpoint must declare its encoder working size")
    if not isinstance(size, int) or not 28 <= size <= 518 or size % 14 or not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Checkpoint needs an actual step and bounded aligned encoder size")
    adapters = payload.get("adapter_state", {})
    if payload.get("variant") == "lora":
        validate_adapters(adapters)
    elif adapters:
        raise ValueError("Frozen checkpoint contains unexplained adapters")
    return payload, checksum


def checkpoint_info(args) -> dict:
    payload, checksum = checkpoint_snapshot(args.checkpoint, args.expected_sha256)
    return {"checkpoint_path": str(args.checkpoint.resolve()), "sha256": checksum, "schema": payload["schema"], "target": payload["target"], "step": payload["step"], "head_config": payload["head_config"], "encoder": payload["encoder"], "encoder_size": payload["encoder_size"], "variant": payload.get("variant", "frozen"), "compatible": True, "supports_training_warm_start": True, "refinement_policy": FIXED_ADAPTER_POLICY if payload.get("variant") == "lora" else "refine_material_head_with_frozen_encoder"}


def load_photo(path: Path, encoding: str) -> tuple[torch.Tensor, dict]:
    if path.suffix.lower() == ".png":
        from material_dataset import read_png
        codes, details = read_png(path)
        if encoding == "auto":
            encoding = "linear" if details.get("png_gamma") == 1 else "srgb"
    else:
        from PIL import Image, ImageOps
        if path.suffix.lower() in (".heic", ".heif"):
            from pillow_heif import register_heif_opener
            register_heif_opener()
        with Image.open(path) as source:
            photo = ImageOps.exif_transpose(source)
            if photo.mode not in ("RGB", "RGBA", "L"):
                raise ValueError("Photo must be RGB/gray; unsupported color profiles require explicit conversion")
            codes = np.asarray(photo).copy()
        details = {"sample_bits": 8}
        if encoding == "auto":
            encoding = "srgb"
    rgb = linear_rgb(codes, encoding)
    if max(rgb.shape[:2]) > 2048 or min(rgb.shape[:2]) < 16:
        raise ValueError("Use a native crop between 16 and 2048 pixels per side; inference never silently resizes the output")
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0), {"input_encoding": encoding, "source_bits": details.get("sample_bits"), "native_dimensions": [rgb.shape[1], rgb.shape[0]]}


@torch.no_grad()
def infer(args, encoder_loader=load_frozen_encoder) -> dict:
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("MPS inference requires CPU fallback disabled")
    payload, checksum = checkpoint_snapshot(args.checkpoint, args.expected_sha256)
    image_checksum = digest(args.image)
    source, photo = load_photo(args.image, args.input_encoding)
    device = choose_device(args.device)
    encoder, pins = encoder_loader(args.model_directory, args.code_directory, device)
    if payload.get("variant") == "lora":
        from probe_material_adapters import install_adapters
        adapters = install_adapters(encoder)
        for name, module in adapters.items():
            for field, value in payload["adapter_state"][name].items():
                getattr(module, field).copy_(value.to(device))
        encoder.requires_grad_(False).eval()
    head = MaterialMapHead(**payload["head_config"]).to(device).eval()
    head.load_state_dict(payload["head_state"], strict=True)
    source = source.to(device)
    prediction = head(source, extract_features(encoder, source, payload.get("encoder_size", 518)))
    value = prediction[0].cpu().numpy().transpose(1, 2, 0)
    if digest(args.checkpoint) != checksum or digest(args.image) != image_checksum:
        raise ValueError("Selected input/checkpoint changed during inference")
    args.output.mkdir(parents=True, exist_ok=False)
    target = payload["target"]
    outputs = {}
    def save(role, array):
        path = args.output / (role + ".float32.exr")
        write_float_exr(path, array)
        outputs[role] = {"path": str(path.resolve()), "sha256": digest(path), "encoding": "linear_data", "storage": "FLOAT32", "blender_color_space": "Non-Color"}
    if target in ("height", "roughness"):
        save(target, value[..., 0])
        if target == "height":
            save("normal", opengl_normal_from_height(value[..., 0]))
    else:
        save("normal", value)
    report = dict(photo, outputs=outputs, checkpoint_sha256=checksum, checkpoint_step=payload["step"], target=target, encoder_pins=pins, image_sha256=image_checksum, normal_convention="OpenGL +Y", normal_origin="height derivative at artistic height/patch-width ratio0.03" if target == "height" else "independent normal target" if target == "normal" else None, source_bytes_modified=False)
    atomic_bytes(args.output / "inference.json", json_bytes(report))
    return report


def package(args) -> dict:
    payload, checksum = checkpoint_snapshot(args.checkpoint, args.expected_sha256)
    compact = {key: payload[key] for key in ("schema", "head_config", "head_state", "target", "encoder", "step")}
    # Historical diagnostic state has already been strictly validated and
    # normalized above. Canonical inference avoids retaining duplicate tensors
    # or a historical schema whose required training fields were removed.
    compact["source_checkpoint_schema"] = payload["schema"]
    if payload.get("variant") != "lora":
        compact["schema"] = CYCLE_SCHEMA
    compact.update(encoder_size=payload["encoder_size"], variant="lora" if payload.get("variant") == "lora" else "frozen", adapter_state=payload.get("adapter_state", {}))
    compact["encoder_refinement_policy"] = payload.get("encoder_refinement_policy", FIXED_ADAPTER_POLICY if compact["variant"] == "lora" else "refine_material_head_with_frozen_encoder")
    if payload.get("encoder_size_provenance"):
        compact["encoder_size_provenance"] = payload["encoder_size_provenance"]
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "material-head.pt"
    torch.save({"schema": PACKAGE_SCHEMA, "inference_checkpoint": compact, "original_checkpoint_sha256": checksum}, path)
    config = {"schema": PACKAGE_SCHEMA, "target": payload["target"], "head_config": payload["head_config"], "source_checkpoint_sha256": checksum, "step": payload["step"], "variant": compact["variant"], "encoder_dependency": {"repo": "facebook/dinov2-base", "revision": MODEL_REVISION, "checkpoint_sha256": MODEL_SHA256, "bytes": MODEL_BYTES, "official_code_revision": CODE_REVISION, "license": "Apache-2.0", "included": False}, "source_images_included": False, "optimizer_included": False, "license": "GPL-3.0-or-later", "full_da3_model": False, "has_rank8_attention_adapters": compact["variant"] == "lora"}
    atomic_bytes(args.output / "config.json", json_bytes(config))
    (args.output / "README.md").write_text("---\nlicense: gpl-3.0\ntags:\n- material-maps\n- dinov2\n---\n\n# Texture Studio material head\n\n" + ("Rank-8 DINOv2 attention adapters and native material head.\n" if compact["variant"] == "lora" else "Native material head conditioned by frozen DINOv2 Base features.\n") + "\nThis package contains learned head tensors and optional declared adapters; it is not DA3, a full DINOv2 encoder, or a production certification. The separately pinned encoder and official code are Apache-2.0. Targets use linear numeric data; Float32 export does not add information beyond the training source precision. Review visible relief and independent photos before choosing a production model.\n")
    license_path = ROOT / "LICENSE"
    if not license_path.exists():
        license_path = Path(__file__).with_name("LICENSE")
    if not license_path.exists():
        raise ValueError("Project GPL license text is required for packaging")
    shutil.copyfile(license_path, args.output / "LICENSE")
    atomic_bytes(args.output / "SHA256SUMS.json", json_bytes({p.name: digest(p) for p in args.output.iterdir() if p.is_file()}))
    return {"package_path": str(args.output.resolve()), "checkpoint_path": str(path.resolve()), "checkpoint_sha256": digest(path), "source_checkpoint_sha256": checksum, "target": payload["target"], "variant": compact["variant"], "encoder_included": False}


def encoder_files() -> dict[str, dict]:
    result = {
        "model/model.safetensors": {"sha256": MODEL_SHA256, "bytes": MODEL_BYTES, "url": f"https://huggingface.co/facebook/dinov2-base/resolve/{MODEL_REVISION}/model.safetensors"},
        "model/config.json": {"sha256": CONFIG_SHA256, "maximum_bytes": 65536, "url": f"https://huggingface.co/facebook/dinov2-base/resolve/{MODEL_REVISION}/config.json"}}
    result.update({"code/" + name: {"sha256": checksum, "maximum_bytes": 256 * 1024, "url": f"https://raw.githubusercontent.com/facebookresearch/dinov2/{CODE_REVISION}/{name}"} for name, checksum in CODE_HASHES.items()})
    return result


def no_symlink_path(path: Path) -> Path:
    absolute = path.absolute()
    if any(parent.is_symlink() for parent in [absolute, *absolute.parents]):
        raise ValueError("App-owned model paths must not traverse symbolic links")
    return absolute


def verify_owned_encoder(path: Path) -> dict:
    path = no_symlink_path(path)
    marker = json.loads((path / OWNED_ENCODER).read_text())
    expected = encoder_files()
    if marker.get("schema") != SCHEMA or marker.get("owned_directory") != str(path.resolve()) or marker.get("files") != {name: item["sha256"] for name, item in expected.items()}:
        raise ValueError("Directory lacks the exact Texture Studio encoder ownership identity")
    actual = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()}
    directories = {str(directory.relative_to(path)) for directory in path.rglob("*") if directory.is_dir()}
    expected_directories = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."}
    if actual != set(expected) | {OWNED_ENCODER} or directories != expected_directories or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("Owned encoder contains unrecognized files or symbolic links; removal refused")
    for name, item in expected.items():
        file = path / name
        if digest(file) != item["sha256"] or ("bytes" in item and file.stat().st_size != item["bytes"]):
            raise ValueError("Owned encoder bytes changed; removal refused")
    return marker


def download_checked(url: str, path: Path, checksum: str, maximum: int, exact: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "TextureStudio/MaterialWorkbench"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("xb") as stream:
        if not response.geturl().startswith("https://"):
            raise ValueError("Pinned encoder download redirected outside HTTPS")
        count = 0
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > maximum:
                raise ValueError("Pinned encoder download exceeded its exact byte bound")
            stream.write(chunk)
    if (exact is not None and count != exact) or digest(path) != checksum:
        raise ValueError("Pinned encoder download checksum/size mismatch")


def install_encoder(args) -> dict:
    destination = no_symlink_path(args.destination)
    if destination.exists():
        verify_owned_encoder(destination)
        return {"directory": str(destination), "model_directory": str(destination / "model"), "code_directory": str(destination / "code"), "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".texture-studio-encoder-", dir=destination.parent))
    reserved = False
    try:
        for name, item in encoder_files().items():
            print(json.dumps({"event": "download", "file": name}), file=sys.stderr, flush=True)
            download_checked(item["url"], stage / name, item["sha256"], item.get("bytes", item.get("maximum_bytes")), item.get("bytes"))
        atomic_bytes(stage / OWNED_ENCODER, json_bytes({"schema": SCHEMA, "owned_directory": str(destination.resolve()), "files": {name: item["sha256"] for name, item in encoder_files().items()}, "license": "Apache-2.0", "revision": MODEL_REVISION, "official_code_revision": CODE_REVISION}))
        # Reserve a new directory exclusively after all bytes verify. Publishing
        # children into this owned empty reservation cannot overwrite a raced
        # pre-existing directory (POSIX rename alone can replace an empty one).
        destination.mkdir()
        reserved = True
        for child in stage.iterdir():
            if child.name != OWNED_ENCODER:
                os.rename(child, destination / child.name)
        os.rename(stage / OWNED_ENCODER, destination / OWNED_ENCODER)
        verify_owned_encoder(destination)
    except BaseException:
        if reserved and destination.exists():
            # This directory was exclusively created by this invocation after
            # verified downloads; cancellation or failure contains only our bytes.
            shutil.rmtree(destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"directory": str(destination), "model_directory": str(destination / "model"), "code_directory": str(destination / "code"), "reused": False, "weights_sha256": MODEL_SHA256}


def remove_encoder(args) -> dict:
    directory = no_symlink_path(args.directory)
    verify_owned_encoder(directory)
    removed = directory.with_name(".texture-studio-remove-" + next(tempfile._get_candidate_names()))
    os.rename(directory, removed)
    shutil.rmtree(removed)
    return {"directory": str(directory), "removed": True, "external_model_directories_modified": False}


def upload_package(args, api_factory=None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo):
        raise ValueError("Choose an explicit Hugging Face repository as owner/model-name")
    root = no_symlink_path(args.package)
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    expected = {"material-head.pt", "config.json", "README.md", "LICENSE"}
    if set(hashes) != expected or {p.name for p in root.iterdir()} != expected | {"SHA256SUMS.json"}:
        raise ValueError("Upload only a complete bounded material-head package; source photos are never included")
    for name, checksum in hashes.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError("Package changed since export; re-export it before upload")
    checkpoint_snapshot(root / "material-head.pt")
    if api_factory is None:
        from huggingface_hub import HfApi
        api_factory = HfApi
    api = api_factory()
    try:
        api.whoami()  # Saved/environment token; never print or return it.
    except Exception:
        raise ValueError("Hugging Face authentication unavailable; run hf auth login, then retry") from None
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", private=not args.public, exist_ok=True)
        info = api.repo_info(repo_id=args.repo, repo_type="model")
        if bool(info.private) != (not args.public):
            raise ValueError("Existing repository visibility differs from the explicit upload choice")
        commit = api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(root), commit_message="Upload Texture Studio material inference head", allow_patterns=sorted(expected | {"SHA256SUMS.json"}))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Hugging Face upload failed ({type(error).__name__}); verify repository access and retry") from None
    return {"repository": args.repo, "private": not args.public, "url": f"https://huggingface.co/{args.repo}", "commit_url": getattr(commit, "commit_url", None), "source_photos_uploaded": False}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("dataset", "prepare-size", "curate", "checkpoint", "infer", "package", "upload", "install-encoder", "remove-encoder"):
        sub = commands.add_parser(name)
        if name == "upload":
            sub.add_argument("--package", type=Path, required=True)
            sub.add_argument("--repo", required=True)
            sub.add_argument("--public", action="store_true")
        if name == "install-encoder":
            sub.add_argument("--destination", type=Path, required=True)
        if name == "remove-encoder":
            sub.add_argument("--directory", type=Path, required=True)
        if name in ("dataset", "prepare-size", "curate"):
            sub.add_argument("--dataset", type=Path, required=True)
        if name == "prepare-size":
            sub.add_argument("--size", type=int, choices=(1024, 2048), required=True)
            sub.add_argument("--expected-index-sha256")
            sub.add_argument("--automatic-validation", action="store_true")
            sub.add_argument("--material", help="Prepare one automatic check crop for this quick-fit material, retaining all training materials")
            sub.add_argument("--workers", type=int, help="Optional upper bound; CPU/memory limits still apply")
        if name == "curate":
            sub.add_argument("--sample", required=True)
            sub.add_argument("--status", choices=("approved", "excluded", "unreviewed"), required=True)
            sub.add_argument("--split", choices=("train", "validation"))
            sub.add_argument("--note")
            sub.add_argument("--expected-index-sha256")
        if name in ("checkpoint", "infer", "package"):
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--expected-sha256")
        if name in ("infer", "package"):
            sub.add_argument("--output", type=Path, required=True)
        if name == "infer":
            sub.add_argument("--image", type=Path, required=True)
            sub.add_argument("--device", choices=("mps", "cpu"), default="mps")
            sub.add_argument("--input-encoding", choices=("auto", "srgb", "linear"), default="auto")
            sub.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-base-" + MODEL_REVISION[:12]))
            sub.add_argument("--code-directory", type=Path, default=ROOT / "out/material-training/transfer-models" / ("dinov2-code-" + CODE_REVISION[:12]))
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = {"dataset": dataset_info, "prepare-size": prepare_size, "curate": curate, "checkpoint": checkpoint_info, "infer": infer, "package": package, "upload": upload_package, "install-encoder": install_encoder, "remove-encoder": remove_encoder}[args.command](args)
        print(json.dumps(dict(result, schema=SCHEMA if args.command != "checkpoint" else result["schema"], protocol_schema=SCHEMA, command=args.command, ok=True), allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        details = {"code": type(error).__name__, "message": str(error)}
        if hasattr(error, "details"):
            details["details"] = error.details
        print(json.dumps({"schema": SCHEMA, "command": args.command, "ok": False, "error": details}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
