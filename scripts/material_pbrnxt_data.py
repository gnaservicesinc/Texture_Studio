"""Native paired material data and displacement losses; never pad or resize."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from material_dataset import png_image_header, read_png, rectangles_overlap, resolve_map_path, write_png


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its folder")
    return path


def input_variant_path(folder: Path, descriptor: dict) -> Path:
    """Use the same bound-source/contained-crop contract as the primary input."""
    name = descriptor.get("path") or descriptor.get("filename")
    return resolve_map_path(folder, {"maps": {"input": name}, "map_metadata": {"input": descriptor}}, "input")


def select_input_variant(pair: dict, rng: random.Random | None = None) -> dict:
    """Choose real registered color evidence without changing the target pixels.

    The caller supplies the training RNG. Validation/review can instead use a
    sample-bound seeded RNG, so base and refined predictions see the same color.
    """
    variants = pair.get("input_variants")
    if not variants:
        return pair
    selected = variants[rng.randrange(len(variants))] if rng is not None else variants[0]
    return dict(pair, paths=dict(pair["paths"], input=selected["path"]),
                sha256=dict(pair["sha256"], input=selected["descriptor"]["sample_sha256"]),
                metadata=dict(pair["metadata"], map_metadata=dict(pair["metadata"]["map_metadata"], input=selected["descriptor"])),
                input_variant=selected["descriptor"])


def select_pairs(dataset: Path, materials: list[str] | None = None, expected_size: int | None = None,
                 target: str = "height") -> tuple[list[dict], list[dict], dict]:
    if target not in ("height", "roughness", "normal"):
        raise ValueError("Select a material height, roughness, or normal target")
    index_path = dataset / "dataset.json"
    index = json.loads(index_path.read_text())
    requested = set(materials) if materials else {
        item["material_id"] for item in index["samples"]
        if item["split"] == "train" and item["status"] not in ("excluded", "rejected")
    }
    pairs = []
    sample_ids = set()
    for item in index["samples"]:
        if (item["material_id"] not in requested and (materials is not None or item["split"] != "validation")) or item["status"] in ("excluded", "rejected"):
            continue
        if item["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate dataset sample identity: {item['sample_id']}")
        sample_ids.add(item["sample_id"])
        folder = contained(dataset, item["path"])
        metadata = json.loads((folder / "sample.json").read_text())
        if any(metadata.get(key) != item.get(key) for key in ("sample_id", "material_id", "split", "status")):
            raise ValueError(f"Dataset/sample identity differs: {folder}")
        if metadata["split"] not in ("train", "validation"):
            raise ValueError(f"Unknown dataset split: {folder}")
        if not metadata.get("crop_values_verified") or not metadata.get("source_precision_verified"):
            raise ValueError(f"Unverified source crops: {folder}")
        if (target not in metadata.get("maps", {})
                or (metadata.get("available_targets") is not None and target not in metadata["available_targets"])):
            # Older sets with only color/normal/roughness remain useful for
            # their actual targets; they cannot fabricate displacement data.
            continue
        maps, identities, dimensions = {}, {}, []
        for role in ("input", target):
            path = resolve_map_path(folder, metadata, role)
            header = png_image_header(path)
            dimensions.append((header["width"], header["height"]))
            details = metadata["map_metadata"][role]
            identity = digest(path)
            if identity != details["sample_sha256"]:
                raise ValueError(f"Source crop changed: {path}")
            if details.get("sample_bits") != header["sample_bits"] or details.get("channels") != header["channels"]:
                raise ValueError(f"Actual map precision differs from metadata: {path}")
            if role == "height" and (header["sample_bits"] != 16 or details.get("encoding") != "linear_data"):
                raise ValueError(f"Height needs native UInt16 linear numeric codes: {path}")
            if role != "input" and details.get("encoding") != "linear_data":
                raise ValueError(f"Material targets require linear numeric codes: {path}")
            maps[role], identities[role] = path, identity
        if dimensions[0] != dimensions[1] or list(dimensions[0]) != metadata["sample_pixel_dimensions"]:
            raise ValueError(f"Actual paired dimensions disagree with metadata: {folder}")
        if expected_size is not None and dimensions[0] != (expected_size, expected_size):
            raise ValueError(f"Every training map must be exactly {expected_size}×{expected_size}; prepare that size from the original sources: {folder}")
        for role in metadata["maps"].keys() - maps.keys():
            header = png_image_header(resolve_map_path(folder, metadata, role))
            if (header["width"], header["height"]) != dimensions[0]:
                raise ValueError(f"Every paired map must match the selected training grid; regenerate from originals: {folder}/{role}")
        variants = []
        seen_variants = set()
        for descriptor in metadata.get("input_variants", []):
            path = input_variant_path(folder, descriptor)
            header = png_image_header(path)
            actual = digest(path)
            if actual != descriptor.get("sample_sha256"):
                raise ValueError(f"Diffuse color variant changed: {path}")
            if (header["width"], header["height"]) != dimensions[0]:
                raise ValueError(f"Every diffuse color variant must match the exact training grid: {path}")
            if any(descriptor.get(key) != header[key] for key in ("sample_bits", "channels")):
                raise ValueError(f"Diffuse color variant precision differs from metadata: {path}")
            if header["channels"] not in (3, 4):
                raise ValueError(f"Diffuse color variant needs native RGB(A): {path}")
            if descriptor.get("encoding") != metadata["map_metadata"]["input"]["encoding"]:
                raise ValueError(f"Diffuse color variants must use the same explicit color transfer: {path}")
            source = descriptor.get("source", {})
            if source and (source.get("width"), source.get("height")) != tuple(metadata["source_pixel_dimensions"]):
                raise ValueError(f"Diffuse color variant source geometry differs: {path}")
            crop = descriptor.get("crop_rectangle_top_left_xywh")
            if crop is not None and crop != metadata["crop_rectangle_top_left_xywh"]:
                raise ValueError(f"Diffuse color variant crop differs from its numeric target: {path}")
            if descriptor.get("transforms", []) != metadata["map_metadata"]["input"].get("transforms", []):
                raise ValueError(f"Diffuse color variant preprocessing differs from its paired input: {path}")
            identity_key = (str(path.resolve()), actual)
            if identity_key not in seen_variants:
                seen_variants.add(identity_key)
                variants.append({"path": path, "descriptor": descriptor})
        if variants and (str(maps["input"].resolve()), identities["input"]) not in seen_variants:
            raise ValueError("Registered diffuse variants must include the primary paired input")
        rectangle, source_dimensions = metadata["crop_rectangle_top_left_xywh"], metadata["source_pixel_dimensions"]
        if (len(rectangle) != 4 or len(source_dimensions) != 2
                or any(type(value) is not int for value in (*rectangle, *source_dimensions))
                or min(rectangle[:2]) < 0 or min(source_dimensions) < 1
                or (tuple(rectangle[2:]) != dimensions[0] and not metadata.get("native_size_preparation", {}).get("target_resized"))
                or rectangle[0] + rectangle[2] > source_dimensions[0]
                or rectangle[1] + rectangle[3] > source_dimensions[1]):
            raise ValueError(f"Crop provenance differs from real source dimensions: {folder}")
        pairs.append({"metadata": metadata, "paths": maps, "sha256": identities, "dimensions": dimensions[0], "target": target,
                      "input_variants": variants})
    training = [pair for pair in pairs if pair["metadata"]["split"] == "train"]
    checks = [pair for pair in pairs if pair["metadata"]["split"] == "validation"]
    found = {pair["metadata"]["material_id"] for pair in training}
    if materials is None:
        requested = found
    if not training or requested - found:
        raise ValueError("No included training pairs" + (" for: " + ", ".join(sorted(requested - found)) if requested - found else ""))
    if len({pair["dimensions"] for pair in pairs}) != 1:
        raise ValueError("Training and check maps must all have the same exact dimensions; prepare the selected size from originals")
    for check in checks:
        for train in training:
            check_family = check["metadata"].get("source_family_id") or check["metadata"].get("asset_family_id") or check["metadata"]["material_id"]
            train_family = train["metadata"].get("source_family_id") or train["metadata"].get("asset_family_id") or train["metadata"]["material_id"]
            if check_family == train_family or (check["metadata"].get("source_directory") and check["metadata"].get("source_directory") == train["metadata"].get("source_directory")):
                from material_native_size import normalized_rectangle
                check_rect = normalized_rectangle(check["metadata"]["crop_rectangle_top_left_xywh"], check["metadata"]["source_pixel_dimensions"])
                train_rect = normalized_rectangle(train["metadata"]["crop_rectangle_top_left_xywh"], train["metadata"]["source_pixel_dimensions"])
                if index.get("validation_scope") == "known_subject_diagnostic":
                    if check_rect == train_rect:
                        raise ValueError("Learning-check crop repeats an existing training view")
                    continue
                if (check["metadata"].get("source_set_id") != train["metadata"].get("source_set_id")
                        or check["metadata"]["source_pixel_dimensions"] != train["metadata"]["source_pixel_dimensions"]):
                    raise ValueError("Validation would leak the same material across source-resolution sets")
                if rectangles_overlap(check_rect, train_rect):
                    raise ValueError("Check crop overlaps training pixels across source-resolution sets")
    if index.get("split_strategy") == "subject-extra-crops-v2":
        from material_native_size import subject_id
        subjects = [subject_id(pair["metadata"]) for pair in checks]
        if len(subjects) != len(set(subjects)):
            raise ValueError("Validation permits at most one crop per subject folder")
    regional_checks = any((check["metadata"].get("source_family_id") or check["metadata"]["material_id"])
                          == (train["metadata"].get("source_family_id") or train["metadata"]["material_id"])
                          for check in checks for train in training)
    check_scope = ("Held-out source families plus non-overlapping regions of known materials; regional validation does not establish unseen-material generalization"
                   if regional_checks else "Complete held-out source families at the same training resolution")
    if index.get("validation_scope") == "known_subject_diagnostic":
        check_scope = "Known-material learning checks; alternate crops may overlap training pixels. Evaluate novel images separately."
    identity = {"path": str(dataset.resolve()), "index_sha256": digest(index_path),
                "materials": len(found), "training_pairs": len(training), "check_pairs": len(checks),
                "check_scope": check_scope, "validation_scope": index.get("validation_scope"),
                "validation_settings": index.get("validation"), "target": target, "source_bits": (next(iter({pair["metadata"]["map_metadata"][target]["sample_bits"] for pair in pairs}))
                    if len({pair["metadata"]["map_metadata"][target]["sample_bits"] for pair in pairs}) == 1
                    else sorted({pair["metadata"]["map_metadata"][target]["sample_bits"] for pair in pairs})),
                "data_transfer": "native auxiliary integer codes / integer maximum; no gamma or per-image range normalization",
                "pairs": [{"sample_id": pair["metadata"]["sample_id"], "split": pair["metadata"]["split"],
                           "sha256": pair["sha256"], "dimensions": list(pair["dimensions"]),
                           "input_variants": [{"variant_id": item["descriptor"].get("variant_id"),
                                               "path": str(item["path"]), "sha256": item["descriptor"]["sample_sha256"]}
                                              for item in pair["input_variants"]]} for pair in pairs]}
    return training, checks, identity


def numeric_height(codes: np.ndarray) -> np.ndarray:
    if codes.dtype != np.uint16 or codes.ndim not in (2, 3):
        raise ValueError("Height must retain UInt16 source codes")
    if codes.ndim == 3:
        if codes.shape[-1] not in (1, 2, 3, 4) or (codes.shape[-1] >= 3 and np.any(codes[..., :3] != codes[..., :1])):
            raise ValueError("Height must be scalar or identical RGB")
        if codes.shape[-1] in (2, 4) and np.any(codes[..., -1] != 65535):
            raise ValueError("Transparent height maps need review")
        codes = codes[..., 0]
    return codes.astype(np.float32) / np.float32(65535)


def input_rgb(codes: np.ndarray, encoding: str) -> np.ndarray:
    if codes.dtype not in (np.uint8, np.uint16) or codes.ndim != 3 or codes.shape[-1] not in (3, 4):
        raise ValueError("Diffuse input needs native RGB(A) codes")
    maximum = np.iinfo(codes.dtype).max
    if codes.shape[-1] == 4 and np.any(codes[..., 3] != maximum):
        raise ValueError("Transparent crops need review; no fabricated context pixels")
    rgb = codes[..., :3].astype(np.float32) / np.float32(maximum)
    if encoding in ("linear", "linear_rgb", "linear_color", "linear_light"):
        rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    elif encoding not in ("srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"):
        raise ValueError(f"Diffuse color transfer must be explicit: {encoding}")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def numeric_auxiliary(codes: np.ndarray, target: str) -> np.ndarray:
    if target == "height":
        return numeric_height(codes)[None]
    if codes.dtype not in (np.uint8, np.uint16):
        raise ValueError("Material targets require native integer codes")
    maximum = np.iinfo(codes.dtype).max
    if codes.ndim == 2:
        codes = codes[..., None]
    if codes.ndim != 3 or codes.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("Material target requires scalar or RGB(A) integer codes")
    if codes.shape[-1] in (2, 4) and np.any(codes[..., -1] != maximum):
        raise ValueError("Transparent material targets need review")
    if target == "normal":
        if codes.shape[-1] not in (3, 4):
            raise ValueError("Normal map requires three numeric components")
        values = codes[..., :3]
    elif target == "roughness":
        if codes.shape[-1] >= 3 and np.any(codes[..., :3] != codes[..., :1]):
            raise ValueError("Roughness must be scalar or identical RGB")
        values = codes[..., :1]
    else:
        raise ValueError("Unknown numeric material target")
    return np.ascontiguousarray((values.astype(np.float32) / np.float32(maximum)).transpose(2, 0, 1))


class PairCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, capacity)
        self.bytes = 0
        self.entries: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.file_states: dict[tuple, tuple] = {}

    @staticmethod
    def file_state(path: Path) -> tuple:
        state = path.stat()
        return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns

    @staticmethod
    def key(pair: dict) -> tuple:
        target = pair.get("target", "height")
        return (pair["metadata"]["sample_id"], tuple(pair["dimensions"]), target,
                pair["metadata"]["map_metadata"]["input"]["encoding"],
                *((str(pair["paths"][role].resolve()), pair["sha256"][role]) for role in ("input", target)))

    def load(self, pair: dict) -> tuple[torch.Tensor, torch.Tensor]:
        target = pair.get("target", "height")
        key = self.key(pair)
        states = tuple(self.file_state(pair["paths"][role]) for role in ("input", target))
        if key in self.entries:
            if states == self.file_states[key]:
                self.entries.move_to_end(key)
                return self.entries[key]
            removed = self.entries.pop(key)
            self.file_states.pop(key)
            self.bytes -= sum(value.numel() * value.element_size() for value in removed)
        arrays = {}
        for role in ("input", target):
            path = pair["paths"][role]
            arrays[role], metadata = read_png(path)
            if metadata["file_sha256"] != pair["sha256"][role]:
                raise ValueError(f"Dataset changed during training: {path}")
        if states != tuple(self.file_state(pair["paths"][role]) for role in ("input", target)):
            raise ValueError("Dataset changed while decoding training pixels")
        rgb = input_rgb(arrays["input"], pair["metadata"]["map_metadata"]["input"]["encoding"])
        if target == "normal" and any(t.get("type") == "directx_to_opengl" and t.get("applied_in_memory")
                                      for t in pair["metadata"]["map_metadata"][target].get("transforms", [])):
            arrays[target][..., 1] = np.iinfo(arrays[target].dtype).max - arrays[target][..., 1]
        height = numeric_auxiliary(arrays[target], target)
        if rgb.shape[-2:] != height.shape[-2:] or height.shape[-2:][::-1] != pair["dimensions"]:
            raise ValueError("Decoded pair dimensions disagree; no repair by padding/resizing")
        values = torch.from_numpy(rgb[None]), torch.from_numpy(height[None])
        size = sum(value.numel() * value.element_size() for value in values)
        while self.entries and self.bytes + size > self.capacity:
            removed_key, removed = self.entries.popitem(last=False)
            self.file_states.pop(removed_key)
            self.bytes -= sum(value.numel() * value.element_size() for value in removed)
        if size <= self.capacity:
            self.entries[key] = values
            self.file_states[key] = states
            self.bytes += size
        return values


def crop_pair(rgb: torch.Tensor, height: torch.Tensor, size: int, rng: random.Random,
              augment: bool = False, whole_maps: bool = False) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if rgb.shape[-2:] != height.shape[-2:] or size < 64 or size % 64:
        raise ValueError("Paired native crop must be a multiple of64 with matching dimensions")
    rows, columns = height.shape[-2:]
    if (columns, rows) != (size, size):
        raise ValueError(f"Every map must match the selected {size}×{size} training grid; regenerate from originals before training")
    x = y = 0
    width = height_pixels = size
    if augment:
        for axis in (-1, -2):
            if rng.random() < .5:
                rgb, height = rgb.flip(axis), height.flip(axis)
        turns = rng.randrange(4)
        rgb, height = rgb.rot90(turns, (-2, -1)), height.rot90(turns, (-2, -1))
    return rgb.contiguous(), height.contiguous(), [x, y, width, height_pixels]


def height_loss(prediction: torch.Tensor, target: torch.Tensor, margin: int = 16) -> tuple[torch.Tensor, dict[str, float]]:
    if type(margin) is not int or margin < 0:
        raise ValueError("Loss margin must be a nonnegative integer")
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] not in (1, 3) or min(target.shape[-2:]) <= 2 * margin + 16:
        raise ValueError("Loss needs matching native grids and real interior pixels")
    if margin:
        prediction, target = prediction[..., margin:-margin, margin:-margin], target[..., margin:-margin, margin:-margin]
    value = F.l1_loss(prediction, target)
    detail = prediction.new_zeros(())
    for step in (1, 2, 4, 8):
        for axis in (-1, -2):
            length = prediction.shape[axis] - step
            predicted_gradient = prediction.narrow(axis, step, length) - prediction.narrow(axis, 0, length)
            target_gradient = target.narrow(axis, step, length) - target.narrow(axis, 0, length)
            detail = detail + F.l1_loss(predicted_gradient, target_gradient) / step
    total = value + 4 * detail
    return total, {"value_l1": float(value.detach()), "detail_l1": float(detail.detach()), "total": float(total.detach())}


def write_exr(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if (values.ndim != 2 and not (values.ndim == 3 and values.shape[-1] == 3)) or not np.isfinite(values).all():
        raise ValueError("Material EXR requires finite scalar or RGB Float32 data")
    import OpenEXR
    channels = {"Y": values} if values.ndim == 2 else {name:np.ascontiguousarray(values[...,i]) for i,name in enumerate("RGB")}
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage,
                 "textureStudioEncoding": "linear_data"}, channels).write(str(path))
    returned = OpenEXR.File(str(path), separate_channels=True).channels()
    decoded = returned["Y"].pixels if values.ndim == 2 else np.stack([returned[name].pixels for name in "RGB"], -1)
    if decoded.dtype != np.float32 or not np.array_equal(decoded, values):
        raise ValueError("EXR export changed height values")


def write_display(path: Path, values: np.ndarray) -> None:
    # Deliberately display-only. Raw unclipped values are always exported to EXR.
    if (values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] not in (1, 3, 4))
            or not np.isfinite(values).all()):
        raise ValueError("Display PNG requires finite scalar or RGB(A) values")
    codes = np.rint(np.clip(values, 0, 1) * 65535).astype(np.uint16)
    if codes.ndim == 2:
        codes = codes[..., None]
    write_png(path, codes, compression=3)
