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

from material_dataset import png_image_header, read_png, rectangles_overlap, write_png


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


def select_pairs(dataset: Path, materials: list[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    index_path = dataset / "dataset.json"
    index = json.loads(index_path.read_text())
    requested = set(materials) if materials else {
        item["material_id"] for item in index["samples"]
        if item["split"] == "train" and item["status"] not in ("excluded", "rejected")
    }
    pairs = []
    sample_ids = set()
    for item in index["samples"]:
        if item["material_id"] not in requested or item["status"] in ("excluded", "rejected"):
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
        maps, identities, dimensions = {}, {}, []
        for role in ("input", "height"):
            path = contained(folder, metadata["maps"][role])
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
            maps[role], identities[role] = path, identity
        if dimensions[0] != dimensions[1] or list(dimensions[0]) != metadata["sample_pixel_dimensions"]:
            raise ValueError(f"Actual paired dimensions disagree with metadata: {folder}")
        rectangle, source_dimensions = metadata["crop_rectangle_top_left_xywh"], metadata["source_pixel_dimensions"]
        if (len(rectangle) != 4 or len(source_dimensions) != 2
                or any(type(value) is not int for value in (*rectangle, *source_dimensions))
                or min(rectangle[:2]) < 0 or min(source_dimensions) < 1
                or tuple(rectangle[2:]) != dimensions[0]
                or rectangle[0] + rectangle[2] > source_dimensions[0]
                or rectangle[1] + rectangle[3] > source_dimensions[1]):
            raise ValueError(f"Crop provenance differs from real source dimensions: {folder}")
        pairs.append({"metadata": metadata, "paths": maps, "sha256": identities, "dimensions": dimensions[0]})
    training = [pair for pair in pairs if pair["metadata"]["split"] == "train"]
    checks = [pair for pair in pairs if pair["metadata"]["split"] == "validation"]
    found = {pair["metadata"]["material_id"] for pair in training}
    if not training or requested - found:
        raise ValueError("No included training pairs" + (" for: " + ", ".join(sorted(requested - found)) if requested - found else ""))
    for check in checks:
        for train in training:
            if check["metadata"]["material_id"] == train["metadata"]["material_id"]:
                if rectangles_overlap(check["metadata"]["crop_rectangle_top_left_xywh"], train["metadata"]["crop_rectangle_top_left_xywh"]):
                    raise ValueError("Check crop overlaps training pixels")
    identity = {"path": str(dataset.resolve()), "index_sha256": digest(index_path),
                "materials": len(found), "training_pairs": len(training), "check_pairs": len(checks),
                "check_scope": "Unique regions of known materials; does not establish unseen-material generalization",
                "source_bits": 16, "data_transfer": "height UInt16 /65535; no gamma or per-image range normalization",
                "pairs": [{"sample_id": pair["metadata"]["sample_id"], "split": pair["metadata"]["split"],
                           "sha256": pair["sha256"], "dimensions": list(pair["dimensions"])} for pair in pairs]}
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
        raise ValueError("Photo input needs native RGB(A) codes")
    maximum = np.iinfo(codes.dtype).max
    if codes.shape[-1] == 4 and np.any(codes[..., 3] != maximum):
        raise ValueError("Transparent crops need review; no fabricated context pixels")
    rgb = codes[..., :3].astype(np.float32) / np.float32(maximum)
    if encoding in ("linear", "linear_rgb", "linear_color", "linear_light"):
        rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
    elif encoding not in ("srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"):
        raise ValueError(f"Diffuse color transfer must be explicit: {encoding}")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


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
        return (pair["metadata"]["sample_id"], tuple(pair["dimensions"]),
                pair["metadata"]["map_metadata"]["input"]["encoding"],
                *((str(pair["paths"][role].resolve()), pair["sha256"][role]) for role in ("input", "height")))

    def load(self, pair: dict) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.key(pair)
        states = tuple(self.file_state(pair["paths"][role]) for role in ("input", "height"))
        if key in self.entries:
            if states == self.file_states[key]:
                self.entries.move_to_end(key)
                return self.entries[key]
            removed = self.entries.pop(key)
            self.file_states.pop(key)
            self.bytes -= sum(value.numel() * value.element_size() for value in removed)
        arrays = {}
        for role in ("input", "height"):
            path = pair["paths"][role]
            arrays[role], metadata = read_png(path)
            if metadata["file_sha256"] != pair["sha256"][role]:
                raise ValueError(f"Dataset changed during training: {path}")
        if states != tuple(self.file_state(pair["paths"][role]) for role in ("input", "height")):
            raise ValueError("Dataset changed while decoding training pixels")
        rgb = input_rgb(arrays["input"], pair["metadata"]["map_metadata"]["input"]["encoding"])
        height = numeric_height(arrays["height"])
        if rgb.shape[-2:] != height.shape or height.shape[::-1] != pair["dimensions"]:
            raise ValueError("Decoded pair dimensions disagree; no repair by padding/resizing")
        values = torch.from_numpy(rgb[None]), torch.from_numpy(height[None, None])
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
              augment: bool = False) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if rgb.shape[-2:] != height.shape[-2:] or size < 64 or size % 64:
        raise ValueError("Paired native crop must be a multiple of64 with matching dimensions")
    rows, columns = height.shape[-2:]
    if min(rows, columns) < size:
        raise ValueError("Crop exceeds real source pixels; choose a lower native size")
    x, y = rng.randrange(columns - size + 1), rng.randrange(rows - size + 1)
    rgb, height = rgb[..., y:y + size, x:x + size], height[..., y:y + size, x:x + size]
    if augment:
        for axis in (-1, -2):
            if rng.random() < .5:
                rgb, height = rgb.flip(axis), height.flip(axis)
        turns = rng.randrange(4)
        rgb, height = rgb.rot90(turns, (-2, -1)), height.rot90(turns, (-2, -1))
        # Only a derived photograph changes. Numeric target codes stay intact.
        rgb = (rgb * (2 ** rng.uniform(-.2, .2))).clamp(0, 1)
    return rgb.contiguous(), height.contiguous(), [x, y, size, size]


def height_loss(prediction: torch.Tensor, target: torch.Tensor, margin: int = 16) -> tuple[torch.Tensor, dict[str, float]]:
    if type(margin) is not int or margin < 0:
        raise ValueError("Loss margin must be a nonnegative integer")
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1 or min(target.shape[-2:]) <= 2 * margin + 16:
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
    import OpenEXR
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Height EXR requires finite scalar Float32 data")
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage,
                 "textureStudioEncoding": "linear_data"}, {"Y": values}).write(str(path))
    decoded = OpenEXR.File(str(path), separate_channels=True).channels()["Y"].pixels
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
