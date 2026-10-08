#!/usr/bin/env python3
"""Prepare verifiable native material crops without modifying source images.

PNG decoding uses OpenCV's unchanged integer path, never Pillow's RGB decoder.
The header, sample type, channel order and round trip are checked explicitly.
Color tags are evidence, not permission to alter numerical map values.
"""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import unicodedata
import zlib

import cv2
import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_CHUNKS = {b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"cICP"}
MAP_NAMES = {"input": "diffuse.png", "height": "displacement.png",
             "normal": "normal.png", "roughness": "roughness.png"}
MAP_ROLES = {"diff": "input", "diffuse": "input", "disp": "height", "disp_gl": "height",
             "displacement": "height", "nor_gl": "normal", "nor_dx": "normal",
             "rough": "roughness", "roughness": "roughness"}
KNOWN_SUFFIXES = "nor_gl|nor_dx|disp_gl|rough_ao|translucent|diffuse|displacement|roughness|diff|disp|rough|anisotropy_rotation|anisotropy_strength|spec_ior|spec|bump|metal|ao|arm"
FILE_PATTERN = re.compile(r"^(.+?)_(" + KNOWN_SUFFIXES + r")_(\d+k)\.png$", re.I)
AMBIENT_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_([1248]K)-PNG_(Color|Displacement|NormalGL|NormalDX|Roughness)\.png$", re.I)
AMBIENT_SUFFIXES = {"color": "diff", "displacement": "disp", "normalgl": "nor_gl", "normaldx": "nor_dx", "roughness": "rough"}
GENERATOR = "ipde-material-dataset-v2"


def source_cache_budget() -> int:
    """Retain reusable source arrays within the current machine's headroom.

    The same reserve as training leaves room for macOS and for decoding the
    active parent, whose temporary buffers do not belong to the retained cache.
    """
    from material_resources import available_memory_bytes, os_reserve_bytes, physical_memory_bytes
    reserve = os_reserve_bytes(physical_memory_bytes())
    return max(0, available_memory_bytes() - reserve)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    """Canonical RGB/HWC little-endian integer sample hash, independent of host."""
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    # Hash the contiguous allocation directly instead of copying every crop
    # into a second, equally large Python bytes object.
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not slug:
        raise ValueError(f"Name has no usable material identifier: {value!r}")
    return slug


def parse_source_filename(filename: str) -> tuple[str, str, str] | None:
    """Recognize original provider names without changing filenames or bytes."""
    match = FILE_PATTERN.fullmatch(filename)
    if match:
        return match.groups()
    match = AMBIENT_PATTERN.fullmatch(filename)
    if match:
        asset, resolution, role = match.groups()
        return asset, AMBIENT_SUFFIXES[role.casefold()], resolution.casefold()
    return None


def png_chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file")
    position = 8
    while position < len(data):
        if len(data) - position < 12:
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack_from(">I", data, position)[0]
        end = position + 12 + size
        if end > len(data):
            raise ValueError("Truncated PNG payload")
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        expected = struct.unpack_from(">I", data, position + 8 + size)[0]
        if zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF != expected:
            raise ValueError(f"Invalid PNG CRC in {kind!r}")
        yield kind, payload
        position = end
        if kind == b"IEND":
            if position != len(data):
                raise ValueError("Unexpected bytes following PNG IEND")
            return
    raise ValueError("PNG is missing IEND")


def png_metadata(data: bytes) -> dict:
    result: dict = {"file_sha256": sha256_bytes(data), "file_md5": hashlib.md5(data, usedforsecurity=False).hexdigest(), "color_chunks": []}
    for kind, payload in png_chunks(data):
        if kind == b"IHDR":
            if len(payload) != 13:
                raise ValueError("Invalid PNG IHDR")
            width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
            if bits not in (8, 16) or channels is None:
                raise ValueError("Only native 8/16-bit grayscale, RGB and alpha PNG maps are supported")
            result.update(width=width, height=height, sample_bits=bits, channels=channels,
                          png_color_type=color, interlace=interlace)
        elif kind in COLOR_CHUNKS:
            result["color_chunks"].append({"type": kind.decode(), "data_base64": base64.b64encode(payload).decode()})
            if kind == b"gAMA" and len(payload) == 4:
                result["png_gamma"] = struct.unpack(">I", payload)[0] / 100000
            elif kind == b"sRGB" and len(payload) == 1:
                result["srgb_rendering_intent"] = payload[0]
            elif kind == b"iCCP":
                name, separator, rest = payload.partition(b"\x00")
                result["icc_profile_name"] = name.decode("latin1")
                result["icc_profile_payload_sha256"] = sha256_bytes(payload)
        elif kind == b"eXIf":
            result["exif_present"] = True
    if "width" not in result:
        raise ValueError("PNG is missing IHDR")
    return result


def read_png(path: Path | str) -> tuple[np.ndarray, dict]:
    """Return native HWC RGB integer samples and inspected PNG metadata."""
    data = Path(path).read_bytes()
    metadata = png_metadata(data)
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"PNG decoder failed: {path}")
    expected_dtype = np.dtype(np.uint16 if metadata["sample_bits"] == 16 else np.uint8)
    if decoded.dtype != expected_dtype:
        raise ValueError(f"Decoder changed sample depth: {path}")
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    elif decoded.shape[2] == 3:
        decoded = decoded[:, :, ::-1]
    elif decoded.shape[2] == 4:
        decoded = decoded[:, :, [2, 1, 0, 3]]
    if metadata["png_color_type"] == 4 and decoded.ndim == 3 and decoded.shape[2] == 4:
        # OpenCV expands grayscale+alpha to RGBA. Reconstruct both original
        # integer channels only after proving that RGB is an exact replication.
        if not np.array_equal(decoded[..., 0], decoded[..., 1]) or not np.array_equal(decoded[..., 0], decoded[..., 2]):
            raise ValueError(f"Decoder changed grayscale+alpha values: {path}")
        decoded = decoded[..., [0, 3]]
    if decoded.shape != (metadata["height"], metadata["width"], metadata["channels"]):
        raise ValueError(f"Decoder changed PNG shape or channels: {path}")
    return np.ascontiguousarray(decoded), metadata


class SourceDecodeCache:
    """Bounded, per-verification integer parent cache; never used by training.

    The budget covers retained decoded arrays and their Python metadata objects.
    Decoding the active source still requires its own temporary working memory.
    """
    def __init__(self, max_bytes: int | None = None):
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("Source decode cache must be a nonnegative byte count; zero disables retention")
        capacity = source_cache_budget()
        self.max_bytes = capacity if max_bytes is None else min(max_bytes, capacity)
        self.requested_bytes = max_bytes
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.entries = OrderedDict()

    @staticmethod
    def key(path: Path) -> tuple:
        canonical = path.resolve()
        state = canonical.stat()
        return (str(canonical), state.st_dev, state.st_ino, state.st_size,
                state.st_mtime_ns, state.st_ctime_ns)

    def _discard(self, key: tuple) -> None:
        _, _, retained_bytes = self.entries.pop(key)
        self.retained_bytes -= retained_bytes
        self.evictions += 1

    @staticmethod
    def metadata_bytes(value) -> int:
        """Count retained Python metadata objects rather than compressed JSON."""
        seen = set()
        def count(item):
            if id(item) in seen:
                return 0
            seen.add(id(item))
            size = sys.getsizeof(item)
            if isinstance(item, dict):
                size += sum(count(key) + count(child) for key, child in item.items())
            elif isinstance(item, (list, tuple)):
                size += sum(count(child) for child in item)
            return size
        return count(value)

    def read(self, path: Path | str) -> tuple[np.ndarray, dict]:
        key = self.key(Path(path))
        if key in self.entries:
            self.hits += 1
            array, metadata, _ = self.entries[key]
            self.entries.move_to_end(key)
            return array, metadata
        # Changed parents must not coexist with stale cached versions.
        for previous in list(self.entries):
            if previous[0] == key[0]:
                self._discard(previous)
        self.misses += 1
        array, metadata = read_png(key[0])
        if self.key(Path(path)) != key:
            raise ValueError(f"Source changed while decoding for verification: {path}")
        array_header = max(0, sys.getsizeof(array) - array.nbytes)
        retained_bytes = array.nbytes + array_header + self.metadata_bytes(metadata) + self.metadata_bytes(key) + 256
        if retained_bytes <= self.max_bytes:
            while self.entries and self.retained_bytes + retained_bytes > self.max_bytes:
                self._discard(next(iter(self.entries)))
            array.setflags(write=False)
            self.entries[key] = (array, metadata, retained_bytes)
            self.retained_bytes += retained_bytes
            self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        return array, metadata

    def report(self) -> dict:
        return {"max_retained_bytes": self.max_bytes, "requested_bytes": self.requested_bytes,
                "budget_policy": "available_memory_minus_system_and_decode_reserve",
                "retained_bytes": self.retained_bytes,
                "peak_retained_bytes": self.peak_retained_bytes, "entries": len(self.entries),
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "key_fields": ["canonical_path", "device", "inode", "size", "mtime_ns", "ctime_ns"],
                "scope": "Source verification only; decode working buffers are outside retained cache budget"}


def published_source_provenance(source: dict, material_id: str, audit: dict | None) -> dict:
    """Attach CC0 evidence only after matching current bytes to an API audit."""
    if not audit:
        return {}
    for entry in audit.get("files", []):
        api_url = entry.get("api_url") or entry.get("api_provenance", {}).get("api_url")
        published_md5 = entry.get("published_md5")
        local_md5 = source.get("file_md5")
        md5_matches = (isinstance(published_md5, str) and isinstance(local_md5, str)
                       and re.fullmatch(r"[0-9a-fA-F]{32}", published_md5) is not None
                       and re.fullmatch(r"[0-9a-fA-F]{32}", local_md5) is not None
                       and published_md5.casefold() == local_md5.casefold())
        if (entry.get("asset_id") == material_id and entry.get("filename") == source["filename"]
                and entry.get("published_exact_match") is True
                and md5_matches
                and entry.get("published_bytes") == source.get("file_bytes")
                and api_url == "https://api.polyhaven.com/files/" + material_id
                and entry.get("published_url", "").startswith("https://dl.polyhaven.org/file/ph-assets/Textures/")):
            return {"provider": "Poly Haven", "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
                    "license_basis": "Poly Haven library asset; current full-file MD5 and byte count match the official API audit",
                    "published_url": entry["published_url"], "published_md5": entry["published_md5"],
                    "published_bytes": entry["published_bytes"], "published_api_url": api_url,
                    "published_audit_timestamp_utc": audit.get("snapshot_completed_utc") or audit.get("timestamp_utc")}
    from material_recreation import package_download_evidence
    package_evidence = package_download_evidence(source, material_id, audit.get("package_audits", []))
    if package_evidence:
        return {"provider": "ambientCG", "asset_id": package_evidence["asset_id"],
                "asset_url": package_evidence["asset_url"], "license": "CC0-1.0",
                "license_url": package_evidence["license"]["url"],
                "license_basis": "Current full source SHA256 and byte count match a verified member of the official ambientCG archive",
                "creation_method": package_evidence.get("creation_method"),
                "published_package_url": package_evidence["url"],
                "published_package_sha256": package_evidence["archive_sha256"],
                "published_package_bytes": package_evidence["archive_bytes"],
                "published_archive_member": package_evidence["archive_member"],
                "published_member_sha256": package_evidence["member_sha256"],
                "published_member_bytes": package_evidence["member_bytes"],
                "published_audit_timestamp_utc": package_evidence["audit_time_utc"]}
    return {}


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_png(path: Path | str, array: np.ndarray, metadata: dict | None = None, compression: int = 3) -> None:
    """Losslessly encode native integer samples and preserve declared color tags."""
    if type(compression) is not int or not 0 <= compression <= 9:
        raise ValueError("PNG compression must be an integer between 0 and 9")
    if array.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or array.ndim != 3:
        raise ValueError("PNG output requires uint8/uint16 HWC samples")
    channels = array.shape[2]
    if channels == 2:
        # OpenCV has no two-channel PNG encoder. Encode native gray+alpha
        # scanlines directly, preserving both components and their precision.
        height, width = array.shape[:2]
        if min(height, width) <= 0:
            raise ValueError("PNG dimensions must be positive")
        header = struct.pack(">IIBBBBB", width, height, array.dtype.itemsize * 8, 4, 0, 0, 0)
        rows = b"".join(b"\0" + row.astype(array.dtype.newbyteorder(">"), copy=False).tobytes() for row in array)
        png_data = PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows, compression)) + _chunk(b"IEND", b"")
    elif channels == 1:
        encoded_array = array[:, :, 0]
    elif channels == 3:
        encoded_array = array[:, :, ::-1]
    elif channels == 4:
        encoded_array = array[:, :, [2, 1, 0, 3]]
    else:
        raise ValueError("PNG encoder supports grayscale, grayscale+alpha, RGB and RGBA")
    if channels != 2:
        success, encoded = cv2.imencode(".png", np.ascontiguousarray(encoded_array), [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not success:
            raise ValueError("PNG encoder failed")
        png_data = encoded.tobytes()
    result = bytearray(PNG_SIGNATURE)
    for kind, payload in png_chunks(png_data):
        result.extend(_chunk(kind, payload))
        if kind == b"IHDR" and metadata:
            for item in metadata.get("color_chunks", []):
                result.extend(_chunk(item["type"].encode(), base64.b64decode(item["data_base64"])))
    with Path(path).open("xb") as stream:
        stream.write(result)


def source_summary(path: Path) -> dict:
    """Inspect without allocating decoded image memory."""
    metadata = png_metadata(path.read_bytes())
    metadata["filename"] = path.name
    metadata["file_bytes"] = path.stat().st_size
    return metadata


def discover_materials(sources: Path, crop_size: int, dimension_policy: str = "strict") -> list[dict]:
    discovered = []
    identifiers: set[str] = set()
    for folder in sorted((item for item in sources.iterdir() if item.is_dir()), key=lambda item: item.name.casefold()):
        candidates: dict[str, dict] = {}
        file_material_ids: set[str] = set()
        ignored = []
        problems = []
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            parsed = parse_source_filename(path.name)
            if not parsed:
                ignored.append(path.name)
                continue
            file_id, suffix, resolution = parsed
            file_material_ids.add(slugify(file_id))
            role = MAP_ROLES.get(suffix.lower(), "extra:" + slugify(suffix))
            # Prefer OpenGL if both convention variants are present.
            if role == "normal" and role in candidates and suffix.lower() == "nor_dx":
                ignored.append(path.name)
                continue
            if role in candidates and not (role == "normal" and suffix.lower() == "nor_gl" and candidates[role]["suffix"] == "nor_dx"):
                problems.append(f"Multiple source maps for {role}")
                continue
            try:
                header = source_summary(path)
            except (OSError, ValueError) as error:
                problems.append(f"{path.name}: {error}")
                continue
            candidates[role] = {"path": str(path.resolve()), "suffix": suffix.lower(), "resolution_label": resolution.lower(), **header}
        material_id = next(iter(file_material_ids)) if len(file_material_ids) == 1 else slugify(folder.name)
        if len(file_material_ids) > 1:
            problems.append("Files identify multiple original materials")
        if material_id in identifiers:
            problems.append("Duplicate normalized material identity; refusing merged materials")
        identifiers.add(material_id)
        for role in MAP_NAMES:
            if role not in candidates:
                problems.append(f"Missing {role} map")
        if "height" in candidates and candidates["height"]["sample_bits"] < 16:
            problems.append("High-detail displacement requires original 16-bit samples; 8-bit height is excluded")
        dimensions = {(item["width"], item["height"]) for item in candidates.values()}
        if len(dimensions) > 1 and dimension_policy == "strict":
            problems.append("Paired map dimensions differ; registration must be reviewed")
        if candidates:
            width = min(item["width"] for item in candidates.values())
            height = min(item["height"] for item in candidates.values())
            if width < crop_size or height < crop_size:
                problems.append(f"Source is smaller than native {crop_size} crop")
            rectangles = [[0, 0, crop_size, crop_size]]
            opposite = [width - crop_size, height - crop_size, crop_size, crop_size]
            if opposite != rectangles[0]:
                rectangles.append(opposite)
        else:
            width = height = 0
            rectangles = []
        warnings = []
        for role, item in candidates.items():
            if role != "input" and ("srgb_rendering_intent" in item or item.get("png_gamma", 1) != 1 or "icc_profile_name" in item):
                warnings.append(f"{role}: display color metadata found; raw codes will be retained, tags alone do not establish pixel transfer")
            if role in ("height", "normal", "roughness") and item["sample_bits"] < 16:
                warnings.append(f"{role}: source has only {item['sample_bits']}-bit samples; preserved without fabricated precision")
        discovered.append({"material_id": material_id, "source_directory": str(folder.resolve()),
                           "source_directory_symlink": folder.is_symlink(),
                           "canonical_directory_name": material_id, "maps": candidates,
                           "crop_rectangles_top_left_xywh": rectangles,
                           "common_pixel_dimensions": [width, height], "dimension_policy": dimension_policy,
                           "ignored_files": ignored, "warnings": warnings,
                           "problems": problems, "ready": not problems})
    return discovered


def split_for(material_id: str, fraction: float) -> str:
    rank = int(hashlib.sha256(material_id.encode()).hexdigest()[:16], 16) / (1 << 64)
    return "validation" if rank < fraction else "train"


def describe_encoding(role: str, metadata: dict) -> str:
    if role != "input":
        return "linear_data"
    if metadata.get("png_gamma") == 1 and "srgb_rendering_intent" not in metadata:
        return "linear"
    return "source_srgb_assumed"


def transformed_crop(array: np.ndarray, role: str, source: dict, override: str | dict | None) -> tuple[np.ndarray, list[dict], dict]:
    result = array.copy()
    transforms: list[dict] = []
    output_metadata = {"color_chunks": source.get("color_chunks", [])}
    if override and override != "preserve":
        if role == "input":
            raise ValueError("Transfer overrides apply to numeric target maps only; declare diffuse encoding separately")
        mode = override if isinstance(override, str) else override["mode"]
        maximum = np.iinfo(result.dtype).max
        normalized = result.astype(np.float64) / maximum
        if mode == "srgb_to_linear":
            values = np.where(normalized <= .04045, normalized / 12.92, ((normalized + .055) / 1.055) ** 2.4)
        elif mode == "gamma_to_linear":
            exponent = float(override["exponent"])
            if not .1 <= exponent <= 10:
                raise ValueError("Gamma exponent must be between 0.1 and 10")
            values = normalized ** exponent
        else:
            raise ValueError(f"Unsupported explicit transfer conversion: {mode}")
        result = np.rint(values * maximum).astype(result.dtype)
        if result.shape[2] in (2, 4):
            # Alpha is independent auxiliary data, never a color-transfer value.
            result[..., -1] = array[..., -1]
        transforms.append({"type": mode, "declared_by": "transfer_overrides", "exponent": override.get("exponent") if isinstance(override, dict) else None,
                           "quantization": f"nearest uint{result.dtype.itemsize * 8}", "not_bit_exact_to_source": True})
        output_metadata = {"color_chunks": [{"type": "gAMA", "data_base64": base64.b64encode(struct.pack(">I", 100000)).decode()}]}
    if role == "normal" and source["suffix"] == "nor_dx":
        if result.shape[2] not in (3, 4):
            raise ValueError("Tangent normal must have three RGB components and optional opaque alpha")
        if result.shape[2] == 4 and np.any(result[:, :, 3] != np.iinfo(result.dtype).max):
            raise ValueError("Non-opaque normal alpha requires review before convention conversion")
        result[:, :, 1] = np.iinfo(result.dtype).max - result[:, :, 1]
        transforms.append({"type": "directx_to_opengl", "component": "G", "operation": "max_integer_code - G", "reversible": True})
    return result, transforms, output_metadata


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def contained_path(root: Path, relative: str, label: str) -> Path:
    """Reject absolute/traversal/symlink escapes from a generated manifest."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"{label} must be a relative path contained in its manifest directory")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"{label} escapes its manifest directory")
    return candidate


def prepare_material(material: dict, output: Path, overrides: dict, approve: bool, validation_fraction: float,
                     published_audit: dict | None = None) -> list[dict]:
    if material["maps"].get("height", {}).get("sample_bits", 0) < 16:
        raise ValueError("High-detail displacement requires original 16-bit samples; 8-bit height is excluded")
    sample_root = output / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    identities = material.get("sample_ids") or [f"{material['material_id']}_auto_{number + 1:03d}" for number in range(len(material["crop_rectangles_top_left_xywh"]))]
    desired_split = material.get("explicit_split") or split_for(material["material_id"], validation_fraction)
    if len(identities) != len(material["crop_rectangles_top_left_xywh"]) or desired_split not in ("train", "validation"):
        raise ValueError("Explicit sample identity or split contract is invalid")
    if any(not isinstance(identity, str) or identity != slugify(identity) for identity in identities):
        raise ValueError("Sample identities must be canonical names without path components")
    material_override = overrides.get(material["material_id"], {})
    signatures = []
    for rectangle in material["crop_rectangles_top_left_xywh"]:
        contract = {"generator": GENERATOR, "material": material["material_id"], "crop": rectangle,
                    "sources": {role: item["file_sha256"] for role, item in material["maps"].items()},
                    "overrides": material_override, "dimension_policy": material["dimension_policy"]}
        signatures.append(sha256_bytes(json.dumps(contract, sort_keys=True).encode()))
    existing = []
    for identity, signature in zip(identities, signatures):
        folder = sample_root / identity
        if folder.exists():
            record = json.loads((folder / "sample.json").read_text()) if (folder / "sample.json").exists() else {}
            if record.get("generator") != GENERATOR or record.get("preparation_signature") != signature:
                raise ValueError(f"Existing sample collision, source changed, or preparation options changed: {folder}; no files overwritten")
            if record.get("status") != ("approved" if approve else "prepared") or record.get("split") != desired_split:
                raise ValueError(f"Existing sample approval or split differs from requested options: {folder}; review the existing metadata explicitly, no metadata overwritten")
            errors = verify_sample(folder, check_sources=False)
            if errors:
                raise ValueError(f"Existing crop failed verification: {folder}: {'; '.join(errors)}")
            existing.append(record)
        else:
            existing.append(None)
    if all(existing):
        return existing
    staging = Path(tempfile.mkdtemp(prefix=".material-preparation-", dir=sample_root))
    records = []
    try:
        for number, (identity, rectangle) in enumerate(zip(identities, material["crop_rectangles_top_left_xywh"])):
            folder = staging / identity
            folder.mkdir()
            records.append({"schema_version": 2, "generator": GENERATOR, "sample_id": identity,
                            "material_id": material["material_id"], "source_url": None, "source_origin": "unverified",
                            "source_directory": material["source_directory"], "maps": {}, "extra_maps": {}, "map_metadata": {},
                            "source_pixel_dimensions": material["common_pixel_dimensions"],
                            "sample_pixel_dimensions": rectangle[2:], "crop_rectangle_top_left_xywh": rectangle,
                            "normal_convention": "OpenGL +Y", "height_units": "relative_source_values",
                            "physical_width_meters": None, "status": "approved" if approve else "prepared",
                            "review_status": "user_selected_for_pilot" if approve else "unreviewed",
                            "source_precision_verified": True, "crop_values_verified": True,
                            "split": desired_split,
                            "preparation_signature": signatures[number], "encoding_warnings": material["warnings"],
                            "dimension_policy": material["dimension_policy"], "source_notes": []})
        for role, source in material["maps"].items():
            decoded, metadata = read_png(source["path"])
            if metadata["file_sha256"] != source["file_sha256"]:
                raise ValueError(f"Source changed during preparation: {source['path']}")
            if role == "normal" and decoded.shape[2] not in (3, 4):
                raise ValueError("Normal source must have RGB components")
            if role in ("height", "roughness") and decoded.shape[2] > 1:
                if decoded.shape[2] == 2:
                    tolerance = 8 if decoded.dtype == np.uint16 else 0
                    if np.any(decoded[..., 1] < np.iinfo(decoded.dtype).max - tolerance):
                        raise ValueError(f"{role} source has non-opaque alpha; scalar target needs review")
                elif decoded.shape[2] not in (3, 4) or not np.array_equal(decoded[:, :, 0], decoded[:, :, 1]) or not np.array_equal(decoded[:, :, 0], decoded[:, :, 2]):
                    raise ValueError(f"{role} source has unequal RGB channels; scalar target needs review")
                if decoded.shape[2] == 4 and np.any(decoded[:, :, 3] != np.iinfo(decoded.dtype).max):
                    raise ValueError(f"{role} source has non-opaque alpha; scalar target needs review")
            for number, record in enumerate(records):
                x, y, width, height = record["crop_rectangle_top_left_xywh"]
                original = decoded[y:y + height, x:x + width]
                crop, transforms, color_metadata = transformed_crop(original, role, source, material_override.get(role))
                filename = MAP_NAMES.get(role, slugify(role.removeprefix("extra:")) + ".png")
                target = staging / record["sample_id"] / filename
                write_png(target, crop, color_metadata)
                round_trip, target_header = read_png(target)
                if not np.array_equal(crop, round_trip) or crop.dtype != round_trip.dtype:
                    raise ValueError(f"PNG round trip changed integer pixels: {target}")
                if role in MAP_NAMES:
                    record["maps"][role] = filename
                else:
                    record["extra_maps"][role.removeprefix("extra:")] = filename
                record["map_metadata"][role] = {
                    "filename": filename, "encoding": describe_encoding(role, source),
                    "encoding_assumption": "Raw numeric codes interpreted as linear data; metadata tags do not trigger conversion" if role != "input" else "PNG gamma=1 is linear; other input assumed sRGB pending color-profile review",
                    "source": {**{key: value for key, value in source.items() if key != "color_chunks"},
                               **published_source_provenance(source, material["material_id"], published_audit)},
                    "sample_bits": source["sample_bits"], "channels": crop.shape[2], "sample_dtype": str(crop.dtype),
                    "sample_sha256": target_header["file_sha256"], "decoded_pixel_sha256": pixel_sha256(crop),
                    "source_crop_pixel_sha256": pixel_sha256(original), "transforms": transforms,
                    "exact_source_crop": not transforms, "transformed_source_crop_verified": True,
                    "png_color_metadata": {key: value for key, value in target_header.items() if key not in ("color_chunks", "file_sha256")}}
                if role in ("height", "roughness") and crop.shape[2] == 2:
                    record["map_metadata"][role]["scalar_alpha_policy"] = {
                        "scalar_component": "grayscale", "alpha_preserved_in_png": True,
                        "alpha_used_to_scale_scalar": False, "uint16_near_opaque_tolerance_codes": 8,
                        "meaningful_transparency_requires_review": True}
            del decoded
        source_folder = Path(material["source_directory"])
        for record in records:
            if all(item["source"].get("provider") == "Poly Haven" for item in record["map_metadata"].values()):
                record.update(source_url="https://polyhaven.com/a/" + material["material_id"],
                              source_origin="all_maps_match_published_polyhaven_files",
                              source_provider="Poly Haven", source_license="CC0-1.0",
                              source_license_url="https://polyhaven.com/license")
            elif all(item["source"].get("provider") == "ambientCG" for item in record["map_metadata"].values()):
                first = next(iter(record["map_metadata"].values()))["source"]
                record.update(source_url=first["asset_url"], source_origin="all_maps_match_verified_ambientcg_package_members",
                              source_provider="ambientCG", source_license="CC0-1.0",
                              source_license_url=first["license_url"], source_asset_id=first["asset_id"],
                              source_creation_method=first.get("creation_method"))
        for note in sorted(source_folder.iterdir()):
            if note.is_file() and not note.name.startswith(".") and note.suffix.lower() in (".txt", ".md", ".json", ".yaml", ".yml"):
                data = note.read_bytes()
                for record in records:
                    relative = "source_notes/" + note.name
                    target = staging / record["sample_id"] / relative
                    target.parent.mkdir(exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(data)
                    record["source_notes"].append({"path": relative, "original_path": str(note), "sha256": sha256_bytes(data)})
        for number, record in enumerate(records):
            folder = staging / record["sample_id"]
            write_json(folder / "sample.json", record)
            if existing[number]:
                continue
            destination = sample_root / record["sample_id"]
            if destination.exists():
                raise ValueError(f"Concurrent sample collision: {destination}")
            os.rename(folder, destination)
        return [old or new for old, new in zip(existing, records)]
    finally:
        shutil.rmtree(staging)


def verify_sample(folder: Path, check_sources: bool = False, source_cache: SourceDecodeCache | None = None) -> list[str]:
    errors = []
    try:
        record = json.loads(contained_path(folder, "sample.json", "Sample metadata").read_text())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        return [str(error)]
    if record.get("generator") != GENERATOR or record.get("schema_version") != 2:
        return ["Sample is not a generated version-2 crop"]
    if record.get("sample_id") != folder.name:
        errors.append("Sample identity disagrees with its containing folder")
    if set(MAP_NAMES) - record.get("maps", {}).keys() or set(MAP_NAMES) - record.get("map_metadata", {}).keys():
        return ["Sample is missing required maps or verified map metadata"]
    if record.get("source_precision_verified") is not True or record.get("crop_values_verified") is not True:
        errors.append("Sample does not declare completed source precision and crop verification")
    dimensions = record.get("sample_pixel_dimensions")
    rectangle = record.get("crop_rectangle_top_left_xywh")
    if not isinstance(dimensions, list) or len(dimensions) != 2 or not all(isinstance(value, int) and value > 0 for value in dimensions):
        return errors + ["Invalid sample dimensions"]
    if not isinstance(rectangle, list) or len(rectangle) != 4 or not all(isinstance(value, int) and value >= 0 for value in rectangle) or rectangle[2:] != dimensions:
        return errors + ["Invalid crop rectangle or dimension mismatch"]
    for role, item in record.get("map_metadata", {}).items():
        try:
            manifest_filename = record["maps"].get(role) if role in MAP_NAMES else record.get("extra_maps", {}).get(role.removeprefix("extra:"))
            if manifest_filename != item["filename"]:
                errors.append(f"{role}: map filename disagrees with verified metadata")
                continue
            path = contained_path(folder, item["filename"], f"{role} map")
            decoded, metadata = read_png(path)
            if metadata["file_sha256"] != item["sample_sha256"]:
                errors.append(f"{role}: file checksum changed")
            if pixel_sha256(decoded) != item["decoded_pixel_sha256"]:
                errors.append(f"{role}: decoded integer pixels changed")
            if list(decoded.shape[:2][::-1]) != record["sample_pixel_dimensions"]:
                errors.append(f"{role}: sample dimensions changed")
            if metadata["sample_bits"] != item["sample_bits"] or metadata["channels"] != item["channels"]:
                errors.append(f"{role}: sample precision or channels changed")
            if str(decoded.dtype) != item["sample_dtype"] or metadata["sample_bits"] != item["source"]["sample_bits"]:
                errors.append(f"{role}: stored sample dtype or source precision disagrees")
            if check_sources:
                original, source_metadata = source_cache.read(item["source"]["path"]) if source_cache else read_png(item["source"]["path"])
                if source_metadata["file_sha256"] != item["source"]["file_sha256"]:
                    errors.append(f"{role}: source checksum changed")
                x, y, width, height = record["crop_rectangle_top_left_xywh"]
                source_crop = original[y:y + height, x:x + width]
                if pixel_sha256(source_crop) != item["source_crop_pixel_sha256"]:
                    errors.append(f"{role}: source crop values changed")
                if not item["transforms"] and not np.array_equal(decoded, source_crop):
                    errors.append(f"{role}: crop differs from original source slice")
                if item["transforms"]:
                    transfer = next((transform for transform in item["transforms"] if transform["type"] in ("srgb_to_linear", "gamma_to_linear")), None)
                    override = {"mode": transfer["type"], "exponent": transfer.get("exponent")} if transfer else None
                    expected, _, _ = transformed_crop(source_crop, role, item["source"], override)
                    if not np.array_equal(decoded, expected):
                        errors.append(f"{role}: crop differs from explicitly transformed source slice")
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(f"{role}: {error}")
    for item in record.get("source_notes", []):
        try:
            if file_sha256(contained_path(folder, item["path"], "Source note")) != item["sha256"]:
                errors.append(f"Source note checksum changed: {item['path']}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    return errors


def folder_inventory(folder: Path) -> list[dict]:
    """Full-file hashes including hidden files; reject symlinks/special files."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"Duplicate folder must be a regular directory: {folder}")
    inventory = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Duplicate folder contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Duplicate folder contains a nonregular file: {path}")
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"Duplicate source changed while hashing: {path}")
        relative = path.relative_to(folder)
        inventory.append({"relative_path": relative.as_posix(), "file_bytes": after.st_size,
                          "sha256": digest, "hidden": any(part.startswith(".") for part in relative.parts)})
    return inventory


def identical_visible_files(incoming: list[dict], canonical: list[dict]) -> bool:
    first = [item for item in incoming if not item["hidden"]]
    second = [item for item in canonical if not item["hidden"]]
    return bool(first) and first == second


def normalization_plan(materials: list[dict], archive_identical_duplicates: Path | None = None) -> list[dict]:
    plan = []
    groups: dict[Path, list[Path]] = {}
    for material in materials:
        if material.get("source_directory_symlink"):
            raise ValueError("Normalization refuses symbolic-link source directories")
        source = Path(material["source_directory"])
        target = source.parent / material["canonical_directory_name"]
        groups.setdefault(target, []).append(source)
    archive_root = archive_identical_duplicates.resolve() if archive_identical_duplicates else None
    if archive_root and any(archive_root.is_relative_to(source.parent.resolve()) for sources in groups.values() for source in sources):
        raise ValueError("Identical duplicate archive must be outside the sources directory")
    reserved: set[Path] = set()
    for target, sources in groups.items():
        sources = sorted(set(sources), key=lambda source: (str(source).casefold(), str(source)))
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"Normalization destination is not a regular directory: {target}")
            primary = next((source for source in sources if source == target),
                           next((source for source in sources if source.samefile(target)), target))
        else:
            primary = sources[0]
        incoming_folders = [source for source in sources if source != primary and not source.samefile(primary)]
        if incoming_folders and not archive_root:
            if len(sources) > 1:
                raise ValueError(f"Normalization collision: {target}; optional identical-duplicate archival requires an explicit archive directory")
            raise ValueError(f"Normalization destination already exists: {target}")
        if primary != target:
            rename = {"from": str(primary), "to": str(target)}
            if target.exists() and primary.samefile(target):
                rename["case_only"] = True
            plan.append(rename)
        canonical_inventory = folder_inventory(primary) if incoming_folders else []
        for incoming in incoming_folders:
            incoming_inventory = folder_inventory(incoming)
            if not identical_visible_files(incoming_inventory, canonical_inventory):
                raise ValueError(f"Normalization duplicate differs or contains unique notes: {incoming}; no archive or merge performed")
            digest = sha256_bytes(json.dumps(incoming_inventory, sort_keys=True).encode())
            base = archive_root / (target.name + "__" + slugify(incoming.name) + "__" + digest[:16])
            destination = base
            suffix = 2
            while destination in reserved or destination.exists() or destination.with_name(destination.name + ".archive.json").exists():
                destination = base.with_name(base.name + f"__{suffix:03d}")
                suffix += 1
            reserved.add(destination)
            plan.append({"kind": "archive_identical_duplicate", "from": str(incoming), "to": str(destination),
                         "canonical_directory": str(target), "canonical_directory_before_apply": str(primary),
                         "archive_log_path": str(destination.with_name(destination.name + ".archive.json")),
                         "restoration": {"from": str(destination), "to": str(incoming)},
                         "incoming_files": incoming_inventory, "canonical_files": canonical_inventory,
                         "comparison": "Exact nonhidden relative filename set, full SHA256 and byte count; hidden files preserved in entire folder move",
                         "source_files_deleted": False})
    return plan


def apply_normalization(plan: list[dict]) -> None:
    """Apply preflighted folder renames, restoring completed names on failure."""
    completed = []
    archive_logs = []
    try:
        for item in plan:
            same_case_only = item.get("case_only") and Path(item["to"]).exists() and Path(item["from"]).samefile(item["to"])
            if Path(item["to"]).exists() and not same_case_only:
                raise ValueError(f"Normalization destination appeared during apply: {item['to']}")
            if item.get("kind") == "archive_identical_duplicate":
                incoming_inventory = folder_inventory(Path(item["from"]))
                canonical_inventory = folder_inventory(Path(item["canonical_directory"]))
                if incoming_inventory != item["incoming_files"] or not identical_visible_files(incoming_inventory, canonical_inventory):
                    raise ValueError(f"Identical duplicate changed after preflight: {item['from']}")
                log = Path(item["archive_log_path"])
                log.parent.mkdir(parents=True, exist_ok=True)
                document = {"schema_version": 1, "generator": GENERATOR, "operation": item,
                            "status": "planned", "restoration_instruction": "Move the archived folder back to restoration.to after ensuring that name is free; do not merge or overwrite existing folders"}
                # A sidecar outside the original folder records every full hash.
                # Exclusive creation prevents overwriting a previous archive log.
                with log.open("x") as stream:
                    json.dump(document, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                archive_logs.append((log, document))
            os.rename(item["from"], item["to"])
            completed.append(item)
        for log, document in archive_logs:
            write_json(log, {**document, "status": "archived"})
    except (OSError, ValueError) as error:
        rollback_errors = []
        for item in reversed(completed):
            try:
                same_case_only = item.get("case_only") and Path(item["from"]).exists() and Path(item["from"]).samefile(item["to"])
                if Path(item["from"]).exists() and not same_case_only:
                    raise ValueError(f"Original folder name appeared during rollback: {item['from']}")
                os.rename(item["to"], item["from"])
            except (OSError, ValueError) as rollback_error:
                rollback_errors.append(str(rollback_error))
        suffix = f"; rollback needs attention: {'; '.join(rollback_errors)}" if rollback_errors else "; completed renames rolled back"
        for log, document in archive_logs:
            try:
                write_json(log, {**document, "status": "rollback_needs_attention" if rollback_errors else "rolled_back", "rollback_errors": rollback_errors})
            except OSError as log_error:
                suffix += f"; archive log update failed: {log_error}"
        raise ValueError(f"Normalization failed: {error}{suffix}") from error


REGION_SPLIT_STRATEGY = "heldout-region-v1"
REGION_VALIDATION_SCOPE = "unseen_regions_of_known_materials"


def rectangles_overlap(first: list[int], second: list[int]) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def heldout_regions(width: int, height: int, crop_size: int) -> list[dict]:
    """Choose two training corners plus a disjoint validation region if possible.

    Try center, bottom-left and top-right in that order. If both training
    corners cover all available space, keep top-left for training and use the
    opposite corner for validation. Every returned crop is native and opaque;
    touching boundaries share no pixels. Small sources cannot be split safely.
    """
    if min(width, height) < crop_size:
        raise ValueError("Source is smaller than the requested native crop")
    first = [0, 0, crop_size, crop_size]
    opposite = [width - crop_size, height - crop_size, crop_size, crop_size]
    if rectangles_overlap(first, opposite):
        raise ValueError("Source has no disjoint native training and validation regions at this crop size")
    candidates = [("center", [(width - crop_size) // 2, (height - crop_size) // 2, crop_size, crop_size]),
                  ("bottom_left", [0, height - crop_size, crop_size, crop_size]),
                  ("top_right", [width - crop_size, 0, crop_size, crop_size])]
    for label, candidate in candidates:
        if not rectangles_overlap(candidate, first) and not rectangles_overlap(candidate, opposite):
            return [{"ordinal": 1, "split": "train", "region": "top_left", "rectangle": first},
                    {"ordinal": 2, "split": "train", "region": "opposite_corner", "rectangle": opposite},
                    {"ordinal": 3, "split": "validation", "region": label, "rectangle": candidate}]
    return [{"ordinal": 1, "split": "train", "region": "top_left", "rectangle": first},
            {"ordinal": 2, "split": "validation", "region": "opposite_corner", "rectangle": opposite}]


def prepare_region_splits(dataset: Path, sources: Path, crop_size: int, migrate: bool = False,
                          published_audit: dict | None = None, report_path: Path | None = None,
                          material_ids: list[str] | None = None) -> dict:
    """Explicit, rollback-capable metadata migration with source/crop verification."""
    dataset = dataset.resolve()
    sources = sources.resolve()
    if dataset.is_relative_to(sources):
        raise ValueError("Region dataset output must be outside the sources directory")
    index_path = dataset / "dataset.json"
    old_index_bytes = index_path.read_bytes() if index_path.exists() else None
    old_index = json.loads(old_index_bytes) if old_index_bytes else None
    if old_index and (old_index.get("generator") != GENERATOR or old_index.get("schema_version") != 2):
        raise ValueError("Region split migration refuses a foreign dataset index")
    if old_index and old_index.get("crop_size") != crop_size:
        raise ValueError("Region split crop size differs from existing dataset; use a separate output directory")
    if old_index and old_index.get("split_strategy") != REGION_SPLIT_STRATEGY and not migrate:
        raise ValueError("Existing dataset uses heldout materials; add --migrate-split-policy to preserve prior metadata and switch explicitly")
    materials = discover_materials(sources, crop_size)
    if material_ids is not None:
        requested = set(material_ids)
        available = {material["material_id"] for material in materials}
        if not requested or requested - available:
            raise ValueError(f"Unknown or empty material selection: {sorted(requested - available)}")
        if old_index and {entry["material_id"] for entry in old_index["samples"]} - requested:
            raise ValueError("Material selection would omit existing indexed samples; use a separate dataset")
        materials = [material for material in materials if material["material_id"] in requested]
    dataset.mkdir(parents=True, exist_ok=True)
    source_states = {item["path"]: SourceDecodeCache.key(Path(item["path"]))
                     for material in materials for item in material["maps"].values()}
    by_material = {material["material_id"]: material for material in materials}
    if len(by_material) != len(materials):
        raise ValueError("Region preparation refuses duplicate material identities")
    existing_records = {}
    old_metadata_bytes = {}
    cache = SourceDecodeCache()
    for entry in old_index.get("samples", []) if old_index else []:
        folder = contained_path(dataset, entry["path"], "Existing region sample")
        print(f"Verifying source regions for {entry['sample_id']}...", flush=True)
        errors = verify_sample(folder, True, cache)
        if errors:
            raise ValueError(f"Cannot migrate unverified sample {entry['sample_id']}: {'; '.join(errors)}")
        metadata_path = contained_path(folder, "sample.json", "Existing sample metadata")
        before = metadata_path.read_bytes()
        record = json.loads(before)
        for field in ("sample_id", "material_id", "split", "status"):
            if record.get(field) != entry.get(field):
                raise ValueError(f"Dataset index disagrees with sample metadata: {entry['sample_id']} {field}")
        material = by_material.get(record["material_id"])
        if not material or not material["ready"]:
            raise ValueError(f"Existing material source is absent or incomplete: {record['material_id']}")
        for role, item in record["map_metadata"].items():
            if role not in material["maps"] or item["source"]["file_sha256"] != material["maps"][role]["file_sha256"]:
                raise ValueError(f"Current source contract differs from existing crop: {record['sample_id']} {role}")
        if record["sample_id"] in existing_records:
            raise ValueError("Duplicate sample identity in dataset index")
        existing_records[record["sample_id"]] = record
        old_metadata_bytes[metadata_path] = before
    staging = Path(tempfile.mkdtemp(prefix=".region-preparation-", dir=dataset))
    records = []
    skipped = []
    moves = []
    backup = None
    modified_paths = []
    published_moves = []
    index_publication_started = False
    try:
        for material in materials:
            if not material["ready"]:
                skipped.append({"material_id": material["material_id"], "problems": material["problems"]})
                continue
            width, height = material["common_pixel_dimensions"]
            try:
                regions = heldout_regions(width, height, crop_size)
            except ValueError as error:
                skipped.append({"material_id": material["material_id"], "problems": [str(error)]})
                continue
            statuses = {item["status"] for item in existing_records.values() if item["material_id"] == material["material_id"]}
            if len(statuses) > 1:
                raise ValueError("Existing material samples have inconsistent approval; review them before migration")
            approve = statuses == {"approved"}
            print(f"Preparing disjoint native regions for {material['material_id']}...", flush=True)
            # Decode each role once for all missing regions. Previously each
            # individual crop decoded the same full-resolution parents again.
            missing = []
            for region in regions:
                identity = f"{material['material_id']}_auto_{region['ordinal']:03d}"
                existing = existing_records.get(identity)
                if existing:
                    if existing["crop_rectangle_top_left_xywh"] != region["rectangle"]:
                        raise ValueError(f"Existing native crop rectangle differs from region policy: {identity}")
                else:
                    destination = dataset / "samples" / identity
                    if destination.exists():
                        raise ValueError(f"Unindexed sample collision: {destination}; no files overwritten")
                    missing.append((identity, region))
            created = {}
            if missing:
                request = {**material, "sample_ids": [identity for identity, _region in missing],
                           "explicit_split": "train",
                           "crop_rectangles_top_left_xywh": [region["rectangle"] for _identity, region in missing]}
                created = {record["sample_id"]:record for record in prepare_material(request, staging, {}, approve, .2, published_audit)}
            for region in regions:
                identity = f"{material['material_id']}_auto_{region['ordinal']:03d}"
                existing = existing_records.get(identity)
                if existing:
                    record = {**existing}
                else:
                    destination = dataset / "samples" / identity
                    record = created[identity]
                    moves.append((staging / "samples" / identity, destination))
                record.update(split=region["split"], split_strategy=REGION_SPLIT_STRATEGY,
                              validation_scope=REGION_VALIDATION_SCOPE, source_region_role=region["split"],
                              source_region_name=region["region"])
                records.append(record)
        if not records:
            raise ValueError("No source can provide disjoint native training and validation regions")
        included_ids = {record["sample_id"] for record in records}
        if set(existing_records) - included_ids:
            raise ValueError("Region policy would omit existing indexed samples; use a separate dataset")
        # Backup exact prior metadata before changing any published file. The
        # sources and all existing crop PNGs remain byte-for-byte untouched.
        changed = any(json.loads(raw) != next(record for record in records if record["sample_id"] == path.parent.name)
                      for path, raw in old_metadata_bytes.items())
        if old_index_bytes and (changed or moves or old_index.get("split_strategy") != REGION_SPLIT_STRATEGY):
            backup_parent = dataset / "metadata-backups"
            backup_parent.mkdir(exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("region-splits-%Y%m%dT%H%M%SZ-"), dir=backup_parent))
            (backup / "dataset.json").write_bytes(old_index_bytes)
            backup_records = [{"path": "dataset.json", "sha256": sha256_bytes(old_index_bytes)}]
            for path, raw in old_metadata_bytes.items():
                relative = path.relative_to(dataset)
                target = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                backup_records.append({"path": relative.as_posix(), "sha256": sha256_bytes(raw)})
            write_json(backup / "backup.json", {"generator": GENERATOR, "original_dataset": str(dataset),
                                               "status": "preserved_before_region_split_migration", "files": backup_records,
                                               "source_images_modified": False, "existing_crop_pngs_modified": False})
        if old_index_bytes is not None and index_path.read_bytes() != old_index_bytes:
            raise ValueError("Dataset index changed during region preparation; refusing concurrent update")
        for source_path, state in source_states.items():
            if SourceDecodeCache.key(Path(source_path)) != state:
                raise ValueError(f"Source changed during region preparation: {source_path}; refusing publication")
        for path, raw in old_metadata_bytes.items():
            if path.read_bytes() != raw:
                raise ValueError(f"Sample metadata changed during region preparation: {path}")
        for source, destination in moves:
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise ValueError(f"Concurrent region sample collision: {destination}")
            os.rename(source, destination)
            published_moves.append(destination)
        for record in records:
            path = dataset / "samples" / record["sample_id"] / "sample.json"
            if path in old_metadata_bytes:
                if json.loads(old_metadata_bytes[path]) == record:
                    continue
                modified_paths.append(path)
            write_json(path, record)
        index = {**(old_index or {}), "schema_version": 2, "generator": GENERATOR, "crop_size": crop_size,
                 "split_strategy": REGION_SPLIT_STRATEGY, "validation_scope": REGION_VALIDATION_SCOPE,
                 "split_policy": "Every material contributes training and validation; validation is a native disjoint source region, not unseen material identity",
                 "region_policy": "Two opposite training corners plus center/bottom-left/top-right validation when disjoint; otherwise top-left training and opposite validation",
                 "height_normalization": "source integer code / maximum code; no per-crop min/max stretch",
                 "normal_convention": "OpenGL +Y", "source_images_modified": False,
                 "existing_crop_pngs_modified": False, "original_sources_required_for_training": False,
                 "source_deletion_authorized": False, "skipped_materials": skipped,
                 "samples": [{"sample_id": record["sample_id"], "material_id": record["material_id"],
                              "path": "samples/" + record["sample_id"], "split": record["split"], "status": record["status"]} for record in records]}
        index.pop("validation_fraction", None)
        index_publication_started = True
        write_json(index_path, index)
        report = {"schema_version": 2, "split_strategy": REGION_SPLIT_STRATEGY,
                  "validation_scope": REGION_VALIDATION_SCOPE, "sample_count": len(records),
                  "train_samples": sum(record["split"] == "train" for record in records),
                  "validation_samples": sum(record["split"] == "validation" for record in records),
                  "material_count": len({record["material_id"] for record in records}),
                  "metadata_backup": str(backup) if backup else None,
                  "new_sample_count": len(moves), "skipped_materials": skipped,
                  "source_images_modified": False, "existing_crop_pngs_modified": False,
                  "source_snapshot_material_count": len(materials),
                  "requested_material_ids": sorted(set(material_ids)) if material_ids is not None else None,
                  "source_verification_cache": cache.report()}
        if report_path:
            write_json(report_path, report)
        return report
    except (ValueError, OSError, KeyError, TypeError):
        for path in reversed(modified_paths):
            path.write_bytes(old_metadata_bytes[path])
        if index_publication_started:
            if old_index_bytes is not None:
                index_path.write_bytes(old_index_bytes)
            elif index_path.exists():
                index_path.unlink()
        for folder in reversed(published_moves):
            shutil.rmtree(folder)
        raise
    finally:
        shutil.rmtree(staging)


def load_source_audits(published_path: Path | None, package_paths: list[Path]) -> dict | None:
    """Read separate provider proofs; their exact hashes are checked per source."""
    audit = json.loads(published_path.read_text()) if published_path else {}
    if package_paths:
        audit = {**audit, "package_audits": [json.loads(path.read_text()) for path in package_paths]}
    return audit or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare", "normalize-names"):
        child = sub.add_parser(command)
        child.add_argument("--sources", type=Path, required=True)
        child.add_argument("--crop-size", type=int, default=1024)
        child.add_argument("--report", type=Path)
        child.add_argument("--dimension-policy", choices=("strict", "common-intersection"), default="strict")
        child.add_argument("--acknowledge-registration", action="store_true")
        if command == "prepare":
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--approve", action="store_true", help="Record explicit user selection of all generated samples for pilot training")
            child.add_argument("--validation-fraction", type=float, default=.2)
            child.add_argument("--transfer-overrides", type=Path, help="JSON {material_id:{height/normal/roughness:'srgb_to_linear' or {mode:'gamma_to_linear',exponent:2.2}}}; never inferred")
            child.add_argument("--published-source-audit", type=Path, help="Optional official Poly Haven API audit; annotate licenses only for current files matching published MD5 and size")
            child.add_argument("--package-source-audit", type=Path, action="append", default=[], help="Verified ambientCG archive/member audit; repeat per asset, preserving original filenames")
        if command == "normalize-names":
            child.add_argument("--apply", action="store_true", help="Rename source folders only; image filenames and bytes stay unchanged")
            child.add_argument("--archive-identical-duplicates", type=Path, help="Explicitly archive byte-identical duplicate material folders outside sources; preserve all original files and a restoration log")
    child = sub.add_parser("verify")
    child.add_argument("--dataset", type=Path, required=True)
    child.add_argument("--check-sources", action="store_true")
    child.add_argument("--report", type=Path)
    child.add_argument("--source-cache-mib", type=int, help="Retained source decode cache for --check-sources; default uses detected memory headroom, 0 disables retention, positive values set a smaller upper bound")
    child = sub.add_parser("prepare-region-splits", help="Prepare disjoint validation regions from every material; explicitly migrate old heldout-material metadata")
    child.add_argument("--dataset", type=Path, required=True)
    child.add_argument("--sources", type=Path, required=True)
    child.add_argument("--crop-size", type=int, default=1024)
    child.add_argument("--migrate-split-policy", action="store_true")
    child.add_argument("--material", action="append", dest="material_ids", help="Select a material ID; repeat to prepare a bounded subset in a separate dataset")
    child.add_argument("--published-source-audit", type=Path)
    child.add_argument("--package-source-audit", type=Path, action="append", default=[])
    child.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare-region-splits":
            audit = load_source_audits(arguments.published_source_audit, arguments.package_source_audit)
            report = prepare_region_splits(arguments.dataset, arguments.sources, arguments.crop_size,
                                           arguments.migrate_split_policy, audit, arguments.report,
                                           arguments.material_ids)
            print(json.dumps(report, indent=2))
            return 0
        if arguments.command == "verify":
            index = json.loads((arguments.dataset / "dataset.json").read_text())
            if index.get("generator") != GENERATOR or index.get("schema_version") != 2 or not isinstance(index.get("samples"), list):
                raise ValueError("Dataset index is not a generated version-2 material dataset")
            checks = []
            source_cache = SourceDecodeCache(None if arguments.source_cache_mib is None else arguments.source_cache_mib * 1024 * 1024) if arguments.check_sources else None
            for item in index["samples"]:
                sample_path = contained_path(arguments.dataset, item["path"], "Dataset sample path")
                errors = verify_sample(sample_path, arguments.check_sources, source_cache)
                sample_record = json.loads(contained_path(sample_path, "sample.json", "Sample metadata").read_text())
                for field in ("sample_id", "material_id", "split", "status"):
                    if item.get(field) != sample_record.get(field):
                        errors.append(f"Dataset index {field} disagrees with sample metadata")
                checks.append({"sample_id": item["sample_id"], "verified": not errors, "errors": errors})
            report = {"schema_version": 2, "checks": checks, "verified": all(item["verified"] for item in checks),
                      "source_comparison": arguments.check_sources, "source_cache": source_cache.report() if source_cache else None}
            if arguments.report:
                write_json(arguments.report, report)
            print(json.dumps({"verified": report["verified"], "sample_count": len(checks), "failed": [item for item in checks if not item["verified"]]}, indent=2))
            return 0 if report["verified"] else 1
        if arguments.crop_size <= 0:
            raise ValueError("Crop size must be positive")
        if arguments.dimension_policy == "common-intersection" and not arguments.acknowledge_registration:
            raise ValueError("Common-intersection cropping needs --acknowledge-registration after checking paired UV alignment")
        if arguments.command == "prepare" and arguments.output.resolve().is_relative_to(arguments.sources.resolve()):
            raise ValueError("Output must be outside the sources directory; source folders cannot contain generated crops")
        materials = discover_materials(arguments.sources.resolve(), arguments.crop_size, arguments.dimension_policy)
        report = {"schema_version": 2, "generator": GENERATOR, "source_images_modified": False,
                  "crop_size": arguments.crop_size, "crop_policy": "top-left and opposite corner; applies to squares and rectangles",
                  "materials": materials}
        if arguments.command == "normalize-names":
            report["renames"] = normalization_plan(materials, arguments.archive_identical_duplicates)
            if arguments.apply:
                apply_normalization(report["renames"])
                report["applied"] = True
            else:
                report["applied"] = False
        elif arguments.command == "prepare":
            if not 0 < arguments.validation_fraction < 1:
                raise ValueError("Validation fraction must be between zero and one")
            overrides = json.loads(arguments.transfer_overrides.read_text()) if arguments.transfer_overrides else {}
            published_audit = load_source_audits(arguments.published_source_audit, arguments.package_source_audit)
            previous_index = arguments.output / "dataset.json"
            if previous_index.exists():
                previous_document = json.loads(previous_index.read_text())
                if previous_document.get("generator") != GENERATOR:
                    raise ValueError(f"Existing dataset index belongs to another workflow: {previous_index}; no metadata overwritten")
                requested_status = "approved" if arguments.approve else "prepared"
                if any(item.get("status") != requested_status for item in previous_document.get("samples", [])) or previous_document.get("validation_fraction") != arguments.validation_fraction:
                    raise ValueError("Existing dataset approval or split options differ; review metadata explicitly before changing them. Existing dataset index preserved")
            records = []
            failures = []
            preparation_errors = []
            for material in materials:
                if not material["ready"]:
                    failures.append({"material_id": material["material_id"], "problems": material["problems"]})
                    continue
                print(f"Preparing {material['material_id']}...", flush=True)
                try:
                    records.extend(prepare_material(material, arguments.output.resolve(), overrides, arguments.approve, arguments.validation_fraction, published_audit))
                except (ValueError, OSError, KeyError) as error:
                    failures.append({"material_id": material["material_id"], "problems": [str(error)]})
                    preparation_errors.append(str(error))
            if preparation_errors and previous_index.exists():
                if arguments.report:
                    report.update(sample_count=len(records), skipped_materials=failures, dataset_index_preserved=True)
                    write_json(arguments.report, report)
                raise ValueError(f"Preparation failed for {len(preparation_errors)} materials; existing dataset index preserved: {preparation_errors[0]}")
            index = {"schema_version": 2, "generator": GENERATOR, "crop_size": arguments.crop_size,
                     "crop_policy": report["crop_policy"], "source_images_modified": False,
                     "height_normalization": "source integer code / maximum code; no per-crop min/max stretch",
                     "normal_convention": "OpenGL +Y", "split_policy": "sha256 material identity; all crops of a material share a split",
                     "validation_fraction": arguments.validation_fraction,
                     "samples": [{"sample_id": item["sample_id"], "material_id": item["material_id"], "path": "samples/" + item["sample_id"],
                                  "split": item["split"], "status": item["status"]} for item in records],
                     "skipped_materials": failures, "original_sources_required_for_training": False,
                     "source_deletion_authorized": False,
                     "preservation_scope": "Generated crops, all discovered paired PNG maps and source text notes are self-contained. Unselected source pixels and ignored files are not retained."}
            write_json(arguments.output / "dataset.json", index)
            report.update(sample_count=len(records), skipped_materials=failures, dataset=str(arguments.output.resolve()))
        if arguments.report:
            write_json(arguments.report, report)
        print(json.dumps({"materials": len(materials), "ready": sum(item["ready"] for item in materials),
                          "sample_count": report.get("sample_count"), "problems": [{"material_id": item["material_id"], "problems": item["problems"]} for item in materials if item["problems"]],
                          "renames": report.get("renames"), "skipped_materials": report.get("skipped_materials")}, indent=2))
        return 1 if arguments.command == "prepare" and preparation_errors else 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"material_dataset: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
