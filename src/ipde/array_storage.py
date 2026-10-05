"""Lossless scientific dataset storage with mandatory bit-for-bit verification."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

from .formats import (FormatError, PNG_SIGNATURE, read_png_exact, sha256_array,
                      sha256_file, write_exr, write_png)


ARRAY_SUFFIXES = frozenset({".npy", ".npz", ".exr", ".png"})
_SHAPE_ATTRIBUTE = "ipdeArrayShape"
_DTYPE_ATTRIBUTE = "ipdeArrayDtype"


def _image_shape(value: np.ndarray) -> bool:
    return (value.ndim in (2, 3) and value.shape[0] > 0 and value.shape[1] > 0
            and (value.ndim == 2 or 1 <= value.shape[2] <= 4))


def _restore_image_shape(value: np.ndarray, attributes: dict) -> np.ndarray:
    """Metadata may restore a singleton channel, but never reinterpret pixels."""
    encoded_shape = attributes.get(_SHAPE_ATTRIBUTE)
    if encoded_shape is not None:
        try:
            shape = json.loads(encoded_shape)
        except (TypeError, ValueError) as exc:
            raise FormatError("Invalid dataset image shape metadata") from exc
        if (not isinstance(shape, list) or len(shape) not in (2, 3)
                or any(type(length) is not int or length < 1 for length in shape)
                or (len(shape) == 3 and not 1 <= shape[2] <= 4)):
            raise FormatError("Invalid dataset image shape metadata")
        if tuple(shape) != value.shape:
            if len(shape) == 3 and shape[-1] == 1 and tuple(shape[:2]) == value.shape:
                value = value[:, :, None]
            else:
                raise FormatError("Dataset image shape metadata disagrees with its pixels")
    encoded_dtype = attributes.get(_DTYPE_ATTRIBUTE)
    if encoded_dtype is not None and encoded_dtype != value.dtype.str:
        raise FormatError("Dataset image dtype metadata disagrees with its pixels")
    return value


def _png_attributes(path: Path) -> dict[str, str]:
    # read_png_exact already verifies the CRCs, lengths and image encoding.
    # Read just our two uncompressed iTXt attributes, without a display library.
    attributes: dict[str, str] = {}
    with path.open("rb") as stream:
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise FormatError("Invalid dataset PNG signature")
        while True:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise FormatError("Truncated dataset PNG header")
            length, kind = struct.unpack(">I4s", chunk)
            if kind == b"iTXt":
                payload = stream.read(length)
                key, separator, body = payload.partition(b"\0")
                if key.decode("latin-1") in {_SHAPE_ATTRIBUTE, _DTYPE_ATTRIBUTE}:
                    if not separator or not body.startswith(b"\0\0\0\0"):
                        raise FormatError("Unsupported dataset PNG metadata encoding")
                    name = key.decode("latin-1")
                    if name in attributes:
                        raise FormatError("Duplicate dataset PNG array metadata")
                    try:
                        attributes[name] = body[4:].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise FormatError("Invalid dataset PNG array metadata") from exc
            else:
                stream.seek(length, 1)
            stream.seek(4, 1)  # CRC, validated by read_png_exact.
            if kind == b"IEND":
                return attributes


def _read_exr(path: Path) -> np.ndarray:
    try:
        import OpenEXR
    except ImportError as exc:
        raise FormatError("Dataset EXR arrays require OpenEXR>=3.4.4") from exc
    with OpenEXR.File(str(path), separate_channels=True) as source:
        if len(source.parts) != 1:
            raise FormatError("Dataset EXR must contain exactly one image part")
        header = source.header()
        if header.get("type") != OpenEXR.scanlineimage or header.get("compression") != OpenEXR.ZIP_COMPRESSION:
            raise FormatError("Dataset EXR must use lossless ZIP-compressed scanlines")
        channels = source.channels()
        names = set(channels)
        layouts = {frozenset({"Y"}): ("Y",), frozenset({"Y", "A"}): ("Y", "A"),
                   frozenset({"R", "G", "B"}): ("R", "G", "B"),
                   frozenset({"R", "G", "B", "A"}): ("R", "G", "B", "A")}
        order = layouts.get(frozenset(names))
        if order is None or any(channel.xSampling != 1 or channel.ySampling != 1 for channel in channels.values()):
            raise FormatError("Unsupported dataset EXR channel layout")
        planes = [channels[name].pixels for name in order]
        if (any(plane.dtype not in (np.dtype("float16"), np.dtype("float32")) for plane in planes)
                or any(plane.shape != planes[0].shape or plane.dtype != planes[0].dtype for plane in planes)):
            raise FormatError("Dataset EXR channels must have matching float16/float32 samples")
        value = planes[0] if len(planes) == 1 else np.stack(planes, axis=2)
        return _restore_image_shape(value, header)


def read_array(path: Path | str, mmap_mode: str | None = "r") -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.load(source, mmap_mode=mmap_mode, allow_pickle=False)
    if suffix == ".exr":
        return _read_exr(source)
    if suffix == ".png":
        return _restore_image_shape(read_png_exact(source), _png_attributes(source))
    if suffix != ".npz":
        raise ValueError("Dataset arrays must use NPY, lossless NPZ, ZIP EXR or PNG")
    # Own the stream explicitly so corrupt ZIP headers cannot leave NumPy's
    # internally opened descriptor alive when concurrent verification fails.
    with source.open("rb") as stream:
        with np.load(stream, allow_pickle=False) as archive:
            if archive.files != ["data"]:
                raise ValueError("A dataset NPZ must contain exactly one array named data")
            return archive["data"]


def write_array(path: Path | str, value: np.ndarray, *, compressed: bool = True,
                storage: str = "numpy") -> Path:
    """Use images only when their native encoding preserves every sample bit.

    Float16 stays float16; no EXR half conversion, normalization, resampling or
    transfer function is applied. Unsupported dtype/layouts stay lossless NPZ.
    Disabling compression always writes an NPY array with its original dtype.
    """
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object/pickle arrays are not supported")
    if storage not in {"numpy", "images"}:
        raise ValueError("Dataset array storage must be numpy or images")
    suffix = ".npz" if compressed else ".npy"
    if compressed and storage == "images" and _image_shape(array):
        if array.dtype in (np.dtype("float16"), np.dtype("float32")):
            suffix = ".exr"
        elif array.dtype in (np.dtype("uint8"), np.dtype("uint16")):
            suffix = ".png"
    destination = Path(path).with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    attributes = {_SHAPE_ATTRIBUTE: json.dumps(list(array.shape), separators=(",", ":")),
                  _DTYPE_ATTRIBUTE: array.dtype.str}
    if suffix == ".exr":
        write_exr(destination, array, attributes=attributes,
                  storage_description="Dataset array: original floating samples and bit patterns; no gamma, normalization or resampling")
    elif suffix == ".png":
        write_png(destination, array, attributes=attributes)
    else:
        # File objects prevent NumPy from silently appending another suffix.
        with destination.open("wb") as stream:
            if compressed:
                np.savez_compressed(stream, data=array)
            else:
                np.save(stream, array, allow_pickle=False)
    decoded = read_array(destination)
    if decoded.dtype != array.dtype or decoded.shape != array.shape or sha256_array(decoded) != sha256_array(array):
        raise ValueError(f"Dataset array failed an exact round-trip: {destination}")
    return destination


def array_record(root: Path, path: Path, value: np.ndarray, *, compressed: bool = True,
                 storage: str = "numpy") -> dict:
    array = np.asarray(value)
    destination = write_array(path, array, compressed=compressed, storage=storage)
    return {"path": destination.relative_to(root).as_posix(), "shape": list(array.shape), "dtype": array.dtype.str,
            "array_sha256": sha256_array(array), "file_sha256": sha256_file(destination)}
