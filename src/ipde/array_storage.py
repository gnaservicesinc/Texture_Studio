"""Lossless dataset storage; compression never changes array values or dtype."""
from pathlib import Path
import numpy as np
from .formats import sha256_array, sha256_file


def read_array(path: Path | str, mmap_mode: str | None = "r") -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source, mmap_mode=mmap_mode, allow_pickle=False)
    if source.suffix.lower() != ".npz":
        raise ValueError("Dataset arrays must use NPY or losslessly compressed NPZ")
    with np.load(source, allow_pickle=False) as archive:
        if archive.files != ["data"]:
            raise ValueError("A dataset NPZ must contain exactly one array named data")
        return archive["data"]


def write_array(path: Path | str, value: np.ndarray, *, compressed: bool = True) -> Path:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object/pickle arrays are not supported")
    destination = Path(path).with_suffix(".npz" if compressed else ".npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def array_record(root: Path, path: Path, value: np.ndarray, *, compressed: bool = True) -> dict:
    destination = write_array(path, value, compressed=compressed)
    return {"path": destination.relative_to(root).as_posix(), "shape": list(value.shape), "dtype": value.dtype.str,
            "array_sha256": sha256_array(value), "file_sha256": sha256_file(destination)}
