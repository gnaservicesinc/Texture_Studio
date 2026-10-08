#!/usr/bin/env python3
"""Bounded CPU DeepBump comparison; raw Float32 outputs, no source edits.

DeepBump is GPL-3.0. Its stock CLI quantizes output; this wrapper instead uses
its pinned tiling functions and ONNX network. It does not train DeepBump.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from material_dataset import read_png

REVISION = "fad19ba87daed12b1d0410a57e74f3d79e82f78d"
FILES = {
    "deepbump256.onnx": (26706979, "01158e2ca4800c3365b2b5b539f7118cc9c5da60"),
    "utils_inference.py": (6415, "d02abf37d0d58e80483b7fa75a32da2605c2e513"),
    "module_color_to_normals.py": (1864, "9d73d68803a158ed0695a1549ec547098449eb15"),
    "module_normals_to_height.py": (2856, "62cbc5b6f49aaa16c1724f701970009c45be6657"),
    "LICENSE": (35149, "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"),
    "readme.md": (2119, "8708330ec320500f6c5672b66d0dbc905a749677"),
}
DEFAULT_MATERIALS = ("white_stucco_02", "farm_soil", "cotton_jersey", "broken_brick_wall", "wicker013", "wood_planks")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_identity(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def model_files(directory: Path, download: bool) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (size, identity) in FILES.items():
        path = directory / name
        url = f"https://raw.githubusercontent.com/HugoTini/DeepBump/{REVISION}/{name}"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; use --download to obtain the pinned GPL-3.0 baseline")
            temporary = path.with_suffix(path.suffix + ".download")
            if temporary.exists():
                raise FileExistsError(f"Existing incomplete download needs review: {temporary}")
            try:
                subprocess.run(["curl", "-fLsS", "--max-time", "60", "-A", "TextureStudioResearch", url, "-o", str(temporary)], check=True)
                payload = temporary.read_bytes()
                if len(payload) != size or git_blob_identity(payload) != identity:
                    raise ValueError(f"Downloaded pinned file verification failed: {name}")
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        data = path.read_bytes()
        if len(data) != size or git_blob_identity(data) != identity:
            raise ValueError(f"Pinned model/source identity mismatch: {path}")
        records.append({"filename": name, "bytes": size, "git_blob_sha1": identity, "sha256": hashlib.sha256(data).hexdigest(), "url": url})
    return records


def checked_relative(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"Dataset path escapes its directory: {name}")
    return path


def select_samples(dataset: Path, materials: list[str]) -> tuple[dict, list[tuple[Path, dict]]]:
    index = json.loads((dataset / "dataset.json").read_text())
    if index.get("split_strategy") != "heldout-region-v1" or index.get("validation_scope") != "unseen_regions_of_known_materials":
        raise ValueError("This comparison requires the explicitly prepared held-out region dataset")
    if len(materials) != len(set(materials)):
        raise ValueError("Duplicate material selection")
    selected = []
    for material in materials:
        matches = [row for row in index["samples"] if row["material_id"] == material and row["split"] == "validation" and row.get("status") not in ("excluded", "rejected")]
        if len(matches) != 1:
            raise ValueError(f"Need exactly one validation region for {material}, got {len(matches)}")
        row = matches[0]
        directory = checked_relative(dataset, row["path"])
        metadata = json.loads((directory / "sample.json").read_text())
        if any(metadata.get(key) != row.get(key) for key in ("sample_id", "material_id", "split", "status")):
            raise ValueError(f"Index and sample metadata differ: {directory}")
        if not metadata.get("crop_values_verified") or not metadata.get("source_precision_verified"):
            raise ValueError(f"Unverified map precision: {directory}")
        if metadata.get("normal_convention") != "OpenGL +Y":
            raise ValueError(f"Reference normal convention must be declared OpenGL +Y: {directory}")
        selected.append((directory, metadata))
    return index, selected


def display_rgb(codes: np.ndarray, encoding: str) -> tuple[np.ndarray, np.ndarray]:
    if codes.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or codes.ndim != 3 or codes.shape[-1] not in (3, 4):
        raise ValueError("Diffuse input must retain native UInt8/UInt16 RGB(A) codes")
    maximum = np.iinfo(codes.dtype).max
    rgb = codes[..., :3].astype(np.float32) / np.float32(maximum)
    if encoding in ("linear", "linear_rgb", "linear_color", "linear_light"):
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    elif encoding not in ("srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"):
        raise ValueError(f"Diffuse transfer must be explicit: {encoding!r}")
    valid = np.ones(codes.shape[:2], dtype=bool)
    if codes.shape[-1] == 4:
        tolerance = 8 if codes.dtype == np.uint16 else 0
        raw_valid = codes[..., 3] >= maximum - tolerance
        if not raw_valid.all():
            import cv2
            valid = cv2.erode(raw_valid.astype(np.uint8), np.ones((17, 17), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
            if valid.mean() < 0.9:
                raise ValueError("More than 10% of input excluded by alpha/context mask")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32), valid


def normal_codes(codes: np.ndarray) -> np.ndarray:
    if codes.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or codes.ndim != 3 or codes.shape[-1] not in (3, 4):
        raise ValueError("Normal reference must be native UInt8/UInt16 RGB(A)")
    if codes.shape[-1] == 4 and np.any(codes[..., 3] != np.iinfo(codes.dtype).max):
        raise ValueError("Transparent reference normals require review")
    return codes[..., :3].astype(np.float32) / np.float32(np.iinfo(codes.dtype).max)


def scalar_height(codes: np.ndarray) -> np.ndarray:
    if codes.dtype != np.uint16 or codes.ndim not in (2, 3):
        raise ValueError("Reference height must retain native UInt16 scalar/identical RGB codes")
    if codes.ndim == 3:
        if codes.shape[-1] not in (1, 3, 4) or (codes.shape[-1] >= 3 and np.any(codes[..., :3] != codes[..., :1])):
            raise ValueError("Reference height is not scalar/identical RGB")
        if codes.shape[-1] == 4 and np.any(codes[..., 3] != 65535):
            raise ValueError("Transparent reference height requires review")
        codes = codes[..., 0]
    return codes.astype(np.float32) / np.float32(65535)


def validate_arrays(codes: dict[str, np.ndarray], metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dimensions = metadata.get("sample_pixel_dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 2 or any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("Sample needs explicit positive native pixel dimensions")
    for role in ("input", "normal", "height"):
        if codes[role].ndim not in (2, 3) or list(codes[role].shape[:2][::-1]) != dimensions:
            raise ValueError(f"Decoded {role} dimensions differ from native sample metadata")
        details = metadata["map_metadata"][role]
        if role != "input" and details.get("encoding") != "linear_data":
            raise ValueError(f"Numeric {role} reference must explicitly declare linear_data")
        if codes[role].dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError(f"Native {role} must retain integer source codes")
        if details.get("sample_bits") is not None and details["sample_bits"] != np.iinfo(codes[role].dtype).bits:
            raise ValueError(f"Decoded {role} precision differs from sample metadata")
    rgb, valid = display_rgb(codes["input"], metadata["map_metadata"]["input"]["encoding"])
    return rgb, valid, normal_codes(codes["normal"]), scalar_height(codes["height"])


def angular_metrics(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[-1] != 3 or prediction.shape[:2] != valid.shape or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("Normal metric shape/validity mismatch")
    vectors = []
    for image in (prediction, reference):
        # Invalid alpha pixels are excluded from both validation and reduction.
        # A zero/undefined reference normal there must not invalidate the score.
        vector = image[valid].astype(np.float64) * 2 - 1
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        if not np.isfinite(vector).all() or np.any(length <= 1e-12):
            raise ValueError("Nonfinite or zero-length normal")
        vectors.append(vector / length)
    angle = np.degrees(np.arccos(np.clip(np.sum(vectors[0] * vectors[1], axis=-1), -1, 1)))
    return {"mean_degrees": float(angle.mean()), "median_degrees": float(np.median(angle)), "p90_degrees": float(np.percentile(angle, 90)), "within_10_degrees_fraction": float((angle < 10).mean())}


def integrate_opengl_normals(normals: np.ndarray) -> np.ndarray:
    """Neumann least-squares integration in pixel coordinates; zero mean only.

    OpenGL X=-dh/dx and Y=+dh/d(image row), each divided by positive Z.
    Output units, offset and displacement amplitude are uncalibrated.
    """
    from scipy.fft import dctn, idctn
    vector = normals.astype(np.float64) * 2 - 1
    if vector.ndim != 3 or vector.shape[-1] != 3 or not np.isfinite(vector).all() or np.any(vector[..., 2] <= 1e-4):
        raise ValueError("Height integration requires finite forward-facing normals with Z > 1e-4")
    p, q = -vector[..., 0] / vector[..., 2], vector[..., 1] / vector[..., 2]
    gx, gy = (p[:, :-1] + p[:, 1:]) / 2, (q[:-1] + q[1:]) / 2
    rhs = np.zeros(p.shape, dtype=np.float64)
    rhs[:, :-1] -= gx
    rhs[:, 1:] += gx
    rhs[:-1] -= gy
    rhs[1:] += gy
    rows, columns = p.shape
    eigenvalues = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows))[:, None] + (2 - 2 * np.cos(np.pi * np.arange(columns) / columns))[None, :]
    transformed = dctn(rhs, type=2, norm="ortho")
    eigenvalues[0, 0] = 1
    transformed /= eigenvalues
    transformed[0, 0] = 0
    return np.ascontiguousarray(idctn(transformed, type=2, norm="ortho"), dtype=np.float32)


def write_float_exr(path: Path, values: np.ndarray) -> None:
    import OpenEXR
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] != 3) or min(values.shape[:2]) < 1 or not np.isfinite(values).all():
        raise ValueError("Float EXR needs finite nonempty scalar or RGB data")
    channels = {"Y": values} if values.ndim == 2 else {name: np.ascontiguousarray(values[..., number]) for number, name in enumerate("RGB")}
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage, "textureStudioEncoding": "linear_data"}, dict(channels)).write(str(path))
    decoded = OpenEXR.File(str(path), separate_channels=True)
    if any(decoded.channels()[name].pixels.dtype != np.float32 or not np.array_equal(decoded.channels()[name].pixels, data) for name, data in channels.items()):
        raise ValueError(f"Float EXR round-trip changed data: {path}")


def infer_normals(rgb: np.ndarray, session: Any, helpers: Any, deadline: float, overlap: str) -> tuple[np.ndarray, int]:
    stride = {"SMALL": 256 - 256 // 6, "MEDIUM": 192, "LARGE": 128}[overlap]
    gray = np.mean(rgb, axis=0, keepdims=True).astype(np.float32)
    tiles, padding = helpers.tiles_split(gray, (256, 256), (stride, stride))
    predictions = []
    for number, tile in enumerate(tiles):
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU comparison budget exhausted before completing all native tiles")
        output = session.run(None, {"input": np.ascontiguousarray(tile[None], dtype=np.float32)})[0]
        if output.shape != (1, 3, 256, 256) or not np.isfinite(output).all():
            raise ValueError("Pinned ONNX model returned invalid native normal tile shape/values")
        predictions.append(output[0])
        if number % 10 == 0:
            print(f"  tile {number + 1}/{len(tiles)}", flush=True)
    merged = helpers.tiles_merge(predictions, (stride, stride), (3, gray.shape[1], gray.shape[2]), padding)
    normal = np.ascontiguousarray(helpers.normalize(merged).transpose(1, 2, 0), dtype=np.float32)
    if normal.shape != (rgb.shape[1], rgb.shape[2], 3) or not np.isfinite(normal).all():
        raise ValueError("Model returned invalid full-resolution normals")
    return normal, len(tiles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "out/material-training/external/deepbump")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--materials", nargs="+", default=list(DEFAULT_MATERIALS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--overlap", choices=("SMALL", "MEDIUM", "LARGE"), default="LARGE")
    parser.add_argument("--integrate-height", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4 or not 1 <= args.seconds <= 600 or not 1 <= len(args.materials) <= 12:
        parser.error("Use 1–4 CPU threads, 1–600 seconds and 1–12 explicitly selected materials")
    if args.output.exists():
        parser.error("Output already exists; choose a new comparison directory")
    provenance = model_files(args.model_dir, args.download)
    index, samples = select_samples(args.dataset, args.materials)
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model_dir / "deepbump256.onnx"), options, providers=["CPUExecutionProvider"])
    spec = importlib.util.spec_from_file_location("deepbump_pinned_tiling", args.model_dir / "utils_inference.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    report = {"schema": "texture-studio-deepbump-comparison-v1", "started_utc": datetime.now(timezone.utc).isoformat(), "revision": REVISION, "license": "GPL-3.0", "files": provenance, "runtime": {"onnxruntime": ort.__version__, "providers": session.get_providers(), "cpu_threads": args.threads}, "validation_scope": index["validation_scope"], "dataset_index_sha256_before": sha256(args.dataset / "dataset.json"), "requested_materials": args.materials, "native_resizing": False, "input": "source display RGB codes / dtype maximum; declared linear diffuse converted to sRGB for model input only; no target gamma conversion", "alpha_mask_policy": "Source alpha with <=8 UInt16 codes of rounding tolerated; otherwise excluded with 8-pixel context margin. This is a metric mask, not a guarantee of clean convolutional context. Global height integration skipped for partially masked inputs.", "normal_convention": "OpenGL +Y interpreted from upstream height gradient signs (-Nx,+Ny); alternative Y score diagnostic only", "height_policy": "Optional raw least-squares integration, mean zero, unknown offset/amplitude; no min/max stretching or calibrated absolute-height claim", "samples": [], "status": "running"}
    try:
        for directory, metadata in samples:
            sample_start = time.monotonic()
            print(f"Evaluating {metadata['sample_id']}", flush=True)
            paths, codes, snapshots = {}, {}, {}
            for role in ("input", "normal", "height"):
                path = checked_relative(directory, metadata["maps"][role])
                details = metadata["map_metadata"][role]
                snapshots[role] = sha256(path)
                if snapshots[role] != details["sample_sha256"]:
                    raise ValueError(f"Prepared {role} checksum mismatch: {path}")
                codes[role], _ = read_png(path)
                paths[role] = path
            rgb, valid, reference, target = validate_arrays(codes, metadata)
            normal, tiles = infer_normals(rgb, session, helpers, start + args.seconds, args.overlap)
            destination = args.output / metadata["sample_id"]
            destination.mkdir()
            np.save(destination / "normal_opengl.float32.npy", normal, allow_pickle=False)
            write_float_exr(destination / "normal_opengl.float32.exr", normal)
            np.save(destination / "evaluation_valid_mask.npy", valid, allow_pickle=False)
            flipped = normal.copy()
            flipped[..., 1] = 1 - flipped[..., 1]
            flat = np.empty_like(normal)
            flat[...] = [0.5, 0.5, 1]
            result = {"sample_id": metadata["sample_id"], "material_id": metadata["material_id"], "source_url": metadata.get("source_url"), "source_license": metadata.get("source_license", metadata.get("license", "unknown")), "dimensions": list(normal.shape[:2][::-1]), "source_input_bits": np.iinfo(codes["input"].dtype).bits, "source_normal_bits": np.iinfo(codes["normal"].dtype).bits, "valid_pixel_fraction": float(valid.mean()), "native_tiles": tiles, "normal_against_reference": angular_metrics(normal, reference, valid), "alternative_flipped_y_diagnostic": angular_metrics(flipped, reference, valid), "flat_normal_baseline": angular_metrics(flat, reference, valid), "source_png_sha256_before": snapshots}
            if args.integrate_height and valid.all():
                integrated = integrate_opengl_normals(normal)
                np.save(destination / "integrated_relative_height.float32.npy", integrated, allow_pickle=False)
                write_float_exr(destination / "integrated_relative_height.float32.exr", integrated)
                values, truth = integrated[valid].astype(np.float64), target[valid].astype(np.float64)
                centered_values, centered_truth = values - values.mean(), truth - truth.mean()
                denominator = np.linalg.norm(centered_values) * np.linalg.norm(centered_truth)
                result["height_shape_correlation_diagnostic"] = None if denominator <= 1e-12 else float(np.dot(centered_values, centered_truth) / denominator)
                result["integrated_height_range_uncalibrated"] = [float(integrated.min()), float(integrated.max())]
            elif args.integrate_height:
                result["height_integration_skipped"] = "Partially invalid alpha input: global gradient integration would propagate invalid-region predictions"
            result["source_png_sha256_after"] = {role: sha256(path) for role, path in paths.items()}
            if result["source_png_sha256_after"] != snapshots:
                raise ValueError("Dataset source PNG changed during comparison")
            result["seconds"] = time.monotonic() - sample_start
            report["samples"].append(result)
            print(f"  mean angular error {result['normal_against_reference']['mean_degrees']:.3f}°, {result['seconds']:.2f}s", flush=True)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
    report["dataset_index_sha256_after"] = sha256(args.dataset / "dataset.json")
    report["dataset_index_unchanged"] = report["dataset_index_sha256_before"] == report["dataset_index_sha256_after"]
    if not report["dataset_index_unchanged"]:
        report["status"] = "incomplete"
        report["error"] = "Dataset index changed during comparison"
    report["seconds"] = time.monotonic() - start
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "completed_samples": len(report["samples"]), "seconds": report["seconds"], "report": str(args.output / "comparison.json"), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
