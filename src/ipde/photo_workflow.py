"""Portrait workflows. Original arrays and explicitly derived products stay separate."""
from __future__ import annotations

import importlib.util
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pillow_heif

from .extractor import Asset, Discovery, ExtractionError, _jsonable, discover_file
from .formats import (FormatError, arrays_bit_equal, sha256_array, sha256_file,
                      verify_exr, verify_npy, verify_png, write_exr, write_npy, write_png)
from .learned_depth import LearnedDepthConfig, infer_learned_depth
from .registration import estimate_display_registration, project_left_depth_to_display
from .spatial import RaftStereoOptions, run_raft_stereo


class MediaWorkflowError(RuntimeError):
    """An actionable media operation failure; no lossy fallback is performed."""


def native_bridge() -> Path | None:
    """Find the packaged helper, or compile the checkout helper once on macOS."""
    if sys.platform != "darwin":
        return None
    resource = Path(__file__).resolve()
    candidates = []
    if os.environ.get("IPDE_MEDIA_BRIDGE"):
        candidates.append(Path(os.environ["IPDE_MEDIA_BRIDGE"]))
    candidates.extend(parent / "media-bridge" for parent in resource.parents)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    root = resource.parents[2]
    source = root / "native" / "media_bridge.swift"
    if not source.is_file() or not shutil.which("xcrun"):
        return None
    cache = Path(tempfile.gettempdir()) / ("ipde-media-bridge-" + sha256_file(source)[:20])
    if not cache.is_file():
        compiled = cache.with_name(cache.name + "-" + uuid.uuid4().hex)
        result = subprocess.run(["xcrun", "swiftc", str(source), "-o", str(compiled),
                                 "-framework", "Foundation", "-framework", "ImageIO",
                                 "-framework", "Vision", "-framework", "CoreImage",
                                 "-framework", "CoreGraphics"], capture_output=True, text=True)
        if result.returncode:
            compiled.unlink(missing_ok=True)
            raise MediaWorkflowError("Native media helper could not compile: " + result.stderr[-3000:])
        os.replace(compiled, cache)
    return cache


def run_native(action: str, source: Path, output: Path, *extra: str) -> dict[str, Any]:
    helper = native_bridge()
    if helper is None:
        raise MediaWorkflowError("This operation requires the bundled macOS ImageIO/Vision media helper.")
    result = subprocess.run([str(helper), action, str(source), str(output), *extra],
                            capture_output=True, text=True)
    if result.returncode:
        raise MediaWorkflowError(result.stderr.strip() or "Native media operation failed")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise MediaWorkflowError("Native media helper returned an invalid report") from exc


def array_summary(array: np.ndarray) -> dict[str, Any]:
    a = np.asarray(array)
    result: dict[str, Any] = {"dtype": a.dtype.name, "shape": list(a.shape),
                             "width": a.shape[1], "height": a.shape[0], "sha256": sha256_array(a)}
    finite = np.isfinite(a)
    result["finite_fraction"] = float(np.count_nonzero(finite) / a.size)
    if finite.any():
        result.update(min=float(a[finite].min()), max=float(a[finite].max()))
    return result


def is_person_matte(asset: Asset) -> bool:
    name = asset.semantic_name.lower()
    return ("portrait" in name and "matte" in name) or ("matte" in name and any(
        part in name for part in ("skin", "hair", "teeth", "glasses", "eyes", "person")))


def discover_photo(source: Path | str) -> Discovery:
    discovery = discover_file(Path(source))
    # The general extractor inventories auxiliary images and stereo views. The
    # portrait editor also needs every top-level color image, with stable indices.
    covered = {a.parent_image_index for a in discovery.assets
               if a.kind in ("display_view", "spatial_view", "color_view")}
    heif = pillow_heif.open_heif(discovery.source.read_bytes(), convert_hdr_to_8bit=False,
                                hdr_to_16bit=False, remove_stride=True)
    for index, image in enumerate(heif):
        if index not in covered:
            context = discovery.top_level_images[index]
            discovery.assets.append(Asset(kind="color_view", parent_image_index=index, ordinal=0,
                array=np.array(image, copy=True, order="C"), mode=image.mode,
                source_bit_depth=int(context["source_bit_depth"]),
                semantic_name="display" if index == discovery.primary_index else f"image_{index}",
                metadata={"coordinate_system": f"encoded image {index} grid", "derived": False}))
    return discovery


def display_asset(discovery: Discovery) -> Asset:
    for asset in discovery.assets:
        if asset.semantic_name == "display":
            return asset
    for asset in discovery.assets:
        if asset.kind in ("color_view", "spatial_view") and asset.parent_image_index == discovery.primary_index:
            return asset
    raise MediaWorkflowError("No primary color image was decoded")


def preview_array(array: np.ndarray, *, nominal_max: float | None = None, size: int = 1200) -> np.ndarray:
    """Create a disposable viewing copy. No exported scientific array enters this path."""
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[2] > 4:
        raise MediaWorkflowError("Cannot preview arrays with more than four channels")
    if a.dtype == np.uint8:
        shown = a.copy()
    elif np.issubdtype(a.dtype, np.integer):
        limit = nominal_max or float(np.iinfo(a.dtype).max)
        shown = np.round(np.clip(a.astype(np.float64) / limit, 0, 1) * 255).astype(np.uint8)
    else:
        finite = np.isfinite(a)
        values = a[finite]
        lo, hi = np.percentile(values, (1, 99)) if values.size else (0.0, 1.0)
        if a.ndim == 3:
            lo, hi = 0.0, 1.0
        scaled = (np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo) - lo) / max(float(hi - lo), 1e-12)
        if a.ndim == 3:
            scaled = np.where(scaled <= 0.0031308, 12.92 * scaled,
                              1.055 * np.maximum(scaled, 0) ** (1 / 2.4) - 0.055)
        shown = np.round(np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    h, w = shown.shape[:2]
    if max(h, w) > size:
        shown = cv2.resize(shown, (max(1, round(w * size / max(h, w))),
                                  max(1, round(h * size / max(h, w)))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(shown)


def write_preview(path: Path, array: np.ndarray, *, nominal_max: float | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, preview_array(array, nominal_max=nominal_max))
    return str(path.resolve())


def inspect_photo(source: Path | str, preview_dir: Path | None = None) -> dict[str, Any]:
    discovery = discover_photo(source)
    helper = native_bridge()
    assets = []
    for index, asset in enumerate(discovery.assets):
        item = {"index": index, "kind": asset.kind, "name": asset.semantic_name,
                "semantic_name": asset.semantic_name, "parent_image_index": asset.parent_image_index,
                "source_bit_depth": asset.source_bit_depth, "person_matte": is_person_matte(asset),
                "grid": f"image-{asset.parent_image_index}", "metadata": _jsonable(asset.metadata),
                **array_summary(asset.array)}
        if preview_dir:
            item["preview_path"] = write_preview(preview_dir / f"asset-{index}.png", asset.array,
                nominal_max=float((1 << asset.source_bit_depth) - 1) if asset.array.dtype == np.uint16 else None)
        assets.append(item)
    display = display_asset(discovery)
    preview = next((a.get("preview_path") for a in assets
                    if discovery.assets[a["index"]] is display), None)
    warnings = ["Portrait and semantic people mattes are intended for people; preview every selected layer.",
                "Previews are display conversions. Original exports retain the decoded samples exactly.",
                "Upscaling and AI depth estimates are derived approximations and add no measured precision."]
    if discovery.spatial_metadata_warning:
        warnings.append(discovery.spatial_metadata_warning)
    capabilities = _photo_capabilities(discovery, warnings, helper)
    known_native_aux = {"portrait_effects_matte", "semantic_skin_matte", "semantic_hair_matte",
                        "semantic_teeth_matte", "semantic_glasses_matte", "semantic_sky_matte", "hdr_gain_map", "iso_gain_map"}
    for record, asset in zip(assets, discovery.assets, strict=True):
        record["replaceable"] = bool(capabilities["rewrite"] and (asset.kind == "native_depth" or
            (asset.kind == "auxiliary" and asset.semantic_name in known_native_aux and asset.array.ndim == 2)))
    return {"ok": True, "operation": "photo-inspect", "source": str(discovery.source),
            "source_sha256": discovery.source_sha256, "preview_path": preview, "assets": assets,
            "spatial": discovery.spatial_photo, "capabilities": capabilities,
            "warnings": warnings}


def _photo_capabilities(discovery: Discovery, warnings: list[str], helper: Path | None) -> dict[str, bool]:
    """A native writer's existence is not evidence it preserves this container."""
    privacy_available = False
    rewrite_available = False
    with tempfile.TemporaryDirectory(prefix="ipde-capability-probe-") as directory:
        root = Path(directory)
        if helper is not None or importlib.util.find_spec("ipde.heif_privacy"):
            try:
                rewrite_photo(discovery.source, root / "privacy", privacy=True)
            except Exception as exc:
                warnings.append("Privacy export unavailable for this container: " + str(exc))
            else:
                privacy_available = True
        if helper is not None:
            candidate = next(((index, asset) for index, asset in enumerate(discovery.assets)
                              if asset.kind == "native_depth"), None)
            if candidate is None:
                candidate = next(((index, asset) for index, asset in enumerate(discovery.assets)
                                  if asset.kind == "auxiliary" and is_person_matte(asset)), None)
            if candidate is not None:
                index, asset = candidate
                replacement = root / "unchanged.npy"
                write_npy(replacement, asset.array)
                try:
                    rewrite_photo(discovery.source, root / "repair", replacements={index: replacement})
                except Exception as exc:
                    warnings.append("Embedded repair unavailable for this container: " + str(exc))
                else:
                    rewrite_available = True
    return {"rewrite": rewrite_available, "privacy": privacy_available,
            "sensor": False, "person_mask": helper is not None}


def _asset(discovery: Discovery, index: int | None) -> Asset:
    if index is None or index < 0 or index >= len(discovery.assets):
        raise MediaWorkflowError("Select an available asset")
    return discovery.assets[index]


def _registered_plane(asset: Asset, display: Asset) -> np.ndarray:
    a = asset.array
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise MediaWorkflowError("Selected map must contain a single channel")
    if asset.parent_image_index != display.parent_image_index:
        raise MediaWorkflowError("Selected map belongs to another camera/image; implicit cross-camera resizing is refused")
    h, w = display.array.shape[:2]
    ah, aw = a.shape
    # Allow only pixel-rounding differences, never stretch a map with a different aspect.
    if abs(aw / ah - w / h) > max(1 / ah, 1 / h):
        raise MediaWorkflowError("Map and color image have different aspect ratios; their registration is unknown")
    return a


def upscale_depth(discovery: Discovery, index: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    display = display_asset(discovery)
    if index is None:
        options = [a for a in discovery.assets if a.kind in ("native_depth", "depth")
                   and a.parent_image_index == display.parent_image_index]
        if not options:
            raise MediaWorkflowError("No embedded depth map is registered to the display image")
        asset = next((a for a in options if a.kind == "native_depth"), options[0])
    else:
        asset = _asset(discovery, index)
        if asset.kind not in ("depth", "native_depth"):
            raise MediaWorkflowError("Select an embedded depth or disparity plane")
    raw = _registered_plane(asset, display)
    h, w = display.array.shape[:2]
    finite = np.isfinite(raw)
    if not finite.any():
        raise MediaWorkflowError("Embedded depth contains no finite samples")
    weights = cv2.resize(finite.astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)
    values = cv2.resize(np.where(finite, raw, 0).astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)
    values /= np.maximum(weights, 1e-15)
    color = display.array[..., :3].astype(np.float64)
    limit = float((1 << display.source_bit_depth) - 1) if np.issubdtype(color.dtype, np.integer) else 1.0
    # source dtype was converted above; preserve the original integer range explicitly.
    if np.issubdtype(display.array.dtype, np.integer):
        limit = float((1 << display.source_bit_depth) - 1)
    guide = np.mean(color / limit, axis=2)
    radius = max(2, round(max(h / raw.shape[0], w / raw.shape[1])))
    kernel = (2 * radius + 1, 2 * radius + 1)
    box = lambda v: cv2.boxFilter(v, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_g, mean_v = box(guide), box(values)
    covariance = box(guide * values) - mean_g * mean_v
    variance = box(guide * guide) - mean_g * mean_g
    slope = covariance / (variance + 1e-4)
    intercept = mean_v - slope * mean_g
    derived = box(slope) * guide + box(intercept)
    derived = np.clip(derived, float(raw[finite].min()), float(raw[finite].max())).astype(np.float32)
    valid = cv2.resize(finite.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    derived[~valid | (weights < 1e-12)] = np.nan
    return np.ascontiguousarray(derived), {"derived": True, "method": "RGB-guided interpolation",
        "grid": "display", "source_map": asset.semantic_name, "source_shape": list(raw.shape),
        "representation": asset.metadata.get("representation", asset.metadata.get("representation_name", "encoded samples")),
        "accuracy": asset.metadata.get("accuracy"), "measured_precision_added": False}


def compose_mattes(discovery: Discovery, indices: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not indices:
        raise MediaWorkflowError("Select at least one portrait/semantic people matte")
    display = display_asset(discovery)
    rgb = display.array[..., :3]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype not in (np.uint8, np.uint16):
        raise MediaWorkflowError("Cutout color image must be unsigned 8/16-bit RGB")
    h, w = rgb.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float64)
    names = []
    for index in dict.fromkeys(indices):
        asset = _asset(discovery, index)
        if not is_person_matte(asset):
            raise MediaWorkflowError(f"{asset.semantic_name} is not a portrait or semantic people matte")
        plane = _registered_plane(asset, display)
        if not np.issubdtype(plane.dtype, np.unsignedinteger):
            raise MediaWorkflowError("Embedded matte coverage must use unsigned integer samples")
        maximum = float((1 << asset.source_bit_depth) - 1)
        if np.any(plane > maximum):
            raise MediaWorkflowError("Matte samples exceed their declared encoded precision")
        coverage = cv2.resize(plane.astype(np.float64) / maximum, (w, h), interpolation=cv2.INTER_LINEAR)
        # Union coverage, not alpha addition: overlapping semantic layers do not
        # make a partially transparent portrait edge artificially opaque.
        alpha = np.maximum(alpha, coverage)
        names.append(asset.semantic_name)
    if display.array.shape[2] == 4:
        alpha *= display.array[..., 3].astype(np.float64) / np.iinfo(rgb.dtype).max
    encoded_alpha = np.rint(np.clip(alpha, 0, 1) * np.iinfo(rgb.dtype).max).astype(rgb.dtype)
    rgba = np.concatenate((rgb, encoded_alpha[..., None]), axis=2)
    profile = discovery.top_level_images[display.parent_image_index].get("color_profile")
    return rgba, encoded_alpha, {"color_profile": profile, "derived": True, "method": "maximum coverage union",
        "matte_layers": names, "grid": "display", "alpha_storage_bits": rgb.dtype.itemsize * 8,
        "rgb_samples_unchanged": True, "alpha_mode": "straight", "source_color_bit_depth": display.source_bit_depth}


def _tiff_profile_tags(metadata: dict[str, Any] | None) -> list[tuple]:
    profile = (metadata or {}).get("color_profile") or {}
    if profile.get("type") in ("prof", "rICC") and isinstance(profile.get("data"), dict):
        value = base64.b64decode(profile["data"].get("data", ""), validate=True)
        return [(34675, "B", len(value), value, False)] if value else []
    return []


def export_array(output_dir: Path, name: str, array: np.ndarray, format: str,
                 *, role: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = {"tiff": "tif"}.get(format, format)
    final = output_dir / f"{name}.{extension}"
    if final.exists():
        raise MediaWorkflowError(f"Output already exists: {final}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".ipde-", suffix="." + extension, dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if format == "npy":
            write_npy(temporary, array); verify_npy(temporary, array)
        elif format == "png":
            write_png(temporary, array)
            profile = (metadata or {}).get("color_profile") or {}
            if profile.get("type") in ("prof", "rICC") and isinstance(profile.get("data"), dict):
                payload = base64.b64decode(profile["data"].get("data", ""), validate=True)
                if payload:
                    from .formats import _png_chunk
                    import zlib
                    # Preserve RGB interpretation without touching any samples.
                    png = temporary.read_bytes()
                    temporary.write_bytes(png[:33] + _png_chunk(b"iCCP", b"Source ICC\x00\x00" + zlib.compress(payload)) + png[33:])
            verify_png(temporary, array)
        elif format == "exr":
            write_exr(temporary, array); verify_exr(temporary, array)
        elif format == "tiff":
            try:
                import tifffile
            except ImportError as exc:
                raise MediaWorkflowError("TIFF export requires tifffile; choose NPY for exact data export") from exc
            tifffile.imwrite(temporary, array, metadata=None, compression=None,
                            photometric="rgb" if array.ndim == 3 and array.shape[2] in (3, 4) else "minisblack",
                            extrasamples="unassalpha" if array.ndim == 3 and array.shape[2] == 4 else None,
                            extratags=_tiff_profile_tags(metadata))
            if not arrays_bit_equal(array, tifffile.imread(temporary)):
                raise MediaWorkflowError("TIFF verification changed the sample bits")
        else:
            raise MediaWorkflowError(f"Unsupported lossless output format: {format}")
        # Atomic, no-clobber install so concurrent export cannot replace a result.
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(final.resolve()), "role": role, "verified_bit_exact": True,
            **array_summary(array), "metadata": _jsonable(metadata or {})}


def export_photo(source: Path | str, output_dir: Path, operation: str, format: str = "npy",
                 asset_index: int | None = None, matte_indices: list[int] | None = None,
                 learned: LearnedDepthConfig | None = None, raft: RaftStereoOptions | None = None) -> dict[str, Any]:
    discovery = discover_photo(source)
    display = display_asset(discovery)
    name = discovery.source.stem + "-" + uuid.uuid4().hex[:8]
    products: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    warnings: list[str] = []
    if operation == "original":
        chosen = [_asset(discovery, asset_index)] if asset_index is not None else discovery.assets
        products = [(f"original-{i}-{a.semantic_name}", a.array,
                     {"derived": False, "source_bit_depth": a.source_bit_depth,
                      "parent_image_index": a.parent_image_index,
                      "color_profile": discovery.top_level_images[a.parent_image_index].get("color_profile")
                          if a.kind in ("color_view", "display_view", "spatial_view") else None, **a.metadata}) for i, a in enumerate(chosen)]
    elif operation == "depth-upscale":
        values, meta = upscale_depth(discovery, asset_index)
        products.append(("depth-upscale", values, meta))
        warnings.append("This display-grid map is an interpolation of embedded samples; it adds no measured precision.")
    elif operation in ("cutout", "isolate"):
        indices = [asset_index] if operation == "isolate" and asset_index is not None else matte_indices or []
        values, alpha, meta = compose_mattes(discovery, indices)
        products.append((operation, values, meta))
    elif operation == "learned-depth":
        config = learned or LearnedDepthConfig()
        config = replace(config, input_max_value=float((1 << display.source_bit_depth) - 1)
                         if np.issubdtype(display.array.dtype, np.integer) else 1.0)
        result = infer_learned_depth(display.array[..., :3], config, reference_label="display")
        products.extend([("learned-depth", result.source_depth, result.metadata),
                         ("learned-native", result.native_depth, {**result.metadata, "grid": "model-native"})])
        warnings.append("Learned depth is an estimate. Relative models require calibration before metric repair.")
    elif operation == "raft-depth":
        spatial = discovery.spatial_photo
        if not spatial:
            raise MediaWorkflowError("RAFT-Stereo requires a calibrated spatial pair; portraits use monocular depth models")
        left = next(a for a in discovery.assets if a.semantic_name == "spatial_left")
        right = next(a for a in discovery.assets if a.semantic_name == "spatial_right")
        result = run_raft_stereo(left.array, right.array, spatial, raft or RaftStereoOptions())
        registration = estimate_display_registration(discovery)
        # estimate_display_registration owns its validated row-registration record;
        # retain it when transporting a left-grid prediction to the display.
        values = project_left_depth_to_display(result.depth_meters, discovery, registration)
        products.extend([("raft-depth-display", values, {"derived": True, "units": "meters", "grid": "display",
                          "registration": registration, "raft": result.details}),
                         ("raft-depth-left", result.depth_meters, {"derived": True, "units": "meters", "grid": "spatial_left"})])
        warnings.append("Unregistered or unsupported depth remains NaN. Review the derived map before embedding it.")
    else:
        raise MediaWorkflowError(f"Unknown photo operation {operation}")
    outputs = []
    try:
        for role, value, meta in products:
            outputs.append(export_array(output_dir, f"{name}-{role}", value, format, role=role, metadata=meta))
    except Exception:
        for output in outputs:
            Path(output["path"]).unlink(missing_ok=True)
        raise
    return {"ok": True, "operation": operation, "source": str(discovery.source),
            "outputs": outputs, "warnings": warnings}


def read_media_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".npy", ".npz"):
        from .array_storage import read_array
        return np.asarray(read_array(path, mmap_mode=None))
    if suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    if suffix == ".png":
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise MediaWorkflowError("PNG replacement could not be decoded")
        if value.ndim == 3 and value.shape[2] == 3:
            value = value[..., ::-1]
        elif value.ndim == 3 and value.shape[2] == 4:
            value = value[..., [2, 1, 0, 3]]
        return np.ascontiguousarray(value)
    if suffix == ".exr":
        import OpenEXR
        with OpenEXR.File(str(path)) as image:
            channels = image.channels()
            if "Y" in channels:
                return channels["Y"].pixels.copy()
            if "RGB" in channels:
                return channels["RGB"].pixels.copy()
        raise MediaWorkflowError("EXR replacement needs a Y or RGB plane")
    raise MediaWorkflowError("Replacement must use NPY, NPZ, PNG, TIFF or OpenEXR")


def _xml_identity(payload: bytes) -> Any:
    import xml.etree.ElementTree as ET
    if not payload:
        return None
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MediaWorkflowError("Auxiliary XMP cannot be validated") from exc
    def identity(element):
        return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
                tuple(sorted((identity(child) for child in element), key=repr)))
    return identity(root)


def _verify_native_metadata(old: Asset, new: Asset) -> None:
    if old.kind != "native_depth":
        return
    for key in ("representation", "accuracy"):
        if old.metadata.get(key) != new.metadata.get(key):
            raise MediaWorkflowError("Native writer changed the depth representation or accuracy metadata")
    old_description, new_description = old.metadata.get("description", {}), new.metadata.get("description", {})
    for key in ("Width", "Height", "PixelFormat", "Orientation"):
        if old_description.get(key) != new_description.get(key):
            raise MediaWorkflowError("Native writer changed the depth coordinate grid or floating storage format")
    old_xmp = next((block.get("data", b"") for block in old.metadata_blocks
                    if block.get("content_type") == "application/rdf+xml"), b"")
    new_xmp = next((block.get("data", b"") for block in new.metadata_blocks
                    if block.get("content_type") == "application/rdf+xml"), b"")
    if _xml_identity(old_xmp) != _xml_identity(new_xmp):
        raise MediaWorkflowError("Native writer changed depth calibration XMP; no output installed")


def _orientation_inventory(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compare effective visual orientations, including hidden auxiliary grids."""
    properties = metadata.get("properties", {})
    records: dict[str, Any] = {}
    for index, image in enumerate(properties.get("images", [])):
        records[f"image:{index}"] = int(image.get("Orientation", 1))
    contents = properties.get("global", {}).get("{FileContents}", {})
    for index, image in enumerate(contents.get("Images", [])):
        records[f"contents-image:{index}"] = int(image.get("Orientation", 1))
        for auxiliary in image.get("AuxiliaryData", []):
            # Optional descriptive/style metadata has no pixel grid. Its
            # removal must not shift indices of real auxiliary images.
            if "Width" in auxiliary and "Height" in auxiliary:
                key = f"aux:{index}:" + str(auxiliary.get("AuxiliaryDataType", ""))
                records[key] = int(auxiliary.get("Orientation", 1))
    return records


def rewrite_photo(source: Path | str, output_dir: Path, *, privacy: bool = False,
                  replacements: dict[int, Path] | None = None, replacement_kind: str = "native") -> dict[str, Any]:
    """Verify fidelity and depth calibration before exposing rewritten containers."""
    discovery = discover_photo(source)
    replacements = replacements or {}
    if not privacy and not replacements:
        raise MediaWorkflowError("Select a replacement asset or enable privacy enhancement")
    if privacy and replacements:
        raise MediaWorkflowError("Repair and privacy export are separate verified operations; repair first, then export privacy")
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / ((uuid.uuid4().hex if privacy else discovery.source.stem + "-repaired-" + uuid.uuid4().hex[:8]) + ".heic")
    with tempfile.TemporaryDirectory(prefix=".ipde-rewrite-", dir=output_dir) as temporary_dir:
        root = Path(temporary_dir)
        replacement_records = []
        expected_arrays: dict[int, np.ndarray] = {}
        if replacements:
            inventory = run_native("inventory", discovery.source, root / "native-inventory")
        else:
            inventory = {}
        for index, path in replacements.items():
            asset = _asset(discovery, index)
            replacement = read_media_array(path)
            if asset.kind == "native_depth":
                if replacement.dtype not in (np.dtype("float16"), np.dtype("float32")) or replacement.ndim != 2:
                    raise MediaWorkflowError("Replacement depth must be unnormalized float16/float32 values")
                if replacement.shape != asset.array.shape:
                    display = display_asset(discovery)
                    if replacement.shape != display.array.shape[:2] or asset.parent_image_index != display.parent_image_index:
                        raise MediaWorkflowError("Replacement grid is neither the embedded map nor its registered display grid")
                    replacement = cv2.resize(replacement, (asset.array.shape[1], asset.array.shape[0]), interpolation=cv2.INTER_AREA)
                if not np.isfinite(replacement).all() or np.any(replacement <= 0):
                    raise MediaWorkflowError("Embedded physical depth/disparity needs finite positive values; review invalid estimates before repair")
                target_kind = asset.metadata["representation"]
                if replacement_kind not in ("native", "depth", "disparity"):
                    raise MediaWorkflowError("Replacement kind must be native, depth or disparity")
                if replacement_kind != "native" and replacement_kind != target_kind:
                    replacement = 1.0 / replacement.astype(np.float64)
                replacement = np.ascontiguousarray(replacement, dtype=asset.array.dtype)
                if not np.isfinite(replacement).all() or np.any(replacement <= 0):
                    raise MediaWorkflowError("Replacement values exceed the embedded map's floating representation")
                semantic = target_kind
            elif asset.kind == "auxiliary":
                if replacement.shape != asset.array.shape or replacement.dtype != asset.array.dtype:
                    raise MediaWorkflowError("Auxiliary replacement must match the selected original grid and dtype exactly")
                if replacement_kind != "native":
                    raise MediaWorkflowError("Matte/gain-map replacement uses native values; depth units apply only to depth/disparity")
                semantic = asset.semantic_name
            else:
                raise MediaWorkflowError("Main color/subimage and encoded depth replacement cannot currently retain every original HEIC sample. Choose a native depth, matte or gain-map plane.")
            native = next((record for record in inventory.get("auxiliary", [])
                           if record["semantic"] == semantic and record["parent"] == asset.parent_image_index), None)
            if native is None:
                raise MediaWorkflowError("ImageIO does not expose this auxiliary type for verified native replacement")
            description = native["description"]
            width, height, stride = (int(description[key]) for key in ("Width", "Height", "BytesPerRow"))
            if replacement.shape[:2] != (height, width) or replacement.ndim != 2 or stride < width * replacement.dtype.itemsize:
                raise MediaWorkflowError("This native auxiliary layout cannot be replaced without a format conversion")
            payload = root / f"replacement-{index}.bin"
            packed = bytearray(Path(native["path"]).read_bytes())
            row_bytes = width * replacement.dtype.itemsize
            for row in range(height):
                packed[row * stride:row * stride + row_bytes] = replacement[row].tobytes()
            payload.write_bytes(packed)
            expected_arrays[index] = np.ascontiguousarray(replacement)
            replacement_records.append({"index": index, "parent": asset.parent_image_index,
                "semantic": semantic, "path": str(payload), "width": width, "height": height,
                "dtype": replacement.dtype.name, "description": description})
        config = root / "config.json"
        config.write_text(json.dumps({"privacy": privacy, "replacements": replacement_records}))
        candidate = root / "candidate.heic"
        if privacy:
            # Prefer the native no-recompression source-copy operation. When its
            # metadata flags are ignored, a verified BMFF metadata-only rewrite
            # can retain the compressed image and every auxiliary payload.
            try:
                native_report = run_native("rewrite", discovery.source, candidate, str(config))
            except MediaWorkflowError as exc:
                try:
                    from .heif_privacy import scrub_heif
                except ImportError:
                    raise MediaWorkflowError(str(exc) + "; no verified lossless metadata rewrite is available") from exc
                candidate.unlink(missing_ok=True)
                native_report = scrub_heif(discovery.source, candidate)
                native_report["native_copy_limitation"] = str(exc)
        else:
            native_report = run_native("rewrite", discovery.source, candidate, str(config))
        actual = discover_photo(candidate)
        # EXIF-only orientation can change without changing decoded sample
        # bits. Compare the authoritative effective parent/aux orientations.
        if native_bridge() is not None:
            if not inventory:
                inventory = run_native("inventory", discovery.source, root / "orientation-before")
            candidate_inventory = run_native("inventory", candidate, root / "orientation-after")
            if _orientation_inventory(inventory) != _orientation_inventory(candidate_inventory):
                raise MediaWorkflowError("HEIC rewrite changed image/auxiliary orientation; no output installed")
        if len(actual.top_level_images) != len(discovery.top_level_images) or actual.primary_index != discovery.primary_index:
            raise MediaWorkflowError("HEIC writer changed the image inventory; no output installed")
        if _jsonable(actual.spatial_photo) != _jsonable(discovery.spatial_photo):
            raise MediaWorkflowError("HEIC writer changed spatial calibration/structure; no output installed")
        if len(actual.assets) != len(discovery.assets):
            raise MediaWorkflowError("HEIC writer dropped or added auxiliary images; no output installed")
        repaired_depth_parents = {discovery.assets[index].parent_image_index for index in expected_arrays
                                 if discovery.assets[index].kind == "native_depth"}
        for index, (old, new) in enumerate(zip(discovery.assets, actual.assets, strict=True)):
            if (old.kind, old.semantic_name, old.parent_image_index, old.source_bit_depth) != (new.kind, new.semantic_name, new.parent_image_index, new.source_bit_depth):
                raise MediaWorkflowError("HEIC writer changed auxiliary meaning/precision; no output installed")
            _verify_native_metadata(old, new)
            if index in expected_arrays:
                expected = expected_arrays[index]
            elif old.kind == "depth" and old.parent_image_index in repaired_depth_parents:
                # The encoded plane is regenerated from the replacement native
                # map. Exact native floats plus unchanged calibration XMP are
                # independently checked; do not demand old encoded samples.
                continue
            else:
                expected = old.array
            if not arrays_bit_equal(expected, new.array):
                raise MediaWorkflowError(f"HEIC rewrite changed {old.semantic_name} sample bits. Lossless repair is unavailable for this container; no output installed.")
        if privacy and not native_report.get("privacy_verified"):
            raise MediaWorkflowError("HEIC writer did not verify metadata removal; no output installed")
        if replacement_records:
            native_report["replacement_metadata_verified"] = True
        os.link(candidate, final)
    return {"ok": True, "operation": "privacy" if privacy else "repair", "source": str(discovery.source),
            "outputs": [{"path": str(final.resolve()), "role": "heic", "verified_bit_exact": True,
                         "sha256": sha256_file(final), "privacy_verified": privacy}],
            "warnings": ["Privacy enhancement removes recorded identifying metadata; visible people and scene content remain identifiable."] if privacy else [],
            "native": native_report}
