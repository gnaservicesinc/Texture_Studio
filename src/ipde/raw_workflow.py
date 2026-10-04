"""RAW capture with exact sensor copies and separately labelled rendered products."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .apple_imageio import decode_native_depth_buffer
from .extractor import _jsonable
from .formats import arrays_bit_equal, sha256_file
from .learned_depth import LearnedDepthConfig, infer_learned_depth
from .photo_workflow import (MediaWorkflowError, array_summary, export_array,
                             native_bridge, run_native, write_preview, preview_array)


@dataclass
class RawCapture:
    source: Path
    sensor: np.ndarray | None = None
    visible_sensor: np.ndarray | None = None
    rendered: np.ndarray | None = None
    sensor_metadata: dict[str, Any] = field(default_factory=dict)
    native_metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _decode_auxiliary(data: bytes, description: dict[str, Any]) -> np.ndarray | None:
    try:
        width, height = int(description["Width"]), int(description["Height"])
        stride, pixel = int(description["BytesPerRow"]), int(description["PixelFormat"])
        if pixel in {int.from_bytes(x, "big") for x in (b"hdis", b"fdis", b"hdep", b"fdep")}:
            return decode_native_depth_buffer(data, description)
        dtype = {int.from_bytes(b"L008", "big"): np.dtype("uint8"),
                 int.from_bytes(b"L016", "big"): np.dtype("uint16"),
                 int.from_bytes(b"L00h", "big"): np.dtype("float16"),
                 int.from_bytes(b"L00f", "big"): np.dtype("float32")}.get(pixel)
        if dtype is None or width <= 0 or height <= 0 or stride < width * dtype.itemsize or len(data) < stride * height:
            return None
        return np.ndarray((height, width), dtype=dtype, buffer=data,
                          strides=(stride, dtype.itemsize)).copy(order="C")
    except (KeyError, ValueError, TypeError):
        return None


def _capture_dng_rasters(capture: RawCapture) -> None:
    """Read CFA/LinearRaw TIFF pages without applying DNG linearization tables.

    ProRAW is often stored as demosaiced, JPEG-XL compressed 10-bit codes.
    Its uint16 container is not a 16-bit measurement, and LinearizationTable
    is calibration metadata, not an invitation to modify the stored raster.
    """
    try:
        import tifffile
    except ImportError:
        return
    if capture.source.suffix.lower() not in (".dng", ".tif", ".tiff"):
        return
    try:
        with tifffile.TiffFile(capture.source) as container:
            pages = []
            def visit(page, identity):
                record = {"ifd": identity, "shape": list(page.shape), "dtype": page.dtype.name,
                          "photometric": int(page.photometric), "compression": int(page.compression),
                          "tags": {tag.name: _jsonable(tag.value.tolist() if isinstance(tag.value, np.ndarray) else tag.value)
                                   for tag in page.tags.values()}}
                pages.append(record)
                if int(page.photometric) in (32803, 34892):
                    if capture.sensor is None:
                        try:
                            samples = page.asarray()
                        except Exception as exc:
                            capture.warnings.append(f"Stored DNG raster decoding unavailable (install imagecodecs for JPEG XL): {exc}")
                        else:
                            capture.sensor = np.array(samples, copy=True, order="C")
                            representation = "CFA sensor mosaic" if int(page.photometric) == 32803 else "stored demosaiced LinearRaw codes"
                            capture.sensor_metadata.update({"decoder": "tifffile/imagecodecs DNG stored raster",
                                "representation": representation, "source_ifd": identity,
                                "encoded_bits_per_sample": _jsonable(page.bitspersample),
                                "storage_dtype": samples.dtype.name, "linearization_applied": False,
                                "orientation_applied": False, "source_sha256": sha256_file(capture.source)})
                            capture.warnings.append("Stored DNG codes retain the source bit depth. Linearization tables/calibration are exported separately and remain unapplied.")
                    else:
                        record["sensor_already_decoded"] = True
                if page.pages:
                    for index, child in enumerate(page.pages):
                        visit(child, identity + "/" + str(index))
            for index, page in enumerate(container.pages):
                visit(page, str(index))
            capture.sensor_metadata["tiff_ifds"] = pages
    except Exception as exc:
        capture.warnings.append(f"DNG TIFF calibration inventory unavailable: {exc}")


def capture_raw(source: Path | str) -> RawCapture:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise MediaWorkflowError(f"Input is not a regular file: {path}")
    result = RawCapture(path)
    if importlib.util.find_spec("rawpy"):
        import rawpy
        try:
            with rawpy.imread(str(path)) as raw:
                result.sensor = np.array(raw.raw_image, copy=True, order="C")
                result.visible_sensor = np.array(raw.raw_image_visible, copy=True, order="C")
                for name in ("raw_pattern", "color_desc", "num_colors", "black_level_per_channel",
                             "camera_white_level_per_channel", "white_level", "camera_whitebalance",
                             "daylight_whitebalance", "color_matrix", "rgb_xyz_matrix", "tone_curve",
                             "raw_type", "raw_image", "raw_image_visible"):
                    if name in ("raw_image", "raw_image_visible"):
                        continue
                    try:
                        value = getattr(raw, name)
                        result.sensor_metadata[name] = value.tolist() if isinstance(value, np.ndarray) else _jsonable(value)
                    except (AttributeError, ValueError, RuntimeError):
                        pass
                sizes = raw.sizes
                result.sensor_metadata["sizes"] = {name: getattr(sizes, name) for name in (
                    "raw_height", "raw_width", "height", "width", "top_margin", "left_margin",
                    "iheight", "iwidth", "pixel_aspect", "flip") if hasattr(sizes, name)}
                result.sensor_metadata["sensor_dtype"] = result.sensor.dtype.name
                result.sensor_metadata["source_sha256"] = sha256_file(path)
                result.sensor_metadata["decoder"] = "LibRaw via rawpy; untouched full mosaic including sensor margins"
                result.sensor_metadata["rawpy_version"] = rawpy.__version__
                # CFA color indices are calibration information; retain the full
                # map when it is not a repeated Bayer pattern (e.g. X-Trans).
                if result.sensor_metadata.get("raw_pattern") is None:
                    result.sensor_metadata["raw_colors"] = np.asarray(raw.raw_colors).tolist()
                try:
                    result.rendered = raw.postprocess(output_bps=16, gamma=(1.0, 1.0),
                        no_auto_bright=True, no_auto_scale=True, use_camera_wb=True,
                        output_color=rawpy.ColorSpace.sRGB, user_flip=0)
                    result.native_metadata["rendering"] = {
                        "description": "LibRaw derived demosaiced linear RGB; sensor samples are separate",
                        "nominal_max": 65535, "derived": True, "gamma": [1.0, 1.0],
                        "no_auto_scale": True, "no_auto_bright": True}
                except Exception as exc:
                    result.warnings.append(f"LibRaw RGB rendering unavailable: {exc}")
        except Exception as exc:
            result.warnings.append(f"LibRaw does not support this file: {exc}")
    else:
        result.warnings.append("Install rawpy for exact sensor mosaic access; ImageIO RGB is a processed derivative.")
    _capture_dng_rasters(result)
    try:
        bridge_available = native_bridge() is not None
    except MediaWorkflowError as exc:
        bridge_available = False
        result.warnings.append(str(exc))
    result.native_metadata["vision_available"] = bridge_available
    if bridge_available:
        with tempfile.TemporaryDirectory(prefix="ipde-raw-inspect-") as directory:
            try:
                native = run_native("inspect", path, Path(directory))
                result.native_metadata["properties"] = native.get("properties", {})
                rendered = native.get("rendered")
                if rendered:
                    value = np.fromfile(rendered["path"], dtype=np.float32)
                    expected = int(rendered["height"]) * int(rendered["width"]) * int(rendered["channels"])
                    if value.size != expected:
                        raise MediaWorkflowError("Truncated native RAW RGB buffer")
                    result.rendered = value.reshape(rendered["height"], rendered["width"], rendered["channels"])[..., :3].copy()
                    result.native_metadata["rendering"] = {**rendered, "nominal_max": 1.0}
                    result.native_metadata["rendering"].pop("path", None)
                for auxiliary in native.get("auxiliary", []):
                    data = Path(auxiliary["path"]).read_bytes()
                    item = {key: value for key, value in auxiliary.items() if key != "path"}
                    item["data"] = data
                    item["array"] = _decode_auxiliary(data, auxiliary["description"])
                    result.auxiliary.append(item)
            except MediaWorkflowError as exc:
                result.warnings.append(f"Native RAW data access unavailable: {exc}")
    if result.sensor is None and result.rendered is None:
        result.warnings.append("No decoded RAW raster is available. Exact original-file capture remains available; rendered/depth/mask operations are unavailable.")
    return result


def inspect_raw(source: Path | str, preview_dir: Path | None = None) -> dict[str, Any]:
    capture = capture_raw(source)
    assets = []
    values = [("sensor", capture.sensor, False), ("sensor-visible", capture.visible_sensor, False),
              ("rendered-linear-rgb", capture.rendered, True)]
    for role, array, derived in values:
        if array is None:
            continue
        label = "stored-linear-raw" if role == "sensor" and capture.sensor_metadata.get("representation") == "stored demosaiced LinearRaw codes" else role
        item = {"index": len(assets), "kind": role, "name": label, "derived": derived,
                "representation": capture.sensor_metadata.get("representation") if role == "sensor" else None,
                "grid": "sensor-full" if role == "sensor" else "sensor-active" if role == "sensor-visible" else "rendered",
                **array_summary(array)}
        if preview_dir and role == "rendered-linear-rgb":
            item["preview_path"] = write_preview(preview_dir / "raw-preview.png", array,
                nominal_max=capture.native_metadata.get("rendering", {}).get("nominal_max"))
        assets.append(item)
    for auxiliary in capture.auxiliary:
        array = auxiliary["array"]
        item = {"index": len(assets), "kind": "auxiliary", "name": auxiliary["semantic"],
                "description": auxiliary["description"], "derived": False}
        if array is not None:
            item.update(array_summary(array))
            if preview_dir:
                item["preview_path"] = write_preview(preview_dir / f"aux-{item['index']}.png", array)
        else:
            item.update(dtype="opaque", shape=[], byte_count=len(auxiliary["data"]))
        assets.append(item)
    return {"ok": True, "operation": "raw-inspect", "source": str(capture.source),
            "source_sha256": sha256_file(capture.source), "assets": assets,
            "metadata": _jsonable({"sensor": capture.sensor_metadata, "native": capture.native_metadata}),
            "preview_path": next((a.get("preview_path") for a in assets if a["kind"] == "rendered-linear-rgb"), None),
            "capabilities": {"rewrite": False, "privacy": False, "sensor": capture.sensor is not None,
                             "person_mask": bool(capture.native_metadata.get("vision_available")) and capture.rendered is not None,
                             "rendered": capture.rendered is not None, "learned_depth": capture.rendered is not None,
                             "auxiliary": bool(capture.auxiliary)},
            "warnings": capture.warnings + ["Stored RAW samples and processed camera RGB use different grids and representations.",
                "AI depth/people masks are derived estimates on rendered RGB; they do not recover sensor or measured depth precision."]}


def validate_crop(crop: tuple[int, int, int, int] | None, shape: tuple[int, ...]) -> tuple[slice, slice]:
    if crop is None:
        return slice(None), slice(None)
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > shape[1] or y + height > shape[0]:
        raise MediaWorkflowError("Crop must be a nonempty rectangle inside the selected product's pixel grid")
    return slice(y, y + height), slice(x, x + width)


def _write_bytes(output_dir: Path, name: str, data: bytes, role: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / name
    descriptor, temporary_name = tempfile.mkstemp(prefix=".ipde-raw-", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise MediaWorkflowError("Byte-preserving RAW output verification failed")
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(final.resolve()), "role": role, "verified_bit_exact": True,
            "byte_count": len(data), "sha256": sha256_file(final)}


def export_raw(source: Path | str, output_dir: Path, operation: str, format: str = "npy",
               crop: tuple[int, int, int, int] | None = None,
               learned: LearnedDepthConfig | None = None, asset_index: int | None = None) -> dict[str, Any]:
    capture = capture_raw(source)
    name = capture.source.stem + "-" + uuid.uuid4().hex[:8]
    outputs = []
    metadata = {"source_sha256": sha256_file(capture.source), "sensor": capture.sensor_metadata,
                "native": capture.native_metadata}
    try:
        if operation == "original":
            if crop:
                raise MediaWorkflowError("Original RAW file capture cannot be cropped; export sensor/RGB for a selected region")
            outputs.append(_write_bytes(output_dir, name + capture.source.suffix, capture.source.read_bytes(), "original-raw"))
        elif operation == "sensor":
            if capture.sensor is None:
                raise MediaWorkflowError("Exact sensor raster is unavailable for this file; use the original RAW file capture")
            ys, xs = validate_crop(crop, capture.sensor.shape)
            outputs.append(export_array(output_dir, name + "-sensor", capture.sensor[ys, xs], format,
                                        role="sensor", metadata={"derived": False, "grid": "sensor-full", "crop": crop, "representation": capture.sensor_metadata.get("representation", "sensor mosaic")}))
            if crop is None and capture.visible_sensor is not None:
                outputs.append(export_array(output_dir, name + "-sensor-active", capture.visible_sensor, format,
                                            role="sensor-active", metadata={"derived": False, "grid": "sensor-active"}))
        elif operation == "rendered":
            if capture.rendered is None:
                raise MediaWorkflowError("No derived RGB render is available")
            ys, xs = validate_crop(crop, capture.rendered.shape)
            outputs.append(export_array(output_dir, name + "-rendered", capture.rendered[ys, xs], format,
                                        role="rendered-linear-rgb", metadata={"derived": True, "crop": crop,
                                        "rendering": capture.native_metadata.get("rendering")}))
        elif operation == "auxiliary":
            if crop:
                raise MediaWorkflowError("Auxiliary capture preserves each original grid; use an explicit asset export for a crop")
            if not capture.auxiliary:
                raise MediaWorkflowError("No native auxiliary data was reported in this RAW file")
            first_aux_index = sum(value is not None for value in (capture.sensor, capture.visible_sensor, capture.rendered))
            if asset_index is None:
                chosen = list(enumerate(capture.auxiliary))
            else:
                auxiliary_index = asset_index - first_aux_index
                if auxiliary_index < 0 or auxiliary_index >= len(capture.auxiliary):
                    raise MediaWorkflowError("Select an available RAW auxiliary asset")
                chosen = [(auxiliary_index, capture.auxiliary[auxiliary_index])]
            for index, auxiliary in chosen:
                if auxiliary["array"] is not None:
                    outputs.append(export_array(output_dir, f"{name}-aux-{index}-{auxiliary['semantic']}",
                                                auxiliary["array"], format, role=auxiliary["semantic"],
                                                metadata={"derived": False, "description": auxiliary["description"],
                                                          "xmp": auxiliary.get("xmp")}))
                else:
                    outputs.append(_write_bytes(output_dir, f"{name}-aux-{index}.bin", auxiliary["data"], auxiliary["semantic"]))
            metadata["auxiliary"] = [{key: value for key, value in a.items() if key not in ("data", "array")}
                                      for _, a in chosen]
        elif operation == "learned-depth":
            if capture.rendered is None:
                raise MediaWorkflowError("A rendered RGB reference is required for learned depth")
            config = learned or LearnedDepthConfig()
            config = replace(config, input_max_value=capture.native_metadata.get("rendering", {}).get("nominal_max", 1.0))
            rgb = capture.rendered
            # Camera RGB may legitimately exceed 1 or contain negative values.
            # Model input clipping is an explicit derived preprocessing copy,
            # never applied to stored RAW/sensor/linear-RGB arrays.
            nominal = config.input_max_value or 1.0
            rgb_for_model = np.clip(rgb, 0, nominal)
            result = infer_learned_depth(rgb_for_model, config, reference_label="rendered RAW RGB")
            ys, xs = validate_crop(crop, result.source_depth.shape)
            outputs.append(export_array(output_dir, name + "-learned-depth", result.source_depth[ys, xs], format,
                role="learned-depth", metadata={**result.metadata, "derived": True, "grid": "rendered", "crop": crop,
                                                "model_rgb_clipping": [0, nominal]}))
            outputs.append(export_array(output_dir, name + "-learned-native", result.native_depth, format,
                role="learned-native", metadata={**result.metadata, "derived": True, "grid": "model-native"}))
            # A matched region of RGB makes the portion immediately usable for
            # displacement without changing or duplicating the original RAW.
            outputs.append(export_array(output_dir, name + "-color-region", rgb[ys, xs], "npy",
                role="color-region", metadata={"derived": True, "grid": "rendered", "crop": crop}))
        elif operation == "person-mask":
            if capture.rendered is None:
                raise MediaWorkflowError("A rendered reference is required for person segmentation")
            with tempfile.TemporaryDirectory(prefix="ipde-vision-") as directory:
                mask_path = Path(directory) / "mask.bin"
                # Vision operates on an explicit model-input copy of the rendered
                # grid. This avoids assuming CoreImage RAW rendering and
                # CGImageSource RAW decode have identical crops/orientation.
                reference_path = Path(directory) / "vision-reference.png"
                from .formats import write_png
                reference = preview_array(capture.rendered,
                    nominal_max=capture.native_metadata.get("rendering", {}).get("nominal_max"),
                    size=max(capture.rendered.shape[:2]))
                write_png(reference_path, reference[..., :3])
                native = run_native("mask", reference_path, mask_path)
                if (native["reference_height"], native["reference_width"]) != capture.rendered.shape[:2]:
                    raise MediaWorkflowError("Vision and processed RAW grids differ; an unregistered mask is refused")
                native_mask = np.fromfile(mask_path, dtype=np.uint8).reshape(native["height"], native["width"])
                h, w = capture.rendered.shape[:2]
                mask = cv2.resize(native_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                ys, xs = validate_crop(crop, mask.shape)
                outputs.append(export_array(output_dir, name + "-person-mask", mask[ys, xs], format,
                    role="person-mask", metadata={**native, "derived": True, "grid": "rendered", "crop": crop,
                                                  "resampled": True, "model_reference": "explicit rendered-grid display RGB copy"}))
                rgb = capture.rendered[ys, xs]
                alpha = mask[ys, xs].astype(np.float32) / 255.0
                rgba = np.concatenate((rgb.astype(np.float32), alpha[..., None]), axis=2)
                outputs.append(export_array(output_dir, name + "-person-cutout", rgba,
                    "exr" if format == "exr" else "npy" if format == "png" else format,
                    role="person-cutout", metadata={"derived": True, "grid": "rendered", "alpha_mode": "straight"}))
        else:
            raise MediaWorkflowError(f"Unknown RAW operation {operation}")
        # RAW calibration is required to interpret the mosaic faithfully; it is
        # an explicit companion product, not an automatically generated manifest.
        outputs.append(_write_bytes(output_dir, name + "-calibration.json",
            json.dumps(_jsonable(metadata), indent=2, allow_nan=False).encode(), "calibration-metadata"))
    except Exception:
        for output in outputs:
            Path(output["path"]).unlink(missing_ok=True)
        raise
    return {"ok": True, "operation": operation, "source": str(capture.source), "outputs": outputs,
            "warnings": capture.warnings}
