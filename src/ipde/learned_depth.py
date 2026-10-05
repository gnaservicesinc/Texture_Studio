"""Explicit local, FP32 inference for high-detail learned depth estimates.

The RGB preprocessing here belongs to the neural model and operates on a copy.
It never accepts or modifies an extracted auxiliary depth/matte/gain-map plane.
Predictions retain their numerical values; native and source-grid products are
distinct, and a relative model is never relabelled as measured metric depth.
"""

from __future__ import annotations

import hashlib
import gc
import importlib
import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


class LearnedDepthError(RuntimeError):
    """An actionable local-model configuration or inference failure."""


@dataclass(frozen=True)
class LearnedDepthConfig:
    model: str = "depthpro"
    model_path: Path | None = None
    source_dir: Path | None = None
    device: str = "auto"
    input_size: int = 518
    # A nominal code-value range, never the observed image min/max. Defaults
    # to 255 for uint8, 65535 for uint16, and 1 for floating RGB input.
    input_max_value: float | None = None


@dataclass
class LearnedDepthResult:
    native_depth: np.ndarray
    source_depth: np.ndarray
    metadata: dict[str, Any]
    confidence: np.ndarray | None = None


_MODELS = {
    "depthpro": ("apple/DepthPro", "depth_pro.pt", "ml-depth-pro", "depth_pro.depth_pro"),
    "depth-anything-v2": (
        "depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth",
        "Depth-Anything-V2", "depth_anything_v2.dpt",
    ),
    "depth-anything-v2-small": (
        "depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth",
        "Depth-Anything-V2", "depth_anything_v2.dpt",
    ),
    "depth-anything-3": (
        "depth-anything/DA3-GIANT-1.1", "DA3-GIANT-1.1",
        "Depth-Anything-3", "depth_anything_3.cfg",
    ),
}
_ENV_NAMES = {
    "depthpro": ("IPDE_DEPTHPRO_MODEL", "IPDE_DEPTHPRO_DIR"),
    "depth-anything-v2": ("IPDE_DEPTH_ANYTHING_V2_MODEL", "IPDE_DEPTH_ANYTHING_V2_DIR"),
    "depth-anything-v2-small": ("IPDE_DEPTH_ANYTHING_V2_SMALL_MODEL", "IPDE_DEPTH_ANYTHING_V2_DIR"),
    "depth-anything-3": ("IPDE_DA3_MODEL_DIR", "IPDE_DA3_DIR"),
}
_IMPORT_LOCK = threading.RLock()
# DA3's global FP32 attention grows quadratically with the number of patches.
# Bound the supported single-view workload before allocating weights/tensors.
# This is an admission limit, not a guarantee that every device has enough RAM.
DA3_MAX_PATCH_TOKENS = 8192


def validate_learned_depth_input(shape: Any, config: LearnedDepthConfig) -> tuple[int, int] | None:
    """Reject unsupported DA3 workloads without decoding RGB or loading a model.

    Match the upstream longest-side resize and nearest-14 rounding, including
    its upward tie break. Never silently reduce a requested processing grid.
    """
    if config.model != "depth-anything-3":
        return None
    if len(shape) != 3 or shape[2] != 3 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0
        for value in shape
    ):
        raise LearnedDepthError("DA3 requires a nonempty HxWx3 RGB reference image")
    if isinstance(config.input_size, bool) or not isinstance(config.input_size, int) or (
        config.input_size != 0 and not 14 <= config.input_size <= 8192
    ):
        raise LearnedDepthError("model input_size must be 0 (native source size) or an integer in [14, 8192]")
    height, width = map(int, shape[:2])
    requested = config.input_size or max(height, width)
    scale = requested / max(height, width)

    def patch_dimension(value: int) -> int:
        resized = max(1, int(round(value * scale)))
        down = resized // 14 * 14
        rounded = down + 14 if resized - down >= 7 else down
        return max(1, rounded)

    processed = patch_dimension(height), patch_dimension(width)
    if min(processed) < 14:
        raise LearnedDepthError("DA3 processing grid is too narrow for a 14-pixel patch; choose a less extreme image aspect ratio")
    tokens = (processed[0] // 14) * (processed[1] // 14)
    if tokens > DA3_MAX_PATCH_TOKENS:
        raise LearnedDepthError(
            f"DA3 FP32 processing would use {processed[1]} x {processed[0]} pixels "
            f"({tokens:,} transformer patches), exceeding the supported {DA3_MAX_PATCH_TOKENS:,}-patch limit. "
            "Native photo resolution can exhaust memory and stall global attention. "
            "Set teacher input_size to 1036 or smaller. The original RGB remains untouched; "
            "the full photo-size float depth is a separately recorded interpolation of the model output."
        )
    return processed


@contextmanager
def _fp32_math(torch: Any, device: str):
    # CUDA TF32 keeps FP32 storage but truncates multiplicands. Preserve the
    # caller's settings while enforcing full mantissas for this inference.
    previous = None
    if device == "cuda":
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type=device, enabled=False):
            yield
    finally:
        if previous is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
            check=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _local_roots() -> list[Path]:
    roots = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    return list(dict.fromkeys(roots))


def resolve_learned_depth_resources(config: LearnedDepthConfig) -> tuple[Path, Path | None]:
    """Resolve only explicit paths, environment paths, or conventional local files."""
    if config.model not in _MODELS:
        raise LearnedDepthError(f"unknown depth model {config.model!r}; choose {', '.join(_MODELS)}")
    model_id, filename, source_name, _ = _MODELS[config.model]
    model_env, source_env = _ENV_NAMES[config.model]
    explicit_model = config.model_path or os.environ.get(model_env)
    if explicit_model:
        model_path = Path(explicit_model).expanduser().resolve()
    else:
        candidates = [root / "models" / filename for root in _local_roots()]
        model_path = next((path for path in candidates if path.exists()), candidates[0])
    is_directory = config.model == "depth-anything-3"
    if not (model_path.is_dir() if is_directory else model_path.is_file()):
        raise LearnedDepthError(
            f"local {config.model} {'model directory' if is_directory else 'checkpoint'} is missing: "
            f"{model_path}. Download {model_id} explicitly with hf download, then set model_path "
            f"or {model_env}. Inference never downloads model files."
        )
    if is_directory:
        for required in ("config.json", "model.safetensors"):
            if not (model_path / required).is_file():
                raise LearnedDepthError(f"DA3 model directory must contain {required}: {model_path}")
    explicit_source = config.source_dir or os.environ.get(source_env)
    if explicit_source:
        source = Path(explicit_source).expanduser().resolve()
        if not source.is_dir():
            raise LearnedDepthError(f"local model source directory does not exist: {source}")
    else:
        source = next((root / source_name for root in _local_roots() if (root / source_name).is_dir()), None)
    if config.device not in {"auto", "cpu", "mps", "cuda"}:
        raise LearnedDepthError(f"unsupported inference device {config.device!r}")
    if isinstance(config.input_size, bool) or not isinstance(config.input_size, int) or (config.input_size != 0 and not 14 <= config.input_size <= 8192):
        raise LearnedDepthError("model input_size must be 0 (native source size) or an integer in [14, 8192]")
    return model_path.resolve(), source


def _import_model_source(config: LearnedDepthConfig, source: Path | None) -> Any:
    module_name = _MODELS[config.model][3]
    package_name = module_name.split(".")[0]
    search_path = None
    if source is not None:
        search_path = source / "src" if (source / "src" / package_name).is_dir() else source
        if not (search_path / package_name).is_dir():
            raise LearnedDepthError(f"{source} does not contain the {package_name} model package")
    # Python retains imported packages. Avoid silently using a different source
    # tree after the user changes the configured directory within one process.
    with _IMPORT_LOCK:
        existing = sys.modules.get(package_name)
        if search_path is not None and existing is not None and getattr(existing, "__file__", None):
            if not Path(existing.__file__).resolve().is_relative_to(search_path):
                raise LearnedDepthError(f"{package_name} is already imported from a different directory; restart IPDE to change its source")
        inserted = search_path is not None and str(search_path) not in sys.path
        if inserted:
            sys.path.insert(0, str(search_path))
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise LearnedDepthError(
                f"could not import {config.model} source: {exc}. Configure source_dir/{_ENV_NAMES[config.model][1]} "
                "and install requirements-depth.txt (DA3 also needs its upstream dependencies)."
            ) from exc
        finally:
            if inserted:
                sys.path.remove(str(search_path))
        if search_path is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(search_path):
                raise LearnedDepthError(f"{module_name} is already imported from a different directory; restart IPDE to change its source")
        return module


def _select_device(torch: Any, requested: str) -> str:
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise LearnedDepthError("CUDA was requested but is unavailable")
    if requested == "mps" and not mps:
        raise LearnedDepthError("Apple Metal (MPS) was requested but is unavailable")
    return requested


def _validate_rgb(rgb: np.ndarray, config: LearnedDepthConfig) -> tuple[np.ndarray, float]:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise LearnedDepthError("learned depth requires a nonempty HxWx3 RGB reference image")
    if array.dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"), np.dtype("float64")):
        raise LearnedDepthError(f"unsupported reference RGB dtype {array.dtype}; use uint8, uint16, or floating RGB")
    limit = config.input_max_value
    if limit is None:
        limit = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else 1.0
    if not math.isfinite(limit) or limit <= 0:
        raise LearnedDepthError("input_max_value must be a finite positive nominal RGB code-value limit")
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > limit):
        raise LearnedDepthError("reference RGB values must be finite and within their nominal code-value range")
    return array, limit


def _prepare_rgb(rgb: np.ndarray, config: LearnedDepthConfig) -> tuple[np.ndarray, float]:
    array, limit = _validate_rgb(rgb, config)
    return np.ascontiguousarray(array, dtype=np.float32) / np.float32(limit), limit


def _checkpoint_state(torch: Any, model_path: Path) -> tuple[Mapping[str, Any], dict[str, int]]:
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise LearnedDepthError("checkpoint must contain a model state dictionary")
    counts: dict[str, int] = {}
    for value in state.values():
        if hasattr(value, "dtype"):
            name = str(value.dtype).removeprefix("torch.")
            counts[name] = counts.get(name, 0) + value.numel()
    return state, counts


def _load_safetensors_model(model: Any, checkpoint: Path) -> dict[str, int]:
    from safetensors import safe_open
    from safetensors.torch import load_model
    dtype_names = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16"}
    counts: dict[str, int] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            name = dtype_names.get(tensor_slice.get_dtype(), tensor_slice.get_dtype())
            counts[name] = counts.get(name, 0) + math.prod(tensor_slice.get_shape())
    # DA3's auxiliary LayerNorm is shared by four heads. Safetensors stores
    # each shared buffer once: load_model handles those aliases strictly while
    # rejecting every genuinely missing or unexpected parameter.
    load_model(model, checkpoint, strict=True, device="cpu")
    return counts


class LearnedDepthPredictor:
    """A reusable local model; repeated dataset samples reuse the loaded weights."""

    def __init__(self, config: LearnedDepthConfig):
        self.config = config
        path, source = resolve_learned_depth_resources(config)
        try:
            import torch
        except ImportError as exc:
            raise LearnedDepthError("PyTorch is required; install requirements-depth.txt") from exc
        self.torch = torch
        self.device = _select_device(torch, config.device)
        module = _import_model_source(config, source)
        self.module = module
        checkpoint = path / "model.safetensors" if config.model == "depth-anything-3" else path
        self.base_metadata: dict[str, Any] = {
            "model": config.model, "model_id": _MODELS[config.model][0],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": _file_sha256(checkpoint),
            "model_source_directory": str(source) if source else None,
            "model_source_revision": _git_revision(source) if source else None,
            "device": self.device, "computation_dtype": "float32", "autocast": False,
            "torch_version": str(torch.__version__), "estimated_not_measured": True,
        }
        try:
            if config.model == "depth-anything-3":
                model_configuration = json.loads((path / "config.json").read_text())
                name = model_configuration.get("model_name")
                if name not in {"da3-giant", "da3-large", "da3-base", "da3-small"}:
                    raise LearnedDepthError("this DA3 adapter supports relative DA3 checkpoints only; metric/nested models require their own scaling contract")
                network_config = model_configuration.get("config")
                if not isinstance(network_config, Mapping):
                    raise LearnedDepthError("DA3 config.json must contain its serialized network config")
                # Load exactly the architecture bundled with the checkpoint,
                # avoiding api.py's eager imports of unrelated export packages.
                # Keep the standard 'model.' state-dictionary prefix intact.
                class LocalDA3(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.model = module.create_object(network_config)
                        processor_module = importlib.import_module("depth_anything_3.utils.io.input_processor")
                        self.input_processor = processor_module.InputProcessor()
                self.model = LocalDA3()
                dtype_counts = _load_safetensors_model(self.model, checkpoint)
                self.base_metadata["model_config_sha256"] = _file_sha256(path / "config.json")
            else:
                state, dtype_counts = _checkpoint_state(torch, path)
                if config.model == "depthpro":
                    model_configuration = replace(module.DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=None)
                    self.model, _ = module.create_model_and_transforms(
                        config=model_configuration, device=torch.device("cpu"), precision=torch.float32,
                    )
                else:
                    model_args = (
                        {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
                        if config.model == "depth-anything-v2-small" else
                        {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
                    )
                    self.model = module.DepthAnythingV2(**model_args)
                self.model.load_state_dict(state, strict=True)
                del state
            self.model.to(device=self.device, dtype=torch.float32).eval()
            self.base_metadata["checkpoint_storage_dtypes"] = dtype_counts
            self.base_metadata["checkpoint_precision_note"] = (
                "FP16/BF16 stored weights were promoted to FP32; their lost precision is not recoverable"
                if any(name in dtype_counts for name in ("float16", "bfloat16")) else
                "stored checkpoint tensors loaded without reduced-precision conversion"
            )
        except LearnedDepthError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise LearnedDepthError(f"could not load local {config.model} checkpoint {checkpoint}: {exc}") from exc

    def close(self) -> None:
        """Release model weights before another teacher is loaded."""
        if getattr(self, "model", None) is None:
            return
        self.model = None
        gc.collect()
        _release_accelerator_cache(self.torch, self.device)

    def __call__(self, rgb: np.ndarray, *, focal_pixels: float | None = None, reference_label: str | None = None) -> LearnedDepthResult:
        if self.model is None:
            raise LearnedDepthError("This depth predictor has been closed; load a new predictor before inference")
        validate_learned_depth_input(np.shape(rgb), self.config)
        # DA3 consumes the uint8 input directly. Avoid an unused full-resolution
        # float RGB copy (~280 MiB for an iPhone display photo) and division.
        if self.config.model == "depth-anything-3":
            normalized, nominal_limit = _validate_rgb(rgb, self.config)
        else:
            normalized, nominal_limit = _prepare_rgb(rgb, self.config)
        if focal_pixels is not None and (not math.isfinite(focal_pixels) or focal_pixels <= 0):
            raise LearnedDepthError("focal_pixels must be finite and positive in the supplied reference image grid")
        metadata = dict(self.base_metadata)
        metadata.update({
            "input_rgb_sha256": _array_sha256(np.asarray(rgb)), "input_rgb_shape": list(np.shape(rgb)),
            "input_rgb_dtype": str(np.asarray(rgb).dtype), "input_nominal_max_value": nominal_limit,
            "reference_label": reference_label, "input_focal_pixels": focal_pixels,
            "rgb_preprocessing_scope": "model input copy only; extracted auxiliary arrays remain untouched",
            "output_normalization": "none", "output_dtype": "float32",
        })
        torch = self.torch
        try:
            with torch.inference_mode(), _fp32_math(torch, self.device):
                if self.config.model == "depthpro":
                    native, source, details = self._depthpro(normalized, focal_pixels)
                    confidence = None
                elif self.config.model in {"depth-anything-v2", "depth-anything-v2-small"}:
                    native, source, details = self._depth_anything_v2(normalized)
                    confidence = None
                else:
                    native, source, details, confidence = self._depth_anything_3(np.asarray(rgb), nominal_limit)
            metadata.update(details)
            native_array = _plane(torch, native)
            source_array = _plane(torch, source)
            if source_array.shape != np.shape(rgb)[:2]:
                raise LearnedDepthError("source-grid depth shape does not match the supplied RGB reference")
            if not np.isfinite(native_array).any() or not np.isfinite(source_array).any():
                raise LearnedDepthError("model produced no finite depth estimates")
            metadata.update({"native_prediction_shape": list(native_array.shape), "source_prediction_shape": list(source_array.shape)})
            return LearnedDepthResult(native_array, source_array, metadata, confidence)
        except LearnedDepthError:
            raise
        except Exception as exc:
            raise LearnedDepthError(
                f"{self.config.model} FP32 inference failed on {self.device}: {exc}. "
                "Try device=cpu or a smaller Depth Anything input_size if device memory is insufficient."
            ) from exc

    def _depthpro(self, rgb: np.ndarray, focal_pixels: float | None):
        torch = self.torch
        height, width = rgb.shape[:2]
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x - 0.5) / 0.5
        native_size = self.model.img_size
        x = torch.nn.functional.interpolate(x, (native_size, native_size), mode="bilinear", align_corners=False)
        canonical_inverse, fov = self.model.forward(x)
        if focal_pixels is None:
            if fov is None:
                raise LearnedDepthError("DepthPro returned no focal estimate; supply calibrated focal_pixels")
            focal = 0.5 * width / torch.tan(0.5 * torch.deg2rad(fov.float()))
            focal_source = "DepthPro field-of-view estimate"
        else:
            focal = torch.tensor(focal_pixels, dtype=torch.float32, device=self.device)
            focal_source = "supplied reference-grid calibration"
        if not bool(torch.isfinite(focal).all()) or not bool((focal > 0).all()):
            raise LearnedDepthError("DepthPro focal estimate is nonfinite or nonpositive")
        inverse = canonical_inverse * (width / focal)
        # Match Apple's infer API: resize inverse depth first, then apply its
        # reciprocal guard. Interpolating metric depth gives a different answer.
        source_inverse = torch.nn.functional.interpolate(inverse, (height, width), mode="bilinear", align_corners=False)
        native = 1.0 / torch.clamp(inverse, min=1e-4, max=1e4)
        source = 1.0 / torch.clamp(source_inverse, min=1e-4, max=1e4)
        return native, source, {
            "units": "meters", "quantity": "estimated_camera_z_depth", "metric_scale": "model estimate with focal scaling",
            "focal_pixels": float(focal.squeeze().cpu()), "focal_source": focal_source,
            "processing": "RGB / nominal limit, then (x-0.5)/0.5; bilinear square model input; FP32 forward",
            "source_grid_resampling": "bilinear inverse depth, align_corners=False, then reciprocal; no sharpening",
            "model_inverse_depth_guard": {"minimum": 1e-4, "maximum": 1e4, "source": "Apple DepthPro infer API"},
            "native_grid": "square network prediction covering full reference extent; source image is stretched for model input",
            "model_input_shape": [native_size, native_size], "input_size": native_size,
            "requested_input_size": self.config.input_size, "input_size_policy": "fixed DepthPro network; configurable input size does not apply",
        }

    def _depth_anything_v2(self, rgb: np.ndarray):
        torch = self.torch
        module = self.module
        input_size = self.config.input_size or min(rgb.shape[:2])
        sample = {"image": rgb.copy()}
        sample = module.Resize(
            width=input_size, height=input_size, resize_target=False,
            keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound",
            image_interpolation_method=2,  # OpenCV INTER_CUBIC used by upstream.
        )(sample)
        sample = module.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
        sample = module.PrepareForNet()(sample)
        tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        native = self.model(tensor)[:, None]
        source = torch.nn.functional.interpolate(native, rgb.shape[:2], mode="bilinear", align_corners=True)
        return native, source, {
            "units": "relative_inverse_depth", "quantity": "relative_inverse_depth", "metric_scale": "unavailable",
            "processing": "RGB / nominal limit; upstream aspect-preserving cubic Resize and ImageNet normalization on copy",
            "source_grid_resampling": "bilinear relative inverse depth, align_corners=True; no sharpening",
            "model_input_shape": list(tensor.shape[-2:]), "input_size": input_size,
            "requested_input_size": self.config.input_size, "input_size_policy": "native source shortest side" if not self.config.input_size else "configured shortest side",
            "native_grid": "network output at processed RGB grid covering the full reference extent",
        }

    def _depth_anything_3(self, rgb: np.ndarray, nominal_limit: float):
        if rgb.dtype != np.uint8 or nominal_limit != 255:
            raise LearnedDepthError("DA3 upstream RGB processor supports uint8 RGB only; choose DepthPro/V2 to avoid reducing higher-bit RGB")
        torch = self.torch
        expected_shape = validate_learned_depth_input(rgb.shape, self.config)
        input_size = self.config.input_size or max(rgb.shape[:2])
        imgs, _, _ = self.model.input_processor(
            [rgb.copy()], process_res=input_size,
            process_res_method="upper_bound_resize", sequential=True,
        )
        # Check the real processor result too, in case configured local source
        # code changes its resize policy. Never dispatch oversized attention.
        actual_shape = tuple(imgs.shape[-2:])
        tokens = math.prod(dimension // 14 for dimension in actual_shape)
        if tokens > DA3_MAX_PATCH_TOKENS:
            raise LearnedDepthError("DA3 processor exceeded the supported patch budget; choose input_size=1036 or smaller")
        tensor = imgs[None].to(device=self.device, dtype=torch.float32)
        # The public DA3 wrapper enables FP16/BF16 autocast unconditionally.
        # Call its underlying network to honor IPDE's FP32 inference contract.
        output = self.model.model(tensor, None, None, [], False, False, "first")
        plane = _plane(torch, output["depth"])
        native = torch.from_numpy(plane)[None, None]
        source = torch.nn.functional.interpolate(native, rgb.shape[:2], mode="bilinear", align_corners=False)
        confidence = _plane(torch, output["depth_conf"]) if "depth_conf" in output else None
        return native, source, {
            "units": "relative_depth", "quantity": "relative_camera_z_depth", "metric_scale": "unavailable",
            "processing": "upstream uint8 RGB aspect-preserving upper_bound_resize and ImageNet normalization; underlying network FP32",
            "source_grid_resampling": "bilinear relative depth, align_corners=False; no sharpening",
            "model_input_shape": list(tensor.shape[-2:]), "input_size": input_size,
            "requested_input_size": self.config.input_size, "input_size_policy": "native source longest side" if not self.config.input_size else "configured longest side",
            "native_grid": "network output at processed RGB grid covering the full reference extent",
            "confidence_semantics": "unmodified DA3 depth_conf scores; not calibrated probabilities",
            "processing_patch_tokens": tokens, "processing_patch_limit": DA3_MAX_PATCH_TOKENS,
            "expected_model_input_shape": list(expected_shape),
        }, confidence


def _plane(torch: Any, value: Any) -> np.ndarray:
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
    # Remove only batch/view/channel singleton axes, preserving a spatial
    # dimension of length one in the supplied reference image.
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not array.size:
        raise LearnedDepthError(f"model must return a single nonempty HxW plane, received {array.shape}")
    return np.array(array, dtype=np.float32, order="C", copy=True)


@lru_cache(maxsize=1)
def _cached_predictor(config: LearnedDepthConfig, file_fingerprint: tuple[int, ...]) -> LearnedDepthPredictor:
    return LearnedDepthPredictor(config)


def _release_accelerator_cache(torch: Any, device: str | None = None) -> None:
    for name in ("cuda", "mps"):
        if device is not None and not str(device).startswith(name):
            continue
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            backend.synchronize()
            backend.empty_cache()


def release_learned_depth_cache() -> None:
    """Clear any previous one-off inference model before a batch model phase."""
    _cached_predictor.cache_clear()
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        _release_accelerator_cache(torch)


def infer_learned_depth(rgb: np.ndarray, config: LearnedDepthConfig | None = None, *, focal_pixels: float | None = None, reference_label: str | None = None) -> LearnedDepthResult:
    """Infer locally; preserve native values and separately identify source resampling."""
    configuration = config or LearnedDepthConfig()
    # Validate RGB and configuration before an expensive model load.
    validate_learned_depth_input(np.shape(rgb), configuration)
    _validate_rgb(rgb, configuration)
    path, source = resolve_learned_depth_resources(configuration)
    configuration = replace(configuration, model_path=path, source_dir=source)
    checkpoint = path / "model.safetensors" if configuration.model == "depth-anything-3" else path
    stat = checkpoint.stat()
    fingerprint = (stat.st_size, stat.st_mtime_ns)
    if configuration.model == "depth-anything-3":
        config_stat = (path / "config.json").stat()
        fingerprint += (config_stat.st_size, config_stat.st_mtime_ns)
    predictor = _cached_predictor(configuration, fingerprint)
    return predictor(rgb, focal_pixels=focal_pixels, reference_label=reference_label)
