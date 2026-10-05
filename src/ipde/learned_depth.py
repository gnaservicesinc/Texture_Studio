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
from types import MethodType
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
    input_size: int = 0
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
# The supported grid includes a full 5712 x 4284 iPhone display photograph.
# Global attention below is evaluated in query slices, retaining every key and
# value. No windowed attention, image tiles or depth-map stitching is involved.
DA3_MAX_PATCH_TOKENS = 131072
DA3_ATTENTION_CHUNK_THRESHOLD = 8192
DA3_ATTENTION_SCORE_BYTES = 2 * 1024**3
DA3_DECODER_FEATURE_ELEMENTS = 128 * 1024**2
DA3_DECODER_ROWS = 128


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
            "Choose a smaller explicit input_size or a smaller photograph. "
            "IPDE never silently reduces the requested model grid."
        )
    return processed


def _da3_attention_forward(attention: Any, x: Any, pos: Any = None, attn_mask: Any = None):
    """Upstream FP32 global attention with bounded query working memory.

    Each query still attends to every key in the original image. Slicing only
    independent query rows preserves softmax's complete key axis; it does not
    change the model's receptive field or introduce tile boundaries.
    """
    import torch
    batch, tokens, channels = x.shape
    heads = attention.num_heads
    qkv = attention.qkv(x).reshape(batch, tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = attention.q_norm(q) if hasattr(attention, "q_norm") else q
    k = attention.k_norm(k) if hasattr(attention, "k_norm") else k
    if getattr(attention, "rope", None) is not None and pos is not None:
        q, k = attention.rope(q, pos), attention.rope(k, pos)
    rows = max(1, min(tokens, DA3_ATTENTION_SCORE_BYTES // (batch * heads * tokens * x.element_size())))
    result = torch.empty_like(q)
    # Inference alone uses this adapter: dropout never affects the predictions.
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        mask = attn_mask[:, None, start:stop, :] if attn_mask is not None else None
        result[:, :, start:stop] = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, start:stop], k, v, dropout_p=0.0, attn_mask=mask,
        )
    result = result.transpose(1, 2).reshape(batch, tokens, channels)
    return attention.proj_drop(attention.proj(result))


def _install_bounded_attention(backbone: Any, *, da3: bool) -> int:
    """Adapt only this local model instance; imported upstream classes stay intact."""
    installed = 0
    module_name = "depth_anything_3.model.dinov2.layers.attention" if da3 else "depth_anything_v2.dinov2_layers.attention"
    for attention in backbone.modules():
        if attention.__class__.__module__ != module_name:
            continue
        original = attention.forward

        if da3:
            def forward(instance, x, pos=None, attn_mask=None, _original=original):
                if x.shape[1] <= DA3_ATTENTION_CHUNK_THRESHOLD:
                    return _original(x, pos=pos, attn_mask=attn_mask)
                return _da3_attention_forward(instance, x, pos, attn_mask)
        else:
            def forward(instance, x, attn_bias=None, _original=original):
                if x.shape[1] <= DA3_ATTENTION_CHUNK_THRESHOLD:
                    return _original(x) if attn_bias is None else _original(x, attn_bias=attn_bias)
                if attn_bias is not None:
                    raise LearnedDepthError("Full-resolution DA2 expects one image without a packed attention bias")
                return _da3_attention_forward(instance, x)

        attention.forward = MethodType(forward, attention)
        installed += 1
    return installed


def _da3_depth_only(network: Any, tensor: Any) -> dict[str, Any]:
    """Run the unchanged primary DualDPT branch without ray/camera/GS heads.

    DA3's primary and ray decoder branches are independent after the shared
    projected pyramid. Calling the primary branch alone avoids multiple unused
    full-image auxiliary tensors while retaining its original arithmetic.
    """
    from depth_anything_3.model.utils.head_utils import custom_interpolate
    head = network.head
    if head.__class__.__name__ != "DualDPT" or head.__class__.__module__ != "depth_anything_3.model.dualdpt":
        # Preserve the contract of other explicitly configured relative DA3
        # architectures; the built-in GIANT checkpoint uses DualDPT below.
        return network(tensor, None, None, [], False, False, "first")
    height, width = tensor.shape[-2:]
    feats, _ = network.backbone(tensor, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    batch, views, count, channels = feats[0][0].shape
    patch_h, patch_w = height // head.patch_size, width // head.patch_size
    resized = []
    for stage, take in enumerate(head.intermediate_layer_idx):
        value = head.norm(feats[take][0].reshape(batch * views, count, channels))
        value = value.permute(0, 2, 1).reshape(batch * views, channels, patch_h, patch_w)
        value = head.projects[stage](value)
        if head.pos_embed:
            value = head._add_pos_embed(value, width, height)
        resized.append(head.resize_layers[stage](value))
    del feats
    scratch = head.scratch
    l1, l2, l3, l4 = (getattr(scratch, f"layer{stage}_rn")(value) for stage, value in enumerate(resized, 1))
    del resized
    value = scratch.refinenet4(l4, size=l3.shape[2:])
    del l4
    value = scratch.refinenet3(value, l3, size=l2.shape[2:])
    del l3
    value = scratch.refinenet2(value, l2, size=l1.shape[2:])
    del l2
    value = scratch.refinenet1(value, l1)
    del l1
    value = scratch.output_conv1(value)
    output_shape = (int(height / head.down_ratio), int(width / head.down_ratio))
    if math.prod(output_shape) * value.shape[0] * value.shape[1] > DA3_DECODER_FEATURE_ELEMENTS:
        logits = _da3_primary_logits_rows(head, value, output_shape, width / height)
    else:
        value = custom_interpolate(value, output_shape, mode="bilinear", align_corners=True)
        if head.pos_embed:
            value = head._add_pos_embed(value, width, height)
        logits = scratch.output_conv2(value)
    logits = logits.permute(0, 2, 3, 1)
    depth = head._apply_activation_single(logits[..., :-1], head.activation).squeeze(-1)
    confidence = head._apply_activation_single(logits[..., -1], head.conf_activation)
    return {"depth": depth.reshape(batch, views, *depth.shape[1:]),
            "depth_conf": confidence.reshape(batch, views, *confidence.shape[1:])}


def _bilinear_rows(value: Any, start: int, stop: int, shape: tuple[int, int]):
    """Original align_corners=True bilinear coordinates for selected output rows."""
    import torch
    height, width = shape
    source_height = value.shape[-2]
    scale = (source_height - 1) / (height - 1) if height > 1 else 0.0
    coordinates = torch.arange(start, stop, device=value.device, dtype=value.dtype) * scale
    lower = coordinates.to(torch.long)
    upper = (lower + 1).clamp(max=source_height - 1)
    fraction = (coordinates - lower).reshape(1, 1, -1, 1)
    # Horizontal resizing retains the full original width and boundary. Resize
    # independent source rows before the unchanged vertical linear combination.
    low = torch.nn.functional.interpolate(value.index_select(-2, lower), (stop - start, width), mode="bilinear", align_corners=True)
    high = torch.nn.functional.interpolate(value.index_select(-2, upper), (stop - start, width), mode="bilinear", align_corners=True)
    return (1 - fraction) * low + fraction * high


def _da3_primary_logits_rows(head: Any, value: Any, shape: tuple[int, int], aspect_ratio: float):
    """Bound primary decoder memory without changing its full-image coordinates.

    The head's only spatial output convolution is 3x3. A one-row halo includes
    its complete receptive field at each internal stripe edge. These are decoder
    feature evaluations, not independently inferred image tiles or depth blends.
    """
    import torch
    height, width = shape
    channels = next(layer.out_channels for layer in reversed(head.scratch.output_conv2)
                    if hasattr(layer, "out_channels"))
    logits = torch.empty((value.shape[0], channels, height, width), dtype=value.dtype, device=value.device)
    positional = getattr(head, "pos_embed", False)
    if positional:
        from depth_anything_3.model.utils.head_utils import position_grid_to_embed
        diag = math.sqrt(aspect_ratio**2 + 1)
        span_x, span_y = aspect_ratio / diag, 1 / diag
        xcoords = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width,
                                width, device=value.device, dtype=value.dtype)
        ycoords = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height,
                                height, device=value.device, dtype=value.dtype)
    for start in range(0, height, DA3_DECODER_ROWS):
        stop = min(start + DA3_DECODER_ROWS, height)
        begin, end = max(0, start - 1), min(height, stop + 1)
        stripe = _bilinear_rows(value, begin, end, shape)
        if positional:
            yy, xx = torch.meshgrid(ycoords[begin:end], xcoords, indexing="ij")
            uv = torch.stack((xx, yy), dim=-1)
            embedding = position_grid_to_embed(uv, value.shape[1]) * 0.1
            stripe = stripe + embedding.permute(2, 0, 1)[None]
        output = head.scratch.output_conv2(stripe)
        logits[:, :, start:stop] = output[:, :, start - begin:stop - begin]
    return logits


def _da2_head_forward(head: Any, features: Any, patch_h: int, patch_w: int):
    """Native DA2 decoder with its original pyramid and bounded output features."""
    import torch
    resized = []
    for stage, feature in enumerate(features):
        value = feature[0]
        if head.use_clstoken:
            token = feature[1].unsqueeze(1).expand_as(value)
            value = head.readout_projects[stage](torch.cat((value, token), -1))
        value = value.permute(0, 2, 1).reshape(value.shape[0], value.shape[-1], patch_h, patch_w)
        resized.append(head.resize_layers[stage](head.projects[stage](value)))
    scratch = head.scratch
    l1, l2, l3, l4 = (getattr(scratch, f"layer{stage}_rn")(value) for stage, value in enumerate(resized, 1))
    del resized
    value = scratch.refinenet4(l4, size=l3.shape[2:])
    del l4
    value = scratch.refinenet3(value, l3, size=l2.shape[2:])
    del l3
    value = scratch.refinenet2(value, l2, size=l1.shape[2:])
    del l2
    value = scratch.refinenet1(value, l1)
    del l1
    value = scratch.output_conv1(value)
    shape = (patch_h * 14, patch_w * 14)
    return _da3_primary_logits_rows(head, value, shape, shape[1] / shape[0])


def _install_da2_bounded_decoder(head: Any) -> None:
    original = head.forward

    def forward(instance, features, patch_h, patch_w):
        channels = instance.scratch.output_conv1.out_channels
        if patch_h * patch_w * 14**2 * channels <= DA3_DECODER_FEATURE_ELEMENTS:
            return original(features, patch_h, patch_w)
        return _da2_head_forward(instance, features, patch_h, patch_w)

    head.forward = MethodType(forward, head)


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
                installed = _install_bounded_attention(self.model.model.backbone, da3=True)
                if not installed:
                    raise LearnedDepthError("DA3 source has no supported attention layers; cannot provide bounded full-resolution inference")
                self.base_metadata["bounded_global_attention_layers"] = installed
                self.base_metadata["unused_branches"] = ["ray", "camera_pose", "gaussian_splats"]
                self._da3_primary_only = True
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
                if config.model in {"depth-anything-v2", "depth-anything-v2-small"}:
                    installed = _install_bounded_attention(self.model.pretrained, da3=False)
                    if not installed:
                        raise LearnedDepthError("DA2 source has no supported attention layers; cannot provide bounded full-resolution inference")
                    self.base_metadata["bounded_global_attention_layers"] = installed
                    _install_da2_bounded_decoder(self.model.depth_head)
                    self.base_metadata["decoder_policy"] = "full-grid bilinear decoder with bounded row evaluation and complete convolution halo"
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
        tokens = math.prod(dimension // 14 for dimension in tensor.shape[-2:])
        if tokens > DA3_MAX_PATCH_TOKENS:
            raise LearnedDepthError("DA2 processor exceeded the supported full-resolution patch budget; choose a smaller explicit input_size")
        native = self.model(tensor)[:, None]
        source = (native if tuple(native.shape[-2:]) == rgb.shape[:2] else
                  torch.nn.functional.interpolate(native, rgb.shape[:2], mode="bilinear", align_corners=True))
        return native, source, {
            "units": "relative_inverse_depth", "quantity": "relative_inverse_depth", "metric_scale": "unavailable",
            "processing": "RGB / nominal limit; upstream aspect-preserving cubic Resize and ImageNet normalization on copy",
            "source_grid_resampling": "none; prediction already matches the full reference grid" if tuple(native.shape[-2:]) == rgb.shape[:2] else "bilinear relative inverse depth, align_corners=True; no sharpening",
            "model_input_shape": list(tensor.shape[-2:]), "input_size": input_size,
            "requested_input_size": self.config.input_size, "input_size_policy": "native source shortest side" if not self.config.input_size else "configured shortest side",
            "native_grid": "network output at processed RGB grid covering the full reference extent",
            "attention_policy": "global FP32 attention; independent query rows sliced above 8192 tokens; every query retains all image keys",
            "processing_patch_tokens": tokens, "processing_patch_limit": DA3_MAX_PATCH_TOKENS,
            "decoder_policy": "full-grid bilinear decoder with bounded row evaluation and complete convolution halo",
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
            raise LearnedDepthError("DA3 processor exceeded the supported full-resolution patch budget")
        tensor = imgs[None].to(device=self.device, dtype=torch.float32)
        # The public DA3 wrapper enables FP16/BF16 autocast unconditionally.
        # Call its underlying network to honor IPDE's FP32 inference contract.
        output = (_da3_depth_only(self.model.model, tensor) if getattr(self, "_da3_primary_only", False)
                  else self.model.model(tensor, None, None, [], False, False, "first"))
        plane = _plane(torch, output["depth"])
        native = torch.from_numpy(plane)[None, None]
        source = (native if plane.shape == rgb.shape[:2] else
                  torch.nn.functional.interpolate(native, rgb.shape[:2], mode="bilinear", align_corners=False))
        confidence = _plane(torch, output["depth_conf"]) if "depth_conf" in output else None
        return native, source, {
            "units": "relative_depth", "quantity": "relative_camera_z_depth", "metric_scale": "unavailable",
            "processing": "upstream uint8 RGB aspect-preserving upper_bound_resize and ImageNet normalization; underlying network FP32",
            "source_grid_resampling": "none; prediction already matches the full reference grid" if plane.shape == rgb.shape[:2] else "bilinear relative depth, align_corners=False; no sharpening",
            "model_input_shape": list(tensor.shape[-2:]), "input_size": input_size,
            "requested_input_size": self.config.input_size, "input_size_policy": "native source longest side" if not self.config.input_size else "configured longest side",
            "native_grid": "network output at processed RGB grid covering the full reference extent",
            "confidence_semantics": "unmodified DA3 depth_conf scores; not calibrated probabilities",
            "processing_patch_tokens": tokens, "processing_patch_limit": DA3_MAX_PATCH_TOKENS,
            "expected_model_input_shape": list(expected_shape),
            "attention_policy": "global FP32 attention; independent query rows sliced above 8192 tokens; every query retains all image keys",
            "attention_score_working_bytes": DA3_ATTENTION_SCORE_BYTES,
            "decoder_policy": "primary depth branch only; full-grid bilinear decoder with bounded row evaluation and complete convolution halo",
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
