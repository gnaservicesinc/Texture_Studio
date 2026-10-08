"""Pinned, frozen DINOv2 features plus a native-pixel material-height head.

The official Meta encoder source and Apache-2.0 weights remain separate from
this GPL diagnostic. This module never downloads or executes remote code.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from material_height_model import ConvBlock

MODEL_REPO = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
MODEL_BYTES = 346345912
CONFIG_SHA256 = "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70"
CODE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DIAGNOSTIC_SCHEMA = "texture-studio-frozen-dino-height-diagnostic-v1"
ARCHITECTURE = "native-height-unet-frozen-dino-fusion-v1"
CODE_HASHES = {
    "LICENSE": "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    "dinov2/__init__.py": "0d2b87b7e71c7f7279ccc4b4f8b541d26071425284a0ea01a5a435fac4511cfe",
    "dinov2/models/vision_transformer.py": "7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea",
    "dinov2/layers/__init__.py": "1b55deed39d5ab0b589bef421bbfe06f24a10cd590a0f8403234c9ee4e34109d",
    "dinov2/layers/dino_head.py": "9fdb1fa61c0609dc0876f711f9d0d828c33bf9fd972aab3d9c968d44e16b98d1",
    "dinov2/layers/layer_scale.py": "dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135",
    "dinov2/layers/mlp.py": "255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa",
    "dinov2/layers/patch_embed.py": "40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142",
    "dinov2/layers/swiglu_ffn.py": "e46d2948fb97e497cf991ff82ce30cb59b59fdf2e12a19568417113716b4f119",
    "dinov2/layers/block.py": "60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636",
    "dinov2/layers/attention.py": "79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d",
    "dinov2/layers/drop_path.py": "b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a",
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def import_official_encoder(resolved: Path, device: torch.device):
    """Select native attention without rewriting the checksum-pinned source.

    Each worker uses one accelerator. DINOv2 selects optional xFormers kernels
    by import availability, not tensor device; CPU and Metal use PyTorch SDPA.
    Only the audited layer modules' exact optional-dependency notices are hidden.
    """
    native = device.type in ("mps", "cpu")
    previous = os.environ.get("XFORMERS_DISABLED")
    sys.path.insert(0, str(resolved))
    try:
        if native:
            os.environ["XFORMERS_DISABLED"] = "1"
        with warnings.catch_warnings():
            if native:
                warnings.filterwarnings("ignore", category=UserWarning,
                    message=r"^xFormers is (?:disabled|not available) \((?:SwiGLU|Attention|Block)\)$",
                    module=r"^dinov2\.layers\.(?:swiglu_ffn|attention|block)$")
            official = importlib.import_module("dinov2.models.vision_transformer")
        if native:
            # A previous load from this same verified snapshot may already be
            # cached. Its runtime switches must also select the native path.
            for name in ("attention", "block", "swiglu_ffn"):
                module = sys.modules.get("dinov2.layers." + name)
                if module is not None and hasattr(module, "XFORMERS_AVAILABLE"):
                    module.XFORMERS_AVAILABLE = False
        return official
    finally:
        sys.path.remove(str(resolved))
        if native:
            if previous is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = previous


def convert_hf_state(state: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], depth: int = 12) -> dict[str, torch.Tensor]:
    """Map every HF tensor, including the mask token, to Meta's exact layout."""
    result: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def take(source: str) -> torch.Tensor:
        if source not in state:
            raise ValueError(f"Missing pinned encoder tensor: {source}")
        used.add(source)
        value = state[source]
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Encoder tensor must be finite Float32: {source}")
        return value

    for target, source in {
        "cls_token": "embeddings.cls_token", "mask_token": "embeddings.mask_token",
        "pos_embed": "embeddings.position_embeddings",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "norm.weight": "layernorm.weight", "norm.bias": "layernorm.bias",
    }.items():
        result[target] = take(source)
    for level in range(depth):
        source, target = f"encoder.layer.{level}.", f"blocks.{level}."
        for suffix in ("weight", "bias"):
            result[target + "attn.qkv." + suffix] = torch.cat([take(source + f"attention.attention.{part}." + suffix) for part in ("query", "key", "value")])
            result[target + "attn.proj." + suffix] = take(source + "attention.output.dense." + suffix)
            for layer in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                result[target + layer + "." + suffix] = take(source + layer + "." + suffix)
        for index in (1, 2):
            result[target + f"ls{index}.gamma"] = take(source + f"layer_scale{index}.lambda1")
    if used != set(state):
        raise ValueError(f"Unexpected encoder tensors: {sorted(set(state) - used)}")
    if set(result) != set(expected):
        raise ValueError(f"Converted encoder key mismatch: missing={sorted(set(expected)-set(result))}, extra={sorted(set(result)-set(expected))}")
    for name, value in result.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Encoder tensor shape mismatch for {name}: {value.shape} != {expected[name].shape}")
    return result


def load_frozen_encoder(model_directory: Path, code_directory: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load one audited official source snapshot and one pinned safe checkpoint."""
    checkpoint = model_directory / "model.safetensors"
    if checkpoint.stat().st_size != MODEL_BYTES or file_sha256(checkpoint) != MODEL_SHA256:
        raise ValueError(f"Pinned DINOv2 Base checksum/size mismatch: {checkpoint}")
    config_path = model_directory / "config.json"
    if file_sha256(config_path) != CONFIG_SHA256:
        raise ValueError(f"Pinned DINOv2 config checksum mismatch: {config_path}")
    config = json.loads(config_path.read_text())
    required = {"model_type": "dinov2", "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "patch_size": 14, "image_size": 518, "num_channels": 3, "mlp_ratio": 4, "layer_norm_eps": 1e-6, "qkv_bias": True, "use_swiglu_ffn": False, "hidden_act": "gelu", "torch_dtype": "float32", "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0, "drop_path_rate": 0.0}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("DINOv2 Base config does not match the audited architecture")
    for path, expected_hash in CODE_HASHES.items():
        if file_sha256(code_directory / path) != expected_hash:
            raise ValueError(f"Official DINOv2 source checksum mismatch: {path}")
    if {str(path.relative_to(code_directory)) for path in (code_directory / "dinov2").rglob("*.py")} != {path for path in CODE_HASHES if path.endswith(".py")}:
        raise ValueError("Official source snapshot contains unexpected Python modules")
    # Refuse import reuse from an unrelated installation or a different snapshot.
    resolved = code_directory.resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "dinov2" or name.startswith("dinov2."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(resolved):
                raise ValueError(f"An unrelated DINOv2 module is already imported: {location}")
    official = import_official_encoder(resolved, device)
    encoder = official.vit_base(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1)
    from safetensors.torch import load_file
    state = load_file(str(checkpoint), device="cpu")
    converted = convert_hf_state(state, encoder.state_dict())
    encoder.load_state_dict(converted, strict=True)
    del state, converted
    encoder.requires_grad_(False).eval().to(device)
    native = device.type in ("mps", "cpu")
    if native:
        print(json.dumps({"event": "encoder_runtime", "device": str(device),
            "attention_backend": "pytorch_scaled_dot_product_attention",
            "message": "DINOv2 uses native PyTorch attention on Metal; optional xFormers is not required."
                if device.type == "mps" else "DINOv2 uses native PyTorch attention on CPU; optional xFormers is not required."}), flush=True)
    return encoder, {
        "repo": MODEL_REPO, "revision": MODEL_REVISION, "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": MODEL_SHA256, "checkpoint_bytes": MODEL_BYTES,
        "config_sha256": file_sha256(config_path), "official_meta_code_revision": CODE_REVISION,
        "official_source_sha256": CODE_HASHES, "official_source_directory": str(resolved),
        "weight_mapping": "Strict complete HF-to-official-Meta mapping; Q,K,V concatenated in that order; mask token retained; no missing/extra/shape-mismatched tensors",
        "license": "Apache-2.0", "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "all_encoder_parameters_frozen": all(not parameter.requires_grad for parameter in encoder.parameters()),
        "attention_backend": "pytorch_scaled_dot_product_attention" if native else "upstream_device_default",
        "device": str(device), "xformers_enabled": False if native else None,
    }


def encoder_input(linear_rgb: torch.Tensor, size: int = 518) -> torch.Tensor:
    """Whole-crop encoder-only sRGB conversion; native branch stays linear."""
    if size < 28 or size > 518 or size % 14:
        raise ValueError("Encoder size must be patch-aligned (14px) between 28 and 518")
    if linear_rgb.ndim != 4 or linear_rgb.shape[1] != 3 or not torch.isfinite(linear_rgb).all() or bool((linear_rgb < 0).any()) or bool((linear_rgb > 1).any()):
        raise ValueError("Encoder input must be finite linear RGB in [0,1]")
    rgb = torch.where(linear_rgb <= 0.0031308, linear_rgb * 12.92, 1.055 * linear_rgb.clamp_min(0).pow(1 / 2.4) - 0.055)
    rgb = F.interpolate(rgb, size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    mean = rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (rgb - mean) / std


@torch.no_grad()
def extract_features(encoder: nn.Module, linear_rgb: torch.Tensor, size: int = 518) -> torch.Tensor:
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Feature encoder must be frozen")
    encoder.eval()
    tokens = encoder.forward_features(encoder_input(linear_rgb, size))["x_norm_patchtokens"]
    side = size // 14
    if tokens.ndim != 3 or tokens.shape != (linear_rgb.shape[0], side * side, 768) or not torch.isfinite(tokens).all():
        raise ValueError("Encoder did not return finite final-layer normalized ViT-B14 patch tokens")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], 768, side, side).contiguous().detach()


class ConditionedHeightNet(nn.Module):
    """Native RGB/detail branch; project coarse features before spatial growth."""
    def __init__(self, base_channels: int = 12, feature_channels: int = 768, projection_channels: int = 12) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4 or min(feature_channels, projection_channels) < 1:
            raise ValueError("Invalid native head channel dimensions")
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 6)
        self.feature_channels = feature_channels
        self.encoder = nn.ModuleList(ConvBlock(incoming, outgoing) for incoming, outgoing in zip((3,) + widths[:-1], widths))
        self.feature_projection = nn.Conv2d(feature_channels, projection_channels, 1)
        self.feature_fusion = ConvBlock(widths[-1] + projection_channels, widths[-1])
        self.decoder = nn.ModuleList(ConvBlock(widths[level + 1] + widths[level], widths[level]) for level in (2, 1, 0))
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or features.ndim != 4 or features.shape[:2] != (rgb.shape[0], self.feature_channels):
            raise ValueError("Expected native RGB plus batch-aligned frozen feature grid")
        if features.requires_grad:
            raise ValueError("Cached encoder features must be detached")
        skips, value = [], rgb
        for level, block in enumerate(self.encoder):
            if level:
                value = F.avg_pool2d(value, 2)
            value = block(value)
            skips.append(value)
        # Only projection_channels, never the full 768 channels, are upsampled.
        projected = self.feature_projection(features)
        projected = F.interpolate(projected, size=value.shape[-2:], mode="bilinear", align_corners=False)
        value = self.feature_fusion(torch.cat((value, projected), dim=1))
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return torch.sigmoid(self.head(value))


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
