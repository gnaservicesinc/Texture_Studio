"""Pinned PBRnxt complete texture mapping adapted to a native pixel grid.

PBRnxt: Copyright (c) 2024 Andrejs Krauze, MIT.
SCUNet: Copyright 2022 Kai Zhang, Apache-2.0.
Swin Transformer V2: Copyright (c) Microsoft Corporation, MIT.
ESRGAN+: Apache-2.0; exact upstream notice retained beside its RRDB source.
The exact upstream notices are verified/downloaded alongside the source and
must accompany redistributed model packages. This adapter changes device
handling and boundary execution, never source pixels or pretrained weights.

The complete mapping preserves the generator and every trained RRDB output
convolution/head, bypassing only two nearest-neighbor enlargement operations
per output branch. This is an explicit scale adaptation, not the unchanged
published model. Generator-only outputs are intermediate features; isolated
visual tests found a strong grid, so they are not standalone material maps.

Upstream preprocessing writes UInt8 JPEG references; refinement with
original UInt16 heights is a separate experiment, not an assertion that the
base was trained with 16-bit maps. Direct core execution omits upstream's
reflective image border and supersampling. Native dimensions must be multiples
of 64, including 2048; shape support alone does not establish material quality.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

REVISION = "73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35"
WEIGHTS_NAME = "pbrnxt_402236.pth"
WEIGHTS_BYTES = 349493406
WEIGHTS_SHA256 = "3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d"
SOURCE_FILES = {
    "archs/scunetv2_arch.py": (29227, "52c8a38566e425671c8304e8bc31730674e8ba6cc501c1ad76373c355c01f282"),
    "archs/rrdbnet_arch.py": (11411, "106b48ac8a1fe5538ff040aa7070c6e1ebb55059d4981a7a4d301545487ec013"),
    "LICENSE": (1071, "002f2619d15a6777972e35171a7672a9371981a48a9477455c27adf6ac788008"),
    "licenses/SCUNet_LICENSE": (11408, "25c06365725101ede0aac6fa5c821aa4160b40f2fde5aded8be95beade26d8df"),
    "licenses/SwinTransformer_LICENSE": (1140, "d9a1b1e30d633d5732ea18e3cba9538d293ebc53e1a9e4e96ab739e0c5c4f1cb"),
    "licenses/ESRGANplus_LICENSE": (11356, "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"),
    "README.md": (3620, "81ee1a76efe723e362dd20fc1a35b6dd8f3b567de03f2a3e888fc9a44c0694e6"),
}
GRID = 64
MAP_CHANNELS = (3, 3, 1, 1)
MAP_NAMES = ("diffuse", "normal", "roughness", "height")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned PBRnxt file: {path}")
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Pinned PBRnxt file identity mismatch: {path}")


def source_provenance(source_dir: Path) -> list[dict[str, Any]]:
    records = []
    for name, (size, digest) in SOURCE_FILES.items():
        checked_file(source_dir / name, size, digest)
        records.append({"name": name, "bytes": size, "sha256": digest,
                        "url": f"https://raw.githubusercontent.com/aaf6aa/PBRnxt/{REVISION}/{name}"})
    return records


def _obtain_file(path: Path, size: int, digest: str, url: str, download: bool) -> None:
    if path.exists():
        checked_file(path, size, digest)
        return
    if not download:
        raise FileNotFoundError(f"Missing {path}; enable download to obtain the pinned PBRnxt files")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pbrnxt-download-", dir=path.parent)
    temporary = Path(name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TextureStudio-PBRnxt"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=120) as incoming:
            shutil.copyfileobj(incoming, output, length=4 * 1024 * 1024)
        checked_file(temporary, size, digest)
        # Preserve any concurrently supplied existing file, after verification.
        try:
            os.link(temporary, path)
        except FileExistsError:
            checked_file(path, size, digest)
    finally:
        temporary.unlink(missing_ok=True)


def obtain_pretrained(directory: Path, download: bool = False) -> tuple[Path, Path]:
    """Obtain only pinned architecture/notices and weights, not a whole repo."""
    source_dir = directory / "source"
    for name, (size, digest) in SOURCE_FILES.items():
        _obtain_file(source_dir / name, size, digest,
                     f"https://raw.githubusercontent.com/aaf6aa/PBRnxt/{REVISION}/{name}", download)
    weights = directory / WEIGHTS_NAME
    _obtain_file(weights, WEIGHTS_BYTES, WEIGHTS_SHA256,
                 f"https://media.githubusercontent.com/media/aaf6aa/PBRnxt/{REVISION}/pretrained_models/{WEIGHTS_NAME}", download)
    return source_dir, weights


def obtain_source(directory: Path, download: bool = False) -> Path:
    """Recover the small pinned architecture without redownloading base weights."""
    source_dir = directory / "source"
    for name, (size, digest) in SOURCE_FILES.items():
        _obtain_file(source_dir / name, size, digest,
                     f"https://raw.githubusercontent.com/aaf6aa/PBRnxt/{REVISION}/{name}", download)
    return source_dir


class DeviceGaussianNoise(nn.Module):
    """Same training noise on the input device; deterministic evaluation.

    The upstream CUDA scalar was not a buffer/parameter, so replacing this
    class changes no checkpoint state keys. Its old unconditional inference
    noise is explicitly disabled in eval for reproducible base comparisons.
    """
    def __init__(self, sigma: float = .1, is_relative_detach: bool = False, octaves: int = 3):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.octaves = max(1, octaves)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma == 0:
            return value
        scale = self.sigma * (value.detach() if self.is_relative_detach else value)
        # Swin passes BHWC, ConvNeXt BCHW. The original code applied octave
        # sizes to the final two axes in both cases; retain that behavior.
        batch, channels, height, width = value.shape
        for octave in range(self.octaves):
            small = (batch, channels, max(1, height // (2 ** octave)), max(1, width // (2 ** octave)))
            noise = torch.randn(small, device=value.device, dtype=value.dtype)
            noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
            value = value + noise * scale
        return value


def _architecture(source_dir: Path) -> Any:
    # Verify all executable source and its exact notices before importing.
    source_provenance(source_dir)
    spec = importlib.util.spec_from_file_location("texture_studio_pbrnxt_pinned_arch", source_dir / "archs/scunetv2_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned PBRnxt architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GaussianNoise = DeviceGaussianNoise
    return module


def _rrdb_architecture(source_dir: Path) -> Any:
    source_provenance(source_dir)
    spec = importlib.util.spec_from_file_location("texture_studio_pbrnxt_pinned_rrdb", source_dir / "archs/rrdbnet_arch.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned PBRnxt output architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GaussianNoise = DeviceGaussianNoise
    return module


def check_native_input(value: torch.Tensor) -> None:
    if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != 3 or not value.is_floating_point():
        raise ValueError("PBRnxt input must be floating-point [B,3,H,W] RGB")
    height, width = value.shape[-2:]
    if min(height, width) < GRID or height % GRID or width % GRID:
        raise ValueError("Use a native crop with each dimension divisible by 64; image padding and resizing are forbidden")


def native_generator_forward(gen: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """The published generator's layers, same weights/features, no image pad.

    This deliberately does not call SCUNet.forward (which always reflects a
    64-pixel border), or PbrNxtNet.forward (circular border and 4x upscalers).
    Normal convolution padding inside trained layers is architecture behavior;
    no artificial source pixels are inserted into the input/target tensor.
    """
    check_native_input(value)
    initial = gen.m_head(value)
    stage1, stage2, stage3 = gen.m_enc(initial)
    body = gen.m_body(stage3)
    decoded = [getattr(gen, f"m_dec_{index}")(body, stage1, stage2, stage3)[0]
               for index in range(len(gen.out_nc))]
    concatenated = torch.cat(decoded, dim=1)
    fused = gen.m_fuse(concatenated) + concatenated
    features = torch.split(fused, gen.dim, dim=1)
    output = torch.cat([getattr(gen, f"m_tail_{index}")(feature + initial)
                        for index, feature in enumerate(features)], dim=1)
    if output.shape != (value.shape[0], sum(MAP_CHANNELS), *value.shape[-2:]):
        raise ValueError("PBRnxt core changed native output dimensions/channels")
    return output


class NativePBRnxt(nn.Module):
    """Diagnostic generator-only intermediate, not standalone material maps."""
    def __init__(self, gen: nn.Module, provenance: dict):
        super().__init__()
        self.gen = gen
        self.provenance = provenance

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return native_generator_forward(self.gen, value)

    def maps(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic intermediate channels only; retain raw numbers."""
        return dict(zip(MAP_NAMES, self(value).split(MAP_CHANNELS, dim=1)))


def load_pretrained(source_dir: Path, weights_path: Path, device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.float32) -> NativePBRnxt:
    """Load generator-only diagnostic intermediate, not final height maps.

    Use load_complete_pretrained for the full native-scale output mapping.
    """
    checked_file(weights_path, WEIGHTS_BYTES, WEIGHTS_SHA256)
    architecture = _architecture(source_dir)
    # Exact .gen constructor used by PbrNxtNet(3,[3,3,1,1],96,32,4,256).
    # Its input_resolution128 setting is not a resize operation.
    gen = architecture.SCUNet(3, list(MAP_CHANNELS), 96, 1, 2, 2, 4, .1, 0., 128, 1)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("PBRnxt checkpoint must be a tensor state dictionary")
    generator_state = {key.removeprefix("gen."): value for key, value in state.items() if key.startswith("gen.")}
    if not generator_state:
        raise ValueError("PBRnxt checkpoint lacks its pretrained generator")
    # No missing-key fallback or newly initialized replacement decoder.
    gen.load_state_dict(generator_state, strict=True)
    model = NativePBRnxt(gen, {
        "model": "PBRnxt generator intermediate · not standalone material maps", "revision": REVISION,
        "standalone_material_outputs": False, "production_eligible": False,
        "qualification": "Pinned generator-only intermediate produces a grid; trained RRDB output mapping is required",
        "weights_name": WEIGHTS_NAME, "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
        "source_files": source_provenance(source_dir), "licenses": ["MIT", "Apache-2.0"],
        "native_grid": GRID, "map_channels": dict(zip(MAP_NAMES, MAP_CHANNELS)), "normal_convention": "OpenGL +Y",
        "generator_parameters": sum(parameter.numel() for parameter in gen.parameters()),
        "height_decoder_parameters": sum(parameter.numel() for name, parameter in gen.named_parameters()
                                          if name.startswith(("m_dec_3.", "m_tail_3."))),
        "image_padding": False, "image_resizing": False, "upscaler_used": False,
        "evaluation_noise": "disabled for deterministic comparison; device-native upstream octave noise retained in training",
        "pretraining_target_notice": "Published preprocessing writes UInt8 JPEG references; our UInt16 refinement targets are separate",
    }).to(device=device, dtype=dtype)
    model.eval()
    return model


def native_branch_forward(branch: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Keep every pretrained RRDB convolution; omit only its two enlargements."""
    output = value
    skipped = 0
    for module in branch.model:
        if isinstance(module, nn.Upsample):
            if module.scale_factor != 2 or module.mode != "nearest":
                raise ValueError("Pinned RRDB branch has an unexpected enlargement operator")
            skipped += 1
            continue
        if (getattr(branch, "material_gradient_checkpointing", False) and torch.is_grad_enabled()
                and module.__class__.__name__ == "ShortcutBlock" and isinstance(module.sub, nn.Sequential)):
            # The pinned RRDB implementation already recomputes each dense
            # sub-block. Checkpoint the outer RRDB too, so large native LoRA
            # updates retain one boundary tensor per RRDB rather than all three
            # dense boundaries. State keys, float32 operations, and outputs are
            # unchanged; only backward activation storage/recomputation differs.
            from torch.utils.checkpoint import checkpoint
            residual = output
            for block in module.sub:
                output = checkpoint(block, output, use_reentrant=False) if block.__class__.__name__ == "RRDB" else block(output)
            output = residual + output
        else:
            output = module(output)
    if skipped != 2:
        raise ValueError("Native-scale adaptation requires exactly two pinned RRDB enlargements")
    if output.shape[0] != value.shape[0] or output.shape[-2:] != value.shape[-2:]:
        raise ValueError("Complete PBRnxt branch changed native output dimensions")
    return output


class NativePBRnxtComplete(nn.Module):
    """Full pretrained mapping, explicitly adapted from 4x to native scale.

    .height computes the same final height as .forward without evaluating the
    other three final RRDB branches. Every generator decoder/fusion remains
    necessary because all intermediate features feed the trained height branch.
    """
    def __init__(self, gen: nn.Module, ups: nn.ModuleList, provenance: dict):
        super().__init__()
        self.gen = gen
        self.ups = ups
        self.provenance = provenance

    def _branch_input(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value, native_generator_forward(self.gen, value)), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self._branch_input(value)
        parts = [native_branch_forward(branch, features) for branch in self.ups]
        if len(parts) != len(MAP_CHANNELS) or any(part.shape[1] != channels for part, channels in zip(parts, MAP_CHANNELS)):
            raise ValueError("Complete PBRnxt output branches have unexpected channels")
        return torch.cat(parts, dim=1)

    def height(self, value: torch.Tensor) -> torch.Tensor:
        output = native_branch_forward(self.ups[3], self._branch_input(value))
        if output.shape[1] != 1:
            raise ValueError("Pretrained PBRnxt height branch must return one scalar channel")
        return output

    def map(self, value: torch.Tensor, target: str) -> torch.Tensor:
        """Evaluate one existing trained material output branch."""
        if target not in MAP_NAMES:
            raise ValueError("Unknown material map target")
        index = MAP_NAMES.index(target)
        return native_branch_forward(self.ups[index], self._branch_input(value))

    def maps(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw outputs with no clipping, height range stretch, or gamma changes."""
        return dict(zip(MAP_NAMES, self(value).split(MAP_CHANNELS, dim=1)))


def complete_architecture(source_dir: Path) -> NativePBRnxtComplete:
    """Build the verified architecture for loading a complete fused checkpoint."""
    architecture, rrdb = _architecture(source_dir), _rrdb_architecture(source_dir)
    gen = architecture.SCUNet(3, list(MAP_CHANNELS), 96, 1, 2, 2, 4, .1, 0., 128, 1)
    ups = nn.ModuleList([rrdb.RRDBNet(3 + sum(MAP_CHANNELS), channels, 32, 12, upscale=4, plus=True)
                         for channels in MAP_CHANNELS])
    return NativePBRnxtComplete(gen, ups, {})


def load_complete_pretrained(source_dir: Path, weights_path: Path, device: str | torch.device = "cpu",
                             dtype: torch.dtype = torch.float32) -> NativePBRnxtComplete:
    """Strict-load all published weights for an experimental native mapping.

    No source/target padding or resizing. All trainable convolutions and output
    heads are retained, including the essential trained final height branch.
    The omitted 4x enlargements change spatial operation; this adaptation still
    requires visual qualification and refinement before production selection.
    """
    checked_file(weights_path, WEIGHTS_BYTES, WEIGHTS_SHA256)
    model = complete_architecture(source_dir)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("PBRnxt checkpoint must be a tensor state dictionary")
    # Includes every generator/auxiliary and RRDB branch key. Missing final
    # height parameters cannot fall back to a randomly initialized head.
    model.load_state_dict(state, strict=True)
    model.provenance = {
        "model": "PBRnxt complete pretrained mapping · experimental native-scale adaptation",
        "revision": REVISION, "weights_name": WEIGHTS_NAME, "weights_sha256": WEIGHTS_SHA256,
        "weights_bytes": WEIGHTS_BYTES, "strict_state_keys": len(state),
        "source_files": source_provenance(source_dir), "licenses": ["MIT", "Apache-2.0"],
        "native_grid": GRID, "map_channels": dict(zip(MAP_NAMES, MAP_CHANNELS)), "normal_convention": "OpenGL +Y",
        "complete_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "generator_parameters": sum(parameter.numel() for parameter in model.gen.parameters()),
        "final_height_branch_parameters": sum(parameter.numel() for parameter in model.ups[3].parameters()),
        "adaptation": "All pretrained generator and RRDB convolutions/heads retained; omit only two nearest-neighbor2x enlargements per final branch",
        "published_output_scale": 4, "adapted_output_scale": 1,
        "enlargement_operations_omitted_per_branch": 2, "all_pretrained_parameter_keys_loaded": True,
        "image_padding": False, "image_resizing": False, "upscaler_enlargement_used": False,
        "trained_final_rrdb_mapping_used": True, "production_eligible": False,
        "evaluation_noise": "disabled for deterministic comparison; device-native upstream octave noise retained in training",
        "pretraining_target_notice": "Published preprocessing writes UInt8 JPEG references; our UInt16 refinement targets are separate",
        "qualification_notice": "Native-scale baseline responds to texture details but has border artifacts and weak/inverted bulk relief; production quality unverified",
    }
    model.to(device=device, dtype=dtype).eval()
    return model
