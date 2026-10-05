"""Correspondence-aligned native stereo to display-grid depth model.

The full display teacher is never warped onto a stereo grid. A learned query
decoder predicts directly at each requested display pixel. RAFT provides native
stereo correspondence context, not the output depth grid or a metric label.
Nothing here changes the stored RGB, auxiliary arrays or teacher predictions.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import math
from pathlib import Path
import sys
from typing import Any, Iterator
import warnings

import numpy as np

SCHEMA = "ipde-display-depth-v1"
ARCHITECTURE = "stereo-display-query-v2"
LEGACY_ARCHITECTURE = "stereo-display-query-v1"
UNITS = {"meters", "relative_depth", "relative_inverse_depth"}


class DisplayStudentError(RuntimeError):
    pass


def rgb_tensor(array: np.ndarray, record: Mapping[str, Any], torch: Any, device: str):
    """Copy native RGB into the model's nominal 0..255 convention, without resizing."""
    source = np.asarray(array)
    if source.ndim != 3 or source.shape[2] != 3 or not source.size:
        raise DisplayStudentError("Model inputs require nonempty native HxWx3 RGB")
    if source.dtype.kind == "u":
        bits = record.get("source_bit_depth", source.dtype.itemsize * 8)
        if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= source.dtype.itemsize * 8:
            raise DisplayStudentError("Model RGB bit depth is invalid")
        limit = float((1 << bits) - 1)
    elif source.dtype.kind == "f":
        limit = 1.0
    else:
        raise DisplayStudentError("Model RGB must be unsigned integer or linear floating point")
    if not np.isfinite(source).all() or np.any(source < 0) or np.any(source > limit):
        raise DisplayStudentError("Model RGB exceeds its declared nominal range")
    value = np.array(source, dtype=np.float32, copy=True) * np.float32(255.0 / limit)
    return torch.from_numpy(value).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)


def _upstream(root: Path):
    root = Path(root).expanduser().resolve()
    if not (root / "core/raft_stereo.py").is_file():
        raise DisplayStudentError(f"RAFT-Stereo source is missing: {root}")
    # Refuse a silently cached class from a different configured source tree.
    loaded = sys.modules.get("core.raft_stereo")
    filename = getattr(loaded, "__file__", None)
    if filename and not Path(filename).resolve().is_relative_to(root):
        raise DisplayStudentError("A different RAFT source is already loaded; start a fresh process for this source")
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder
        return RAFTStereo, InputPadder
    finally:
        if inserted:
            sys.path.remove(str(root))


def align_right_to_left(right, flow, reverse_flow):
    """Copy right RGB into left coordinates using signed RAFT correspondence.

    Flow is x_right - x_left; reverse_flow is x_left - x_right on the right
    grid. Both bracketing reverse matches must agree within one native pixel.
    Off-image and occluded samples have zero weight in the fused features.
    The input tensors remain untouched; this is model preprocessing only.
    """
    import torch
    from torch.nn import functional as F
    batch, _, height, width = right.shape
    if flow.shape != (batch, 1, height, width) or reverse_flow.shape != flow.shape:
        raise DisplayStudentError("Stereo correspondence does not match the native RGB grid")
    x = torch.arange(width, device=right.device, dtype=torch.float32)[None, None, None, :]
    y = torch.arange(height, device=right.device, dtype=torch.float32)[None, None, :, None]
    matched = x + flow
    finite = torch.isfinite(matched)
    safe = torch.where(finite, matched, torch.zeros_like(matched)).clamp(0, width - 1)
    low, high = safe.floor().long(), safe.ceil().long()
    low_reverse, high_reverse = reverse_flow.gather(-1, low), reverse_flow.gather(-1, high)
    supported = (finite & (matched >= 0) & (matched <= width - 1)
                 & torch.isfinite(low_reverse) & torch.isfinite(high_reverse)
                 & ((low.to(flow.dtype) + low_reverse - x).abs() <= 1.0)
                 & ((high.to(flow.dtype) + high_reverse - x).abs() <= 1.0))
    grid = torch.stack(((safe[:, 0] + .5) * (2.0 / width) - 1.0,
                        ((y + .5) * (2.0 / height) - 1.0).expand(batch, 1, height, width)[:, 0]), dim=-1)
    aligned = F.grid_sample(right, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return aligned * supported.to(right.dtype), supported


def _model(configuration: Mapping[str, Any], raft_root: Path):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from types import SimpleNamespace

    fields = {"architecture", "units", "channels", "decoder_channels", "iterations", "raft_configuration",
              "input_reference", "output_reference", "teacher_transport", "original_raft_checkpoint_sha256"}
    aligned = configuration.get("architecture") == ARCHITECTURE
    if aligned:
        fields |= {"stereo_alignment", "query_reference"}
    if set(configuration) != fields or configuration.get("architecture") not in {ARCHITECTURE, LEGACY_ARCHITECTURE} or configuration.get("units") not in UNITS:
        raise DisplayStudentError("Unsupported display-depth model architecture or units")
    if aligned and (configuration.get("stereo_alignment") != "right_to_left_raft_reverse_consistency"
                    or configuration.get("query_reference") != "single_fused_left_field"):
        raise DisplayStudentError("Model contradicts the single-reference stereo alignment contract")
    if (configuration.get("input_reference") != "native_stereo_pair" or configuration.get("output_reference") != "display"
            or configuration.get("teacher_transport") != "none"):
        raise DisplayStudentError("Model checkpoint contradicts the native-stereo to display-depth contract")
    digest = configuration.get("original_raft_checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DisplayStudentError("Model checkpoint must identify its original RAFT weights")
    channels = configuration.get("channels", 24)
    hidden = configuration.get("decoder_channels", 64)
    if channels not in {16, 24, 32} or hidden not in {32, 64, 96}:
        raise DisplayStudentError("Invalid model channel configuration")
    iterations = configuration.get("iterations", 16)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 256:
        raise DisplayStudentError("Invalid model RAFT iteration configuration")
    RAFTStereo, InputPadder = _upstream(raft_root)
    from .spatial import checkpoint_model_configuration
    raft_config = vars(checkpoint_model_configuration({"ipde_configuration": configuration["raft_configuration"]}, "base.pth"))

    class StereoDisplayStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.configuration = dict(configuration)
            self.raft = RAFTStereo(SimpleNamespace(**raft_config))
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv2d(16, channels, 3, stride=2, padding=1), nn.GroupNorm(4, channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.GroupNorm(4, channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            )
            self.flow_encoder = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.SiLU(),
                nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(8, 8, 3, stride=2, padding=1))
            if aligned:
                self.fusion = nn.Sequential(nn.Conv2d(2 * channels + 9, channels, 3, padding=1), nn.SiLU(),
                                            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
            context_channels = channels if aligned else 2 * channels + 8
            self.global_encoder = nn.Linear(context_channels, channels)
            query_channels = context_channels + channels + 4
            self.offset = nn.Sequential(nn.Conv2d(query_channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 2 if aligned else 4, 1))
            self.depth = nn.Sequential(nn.Conv2d(query_channels, hidden, 1), nn.SiLU(),
                nn.Conv2d(hidden, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
            # Begin with zero learned camera offsets and a finite positive depth.
            nn.init.zeros_(self.offset[-1].weight); nn.init.zeros_(self.offset[-1].bias)
            nn.init.constant_(self.depth[-1].bias, math.log(math.expm1(1.0)))
            self.set_scope("update")

        def set_scope(self, scope: str):
            if scope not in {"update", "full"}:
                raise DisplayStudentError("Model train scope must be update or full")
            self.raft_trainable = scope == "full"
            for parameter in self.raft.parameters():
                parameter.requires_grad_(self.raft_trainable)
            for name, parameter in self.named_parameters():
                if not name.startswith("raft."):
                    parameter.requires_grad_(True)

        def freeze_bn(self):
            if hasattr(self.raft, "freeze_bn"):
                self.raft.freeze_bn()
            if not self.raft_trainable:
                self.raft.eval()

        def encode(self, left, right):
            if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
                raise DisplayStudentError("Left/right tensors must share native Bx3xHxW dimensions")
            self.freeze_bn()
            padder = InputPadder(left.shape, divis_by=32)
            a, b = padder.pad(left, right)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)
                with torch.set_grad_enabled(torch.is_grad_enabled() and self.raft_trainable):
                    result = self.raft(a, b, iters=iterations, test_mode=True)
                    flow = padder.unpad(result[-1])
                    if aligned:
                        reverse_left, reverse_right = padder.pad(torch.flip(right, dims=[3]), torch.flip(left, dims=[3]))
                        # Reverse correspondence supplies a boolean support mask;
                        # retaining its gradient graph cannot affect the loss.
                        with torch.no_grad():
                            reverse_result = self.raft(reverse_left, reverse_right, iters=iterations, test_mode=True)
                        reverse_flow = -torch.flip(padder.unpad(reverse_result[-1]), dims=[3])
            if flow.shape != (left.shape[0], 1, left.shape[2], left.shape[3]) or not bool(torch.isfinite(flow).all()):
                raise DisplayStudentError("RAFT context produced invalid native-grid correspondence")
            left_features = self.encoder(left / 127.5 - 1.0)
            flow_features = self.flow_encoder(flow / float(left.shape[-1]))
            if aligned:
                right_aligned, support = align_right_to_left(right, flow, reverse_flow)
                coverage = F.adaptive_avg_pool2d(support.to(left.dtype), left_features.shape[-2:])
                right_features = self.encoder(right_aligned / 127.5 - 1.0) * coverage
                fused = self.fusion(torch.cat((left_features, right_features, flow_features, coverage), dim=1))
                return {"fused": fused, "global": self.global_encoder(fused.mean(dim=(2, 3))),
                        "input_shape": tuple(left.shape[-2:]), "stereo_support": support}
            right_features = self.encoder(right / 127.5 - 1.0)
            pooled = torch.cat([item.mean(dim=(2, 3)) for item in (left_features, right_features, flow_features)], dim=1)
            return {"left": left_features, "right": right_features, "flow": flow_features,
                    "global": self.global_encoder(pooled), "input_shape": tuple(left.shape[-2:])}

        def render(self, context, output_shape: tuple[int, int], tile_bounds: tuple[int, int, int, int]):
            height, width = output_shape
            top, bottom, x0, x1 = tile_bounds
            if not (0 <= top < bottom <= height and 0 <= x0 < x1 <= width):
                raise DisplayStudentError("Invalid display query tile bounds")
            features = context["fused" if aligned else "left"]
            y = (torch.arange(top, bottom, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
            x = (torch.arange(x0, x1, device=features.device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1)[None].expand(features.shape[0], -1, -1, -1)
            coordinates = grid.permute(0, 3, 1, 2)
            source_height, source_width = context["input_shape"]
            ratios = features.new_tensor([source_height / height, source_width / width])[None, :, None, None].expand(features.shape[0], 2, bottom-top, x1-x0)
            global_features = context["global"][:, :, None, None].expand(-1, -1, bottom-top, x1-x0)
            def sample(value, where):
                return F.grid_sample(value, where, mode="bilinear", padding_mode="border", align_corners=False)
            initial = [sample(context[key], grid) for key in (("fused",) if aligned else ("left", "right", "flow"))]
            descriptor = torch.cat([*initial, global_features, coordinates, ratios], dim=1)
            displacement = (.25 if aligned else 1.5) * torch.tanh(self.offset(descriptor)).permute(0, 2, 3, 1)
            if aligned:
                # One query coordinate samples one fused left-reference field;
                # no independent per-camera offsets can split an object edge.
                refined = sample(context["fused"], grid + displacement)
                return F.softplus(self.depth(torch.cat([refined, global_features, coordinates, ratios], dim=1))) + 1e-6
            # These learned per-query offsets are not an imposed homography or
            # teacher transport; output loss remains on the untouched display grid.
            refined = [sample(context["left"], grid + displacement[..., :2]),
                       sample(context["right"], grid + displacement[..., 2:]), initial[2]]
            return F.softplus(self.depth(torch.cat([*refined, global_features, coordinates, ratios], dim=1))) + 1e-6

    return StereoDisplayStudent()


def create_student(*, raft_root: Path | None, raft_model: Path | None, device: str = "auto",
                   train_scope: str = "update", iterations: int = 16, seed: int = 0,
                   units: str = "meters", quality: int = 1, raft_model_member: str | None = None):
    import torch
    from .spatial import RaftStereoOptions, resolve_raft_resources, _checkpoint_bytes, native_raft_checkpoint, _select_device
    if units not in UNITS or quality not in {0, 1, 2}:
        raise DisplayStudentError("Unsupported teacher units or quality")
    root, checkpoint, member = resolve_raft_resources(RaftStereoOptions(root=raft_root, model=raft_model, model_member=raft_model_member))
    data, name = _checkpoint_bytes(checkpoint, member)
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    raft_options, state, component = native_raft_checkpoint(payload, name, allow_display=True)
    raft_configuration = vars(raft_options)
    raft_configuration["mixed_precision"] = False
    configuration = {"architecture": ARCHITECTURE, "units": units, "channels": (16, 24, 32)[quality],
        "decoder_channels": (32, 64, 96)[quality], "iterations": iterations,
        "raft_configuration": raft_configuration, "input_reference": "native_stereo_pair",
        "output_reference": "display", "teacher_transport": "none",
        "stereo_alignment": "right_to_left_raft_reverse_consistency", "query_reference": "single_fused_left_field",
        "original_raft_checkpoint_sha256": hashlib.sha256(data).hexdigest()}
    torch.manual_seed(seed)
    model = _model(configuration, root)
    model.raft.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()}, strict=True)
    # A compatible aligned model can initialize a new optimizer while retaining
    # its learned decoder. Legacy models supply their own RAFT weights, but the
    # corrected fusion/query layers require fresh training rather than a relabel.
    if component == "embedded_raft_correspondence":
        previous = payload.get("architecture", payload.get("ipde_configuration", {}))
        if (previous.get("architecture") == ARCHITECTURE and previous.get("units") == units
                and previous.get("channels") == configuration["channels"]
                and previous.get("decoder_channels") == configuration["decoder_channels"]):
            model.load_state_dict(payload["state_dict"], strict=True)
    selected = _select_device(torch, device)
    model.to(device=selected, dtype=torch.float32); model.set_scope(train_scope)
    return model, configuration, selected


def load_student_checkpoint(payload: Mapping[str, Any] | Path | str, *, raft_root: Path | None,
                            device: str = "auto", train_scope: str = "update",
                            checkpoint_member: str | None = None, allow_legacy: bool = True):
    import torch
    from .spatial import RaftStereoOptions, resolve_raft_resources, _select_device, _checkpoint_bytes
    filename = None
    if not isinstance(payload, Mapping):
        filename = Path(payload).expanduser().resolve()
        try:
            data, _ = _checkpoint_bytes(filename, checkpoint_member)
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise DisplayStudentError(f"could not load selected display checkpoint {filename}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise DisplayStudentError("This is not a stereo-to-display depth checkpoint")
    configuration = payload.get("architecture", payload.get("ipde_configuration"))
    if not isinstance(configuration, Mapping) or payload.get("ipde_configuration", configuration) != configuration:
        raise DisplayStudentError("Display-depth model architecture metadata is inconsistent")
    if not allow_legacy and configuration.get("architecture") == LEGACY_ARCHITECTURE:
        raise DisplayStudentError("This checkpoint uses the previous unaligned decoder. Start a new training run for correspondence-aligned fusion; the old weights and optimizer cannot resume into the new architecture.")
    # Resolve only the compatible source tree; model weights are self-contained.
    if raft_root is None:
        if filename is None:
            raise DisplayStudentError("Supply the RAFT source directory when loading an in-memory model")
        raft_root, _, _ = resolve_raft_resources(RaftStereoOptions(model=filename, model_member=checkpoint_member))
    model = _model(configuration, Path(raft_root))
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state or any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise DisplayStudentError("Display-depth model weights are missing or nonfinite")
    expected = model.state_dict()
    if any(key in expected and value.dtype != expected[key].dtype for key, value in state.items()):
        raise DisplayStudentError("Display-depth model weights must retain the model's exact float32 and buffer dtypes")
    model.load_state_dict(state, strict=True)
    selected = _select_device(torch, device)
    model.to(device=selected, dtype=torch.float32); model.set_scope(train_scope)
    return model, dict(configuration), selected


def tile_bounds(output_shape: tuple[int, int], tile_size: int) -> Iterator[tuple[int, int, int, int]]:
    height, width = output_shape
    if min(height, width) < 1 or not 32 <= tile_size <= 2048:
        raise DisplayStudentError("Output dimensions must be positive and display tile size must be 32..2048")
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            yield top, min(top + tile_size, height), left, min(left + tile_size, width)


def predict_display_depth(left: np.ndarray, right: np.ndarray, output_shape: tuple[int, int], checkpoint: Path,
                          *, raft_root: Path | None = None, device: str = "auto", tile_size: int = 256,
                          checkpoint_member: str | None = None,
                          left_record: Mapping[str, Any] | None = None, right_record: Mapping[str, Any] | None = None):
    import torch
    model, configuration, selected = load_student_checkpoint(checkpoint, raft_root=raft_root, device=device,
                                                            checkpoint_member=checkpoint_member)
    model.eval()
    result = np.empty(output_shape, dtype=np.float32)
    with torch.inference_mode():
        context = model.encode(rgb_tensor(left, left_record or {}, torch, selected), rgb_tensor(right, right_record or {}, torch, selected))
        for bounds in tile_bounds(output_shape, tile_size):
            top, bottom, x0, x1 = bounds
            prediction = model.render(context, output_shape, bounds)
            if not bool(torch.isfinite(prediction).all()):
                raise DisplayStudentError("Display-depth model predicted nonfinite depth")
            result[top:bottom, x0:x1] = prediction[0, 0].cpu().numpy()
    return result, {"model": "RAFT stereo depth model", "schema": SCHEMA,
        "architecture": configuration, "reference_image": "display", "units": configuration["units"],
        "native_stereo_shape": list(np.shape(left)[:2]), "output_shape": list(output_shape), "device": selected,
        "input_resize": "none; native RGB copied into model tensors", "teacher_transport": "none",
        "output_resampling": "none; learned decoder evaluated at each display pixel", "tile_size": tile_size,
        "stereo_fusion": ("right aligned to left using RAFT correspondence and reverse-consistency masking; one fused field and one shared query" if configuration["architecture"] == ARCHITECTURE else "legacy independent left/right query features"),
        "accuracy_note": "Estimated teacher distillation; no measured accuracy or recovered missing detail claim"}
