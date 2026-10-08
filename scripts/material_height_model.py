"""Small, native-resolution material-height network (GPL-3.0-or-later).

This is a supervised material model, not a DA3 checkpoint or a LoRA. It takes
linear RGB reflectance and predicts relative source height in [0, 1]. No output
min/max stretching is performed. The first encoder and final decoder operate at
the full input resolution; pooled features supply context rather than replacing
the native-pixel prediction.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "texture-studio-material-height-v1"
ARCHITECTURE = "native-height-unet-v1"


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(4 if outgoing % 4 == 0 else 1, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(4 if outgoing % 4 == 0 else 1, outgoing),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MaterialHeightNet(nn.Module):
    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four >= 4")
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 6)
        self.encoder = nn.ModuleList(
            ConvBlock(incoming, outgoing)
            for incoming, outgoing in zip((3,) + widths[:-1], widths)
        )
        self.decoder = nn.ModuleList(
            ConvBlock(widths[level + 1] + widths[level], widths[level])
            for level in (2, 1, 0)
        )
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected batch × RGB × height × width")
        skips = []
        value = rgb
        for level, block in enumerate(self.encoder):
            if level:
                value = F.avg_pool2d(value, 2)
            value = block(value)
            skips.append(value)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return torch.sigmoid(self.head(value))


def weighted_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    return (value * mask).sum() / mask.sum().clamp_min(1)


def weighted_center(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value - value.mean(dim=(-2, -1), keepdim=True)
    mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    return value - mean


def detail_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else 1 - F.max_pool2d(1 - mask, 9, stride=1, padding=4)


def multiscale_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Float32 finite differences at native, 2, 4, and 8-pixel scales."""
    prediction, target = prediction.float(), target.float()
    terms = []
    for scale in (1, 2, 4, 8):
        if min(target.shape[-2:]) < scale * 2:
            continue
        p = prediction if scale == 1 else F.avg_pool2d(prediction, scale)
        t = target if scale == 1 else F.avg_pool2d(target, scale)
        valid = mask if scale == 1 or mask is None else 1 - F.max_pool2d(1 - mask, scale)
        mx = None if valid is None else valid[..., :, 1:] * valid[..., :, :-1]
        my = None if valid is None else valid[..., 1:, :] * valid[..., :-1, :]
        terms.append(weighted_mean((p[..., :, 1:] - p[..., :, :-1] - t[..., :, 1:] + t[..., :, :-1]).abs(), mx))
        terms.append(weighted_mean((p[..., 1:, :] - p[..., :-1, :] - t[..., 1:, :] + t[..., :-1, :]).abs(), my))
    return torch.stack(terms).mean()


def height_loss(prediction: torch.Tensor, target: torch.Tensor, gradient_weight: float = 8.0, offset_invariant: bool = False, highpass_weight: float = 0.0, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, target = prediction.float(), target.float()
    raw_absolute = weighted_mean((prediction - target).abs(), mask)
    centered = weighted_mean((weighted_center(prediction, mask) - weighted_center(target, mask)).abs(), mask)
    absolute = centered if offset_invariant else raw_absolute
    gradient = multiscale_gradient_loss(prediction, target, mask)
    detail_p = prediction - F.avg_pool2d(F.pad(prediction, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    detail_t = target - F.avg_pool2d(F.pad(target, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    detail = weighted_mean((detail_p - detail_t).abs(), detail_mask(mask))
    total = absolute + gradient_weight * gradient + highpass_weight * detail
    return total, {"height_mae": float(raw_absolute.detach()), "height_loss_mae": float(absolute.detach()), "centered_height_mae": float(centered.detach()), "multiscale_gradient_mae": float(gradient.detach()), "detail_highpass_mae_radius_4": float(detail.detach())}


def squared_objective(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Relative squared error changes loss units, never source height samples.

    Fixed target energies balance centered height, gradients and high-pass
    detail. A flat output scores approximately one per informative component.
    Floors keep nearly flat maps from amplifying quantization noise endlessly.
    """
    prediction, target = prediction.float(), target.float()
    pcenter, tcenter = weighted_center(prediction, mask), weighted_center(target, mask)
    height_energy = weighted_mean(tcenter.square(), mask).detach().clamp_min(1e-6)
    height_mse = weighted_mean((pcenter - tcenter).square(), mask)
    gradients, gradient_mses, gradient_energies = [], [], []
    for scale in (1, 2, 4, 8):
        if min(target.shape[-2:]) < scale * 2:
            continue
        p = prediction if scale == 1 else F.avg_pool2d(prediction, scale)
        t = target if scale == 1 else F.avg_pool2d(target, scale)
        valid = mask if scale == 1 or mask is None else 1 - F.max_pool2d(1 - mask, scale)
        for axis in ("x", "y"):
            if axis == "x":
                pg, tg = p[..., :, 1:] - p[..., :, :-1], t[..., :, 1:] - t[..., :, :-1]
                pair_mask = None if valid is None else valid[..., :, 1:] * valid[..., :, :-1]
            else:
                pg, tg = p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]
                pair_mask = None if valid is None else valid[..., 1:, :] * valid[..., :-1, :]
            mse = weighted_mean((pg - tg).square(), pair_mask)
            energy = weighted_mean(tg.square(), pair_mask).detach().clamp_min(1e-7)
            gradients.append(mse / energy)
            gradient_mses.append(mse)
            gradient_energies.append(energy)
    gradient_loss = torch.stack(gradients).mean()
    phigh = prediction - F.avg_pool2d(F.pad(prediction, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    thigh = target - F.avg_pool2d(F.pad(target, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    high_mask = detail_mask(mask)
    high_energy = weighted_mean(thigh.square(), high_mask).detach().clamp_min(1e-7)
    high_mse = weighted_mean((phigh - thigh).square(), high_mask)
    total = height_mse / height_energy + gradient_loss + high_mse / high_energy
    return total, {
        "centered_height_mse": float(height_mse.detach()),
        "target_centered_height_energy": float(height_energy),
        "relative_centered_height_mse": float((height_mse / height_energy).detach()),
        "mean_gradient_mse": float(torch.stack(gradient_mses).mean().detach()),
        "mean_target_gradient_energy": float(torch.stack(gradient_energies).mean()),
        "relative_multiscale_gradient_mse": float(gradient_loss.detach()),
        "detail_highpass_mse_radius_4": float(high_mse.detach()),
        "target_highpass_energy": float(high_energy),
        "relative_detail_highpass_mse": float((high_mse / high_energy).detach()),
    }
