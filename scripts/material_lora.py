"""Exact low-rank updates for the existing material network's learned layers.

Only adapters receive gradients. Exports use safetensors, bind to an exact base
identity, and retain the adapter even when a complete fused model is requested.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load as load_bytes, save_file

SCHEMA = "texture-studio-material-lora-v1"
FULL_SCHEMA = "texture-studio-material-checkpoint-v1"
TARGET_BRANCH = {"normal": 1, "roughness": 2, "height": 3}


def prefixes(target: str, scope: str) -> tuple[str, ...]:
    if target not in TARGET_BRANCH or scope not in ("final-map", "map-decoder"):
        raise ValueError("Choose height, roughness or normal and final-map or map-decoder")
    index = TARGET_BRANCH[target]
    return ((f"gen.m_dec_{index}.", f"gen.m_tail_{index}.") if scope == "map-decoder" else ()) + (f"ups.{index}.",)


class LowRankLayer(nn.Module):
    def __init__(self, base: nn.Conv2d | nn.Linear, rank: int, alpha: float):
        super().__init__()
        if isinstance(base, nn.Conv2d) and base.groups != 1:
            raise ValueError("Grouped convolutions cannot use this material adapter")
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.weight[0].numel()))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.weight.shape[0], self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).reshape_as(self.base.weight) * self.scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Use the identical weight arithmetic as fusion. No lossy factorization
        # or image transformation is introduced during export.
        weight = self.base.weight + self.delta()
        if isinstance(self.base, nn.Conv2d):
            return self.base._conv_forward(value, weight, self.base.bias)
        return F.linear(value, weight, self.base.bias)


def layers(model: nn.Module) -> dict[str, LowRankLayer]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LowRankLayer)}


def install(model: nn.Module, target: str = "height", scope: str = "final-map", rank: int = 8,
            alpha: float = 8, specifications: dict | None = None) -> list[nn.Parameter]:
    selected = prefixes(target, scope)
    candidates = {name: module for name, module in model.named_modules()
                  if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in selected)
                  and isinstance(module, (nn.Conv2d, nn.Linear))
                  and (not isinstance(module, nn.Conv2d) or module.groups == 1)}
    if not candidates:
        raise ValueError("The base model has no pretrained output layers for this target")
    if specifications is not None and set(candidates) != set(specifications):
        raise ValueError("Adapter layer identities differ from this material architecture")
    if specifications is not None and any(list(base.weight.shape) != specifications[name]["weight_shape"]
                                           for name, base in candidates.items()):
        raise ValueError("Adapter weight dimensions differ from this base")
    model.requires_grad_(False).eval()
    parameters = []
    for name, base in candidates.items():
        spec = specifications[name] if specifications is not None else {"rank": rank, "alpha": alpha}
        module = LowRankLayer(base, spec["rank"], spec["alpha"])
        if specifications is not None and (list(base.weight.shape) != spec["weight_shape"] or module.rank != spec["rank"]):
            raise ValueError("Adapter rank or weight dimensions differ from this base")
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, module)
        parameters.extend((module.lora_A, module.lora_B))
    return parameters


def adapter_state(model: nn.Module) -> tuple[dict, dict]:
    tensors, specifications = {}, {}
    for name, module in layers(model).items():
        specifications[name] = {"rank": module.rank, "alpha": module.alpha,
                                "weight_shape": list(module.base.weight.shape)}
        for field in ("lora_A", "lora_B"):
            tensors[name + "." + field] = getattr(module, field).detach().cpu().contiguous()
    if not tensors:
        raise ValueError("A material export must contain trained LoRA layers")
    return tensors, specifications


def fused_state(model: nn.Module) -> dict[str, torch.Tensor]:
    adapters = layers(model)
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        translated = name
        for layer_name, module in adapters.items():
            marker = layer_name + ".base."
            if name.startswith(marker):
                translated = layer_name + "." + name[len(marker):]
                if translated == layer_name + ".weight":
                    tensor = module.base.weight + module.delta()
                break
        state[translated] = tensor.detach().cpu().contiguous()
    return state


def save_tensors(path: Path, tensors: dict, configuration: dict) -> None:
    if not tensors or any(not isinstance(value, torch.Tensor) or
                          (value.is_floating_point() and not torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Export tensors must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, name,
                  metadata={"configuration": json.dumps(configuration, sort_keys=True, allow_nan=False)})
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read(path: Path) -> tuple[dict, dict]:
    return read_bytes(path.read_bytes())


def read_bytes(raw: bytes) -> tuple[dict, dict]:
    if len(raw) < 8:
        raise ValueError("Material safetensors checkpoint is truncated")
    length = int.from_bytes(raw[:8], "little")
    if length > len(raw) - 8:
        raise ValueError("Material safetensors header is truncated")
    header = json.loads(raw[8:8 + length])
    configuration = json.loads(header.get("__metadata__", {}).get("configuration", "null"))
    tensors = load_bytes(raw)
    if (not isinstance(configuration, dict) or configuration.get("schema") not in (SCHEMA, FULL_SCHEMA)
            or configuration.get("architecture") != "pbrnxt-native-v1"
            or configuration.get("target") not in TARGET_BRANCH
            or type(configuration.get("step")) is not int or configuration["step"] < 0
            or not isinstance(configuration.get("base"), dict)
            or not isinstance(configuration["base"].get("sha256"), str)
            or len(configuration["base"]["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in configuration["base"]["sha256"])
            or configuration.get("image_padding") is not False or configuration.get("image_resizing") is not False
            or type(configuration.get("training_size")) is not int or not 256 <= configuration["training_size"] <= 8192
            or configuration["training_size"] % 64
            or not tensors or any(value.is_floating_point() and not torch.isfinite(value).all() for value in tensors.values())):
        raise ValueError("Expected a complete recorded material safetensors checkpoint")
    if configuration["schema"] == SCHEMA:
        specs = configuration.get("layers", {})
        if not isinstance(specs, dict) or any(not isinstance(name, str) or not isinstance(spec, dict) for name, spec in specs.items()):
            raise ValueError("Adapter requires a recorded material layer specification")
        expected = {name + "." + field for name in specs for field in ("lora_A", "lora_B")}
        if not specs or set(tensors) != expected:
            raise ValueError("Adapter tensor identities do not match its configuration")
        for name, spec in specs.items():
            shape, rank = spec.get("weight_shape"), spec.get("rank")
            if (not isinstance(shape, list) or len(shape) not in (2, 4) or any(type(v) is not int or v < 1 for v in shape)
                    or type(rank) is not int or rank < 1
                    or not isinstance(spec.get("alpha"), (int, float)) or not math.isfinite(spec["alpha"]) or spec["alpha"] <= 0
                    or tuple(tensors[name + ".lora_A"].shape) != (rank, math.prod(shape[1:]))
                    or tuple(tensors[name + ".lora_B"].shape) != (shape[0], rank)
                    or any(tensors[name + "." + f].dtype != torch.float32 for f in ("lora_A", "lora_B"))):
                raise ValueError("Adapter contains invalid layer dimensions or numeric values")
        selected = prefixes(configuration["target"], configuration.get("scope"))
        if any(not any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in selected) for name in specs):
            raise ValueError("Adapter contains layers outside its recorded material target")
    return configuration, tensors


def restore(model: nn.Module, configuration: dict, tensors: dict) -> None:
    install(model, configuration["target"], configuration["scope"], specifications=configuration["layers"])
    with torch.no_grad():
        for name, module in layers(model).items():
            for field in ("lora_A", "lora_B"):
                getattr(module, field).copy_(tensors[name + "." + field].to(module.base.weight.device))


def combine(adapters: list[tuple[dict, dict, float]]) -> tuple[dict, dict]:
    """Concatenate factors so the new adapter is the exact weighted delta sum.

    Ranks can differ. Architecture, target, module set, and exact base must agree.
    There is no SVD truncation, averaging of A/B separately, or lost adapter.
    """
    first = adapters[0][0]
    for configuration, _tensors, weight in adapters:
        if (configuration["schema"] != SCHEMA or any(configuration[key] != first[key] for key in ("architecture", "target", "scope"))
                or configuration["base"]["sha256"] != first["base"]["sha256"]
                or set(configuration["layers"]) != set(first["layers"]) or not math.isfinite(weight)):
            raise ValueError("Mix adapters trained against the same exact base, target and layer set")
    result, tensors = dict(first), {}
    result["layers"] = {}
    for name, spec in first["layers"].items():
        if any(configuration["layers"][name]["weight_shape"] != spec["weight_shape"] for configuration, _, _ in adapters):
            raise ValueError("Adapter layer shapes differ")
        rank = sum(configuration["layers"][name]["rank"] for configuration, _, _ in adapters)
        # A concatenated exact update may exceed the layer's minimum dimension;
        # installation supports that redundant rank without compressing it.
        result["layers"][name] = dict(spec, rank=rank, alpha=float(rank))
        tensors[name + ".lora_A"] = torch.cat([values[name + ".lora_A"] for _, values, _ in adapters], dim=0)
        tensors[name + ".lora_B"] = torch.cat([values[name + ".lora_B"] * (weight * config["layers"][name]["alpha"] / config["layers"][name]["rank"])
                                               for config, values, weight in adapters], dim=1)
    result["mixture_weights"] = [weight for _, _, weight in adapters]
    return result, tensors
