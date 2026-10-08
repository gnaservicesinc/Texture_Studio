"""Strict rank-8 DINOv2 adapter preservation for frozen-encoder head refinement."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from frozen_dino_height import state_sha256

POLICY = "refine_material_head_with_fixed_rank8_attention_adapters"


def validate_adapters(adapters: dict) -> None:
    expected = {f"blocks.{i}.attn.{part}": (output, 768) for i in range(12) for part, output in (("qkv", 2304), ("proj", 768))}
    if not isinstance(adapters, dict) or set(adapters) != set(expected):
        raise ValueError("LoRA layer identities differ from rank-8 DINOv2 Base attention")
    for name, (outgoing, incoming) in expected.items():
        state = adapters[name]
        if not isinstance(state, dict) or set(state) != {"lora_A", "lora_B"}:
            raise ValueError("Unexpected LoRA tensor names")
        for part, shape in (("lora_A", (8, incoming)), ("lora_B", (outgoing, 8))):
            value = state[part]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Invalid LoRA tensor shape, precision or values")


def adapter_sha256(state: dict) -> str:
    validate_adapters(state)
    return state_sha256({path + "." + field: value for path, fields in state.items() for field, value in fields.items()})


def apply_fixed_adapters(encoder: torch.nn.Module, state: dict) -> dict:
    from probe_material_adapters import install_adapters
    validate_adapters(state)
    modules = install_adapters(encoder)
    with torch.no_grad():
        for name, module in modules.items():
            for field, value in state[name].items():
                getattr(module, field).copy_(value.to(getattr(module, field).device))
    encoder.requires_grad_(False).eval()
    actual = {name: {field: getattr(module, field).detach().cpu() for field in ("lora_A", "lora_B")} for name, module in modules.items()}
    if adapter_sha256(actual) != adapter_sha256(state):
        raise ValueError("Fixed adapters differ after encoder installation")
    return modules


def encoder_size(checkpoint: dict, path: Path) -> int:
    size = checkpoint.get("encoder_size")
    if size is None:
        run_path = path.parent.parent / "run.json"
        if not run_path.is_file():
            raise ValueError("Legacy adapted checkpoint needs matching run.json to resolve encoder size")
        run = json.loads(run_path.read_text())
        dimensions = run.get("encoder_dimensions")
        if (run.get("schema") != checkpoint.get("schema") or not isinstance(dimensions, list) or len(dimensions) != 2 or dimensions[0] != dimensions[1]
                or not checkpoint.get("schedule_sha256") or checkpoint.get("schedule_sha256") != run.get("schedule_sha256")
                or not checkpoint.get("selected_files_sha256") or checkpoint.get("selected_files_sha256") != run.get("selected_files_sha256")):
            raise ValueError("Legacy adapted checkpoint run.json does not match its source/schedule identity")
        size = dimensions[0]
    if type(size) is not int or not 28 <= size <= 518 or size % 14:
        raise ValueError("Fixed-adapter checkpoint requires a bounded, aligned encoder size")
    return size
