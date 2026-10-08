"""Hardware-derived resource budgets, without disabling allocator safeguards.

All byte counts use binary GiB when displayed. CPU-only callers can import this
module without importing PyTorch or creating a Metal device.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from typing import Any

GIB = 1024 ** 3


def physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        # Linux/container fallback; do not assume a large machine.
        try:
            with open("/proc/meminfo", encoding="ascii") as source:
                match = re.search(r"^MemTotal:\s+(\d+)\s+kB", source.read(), re.MULTILINE)
            if match:
                return int(match[1]) * 1024
        except OSError:
            pass
        return 8 * GIB


def available_memory_bytes() -> int:
    """Reclaimable memory now, for optional caches and bounded worker queues."""
    physical = physical_memory_bytes()
    if sys.platform == "darwin":
        try:
            output = subprocess.run(["/usr/bin/vm_stat"], capture_output=True,
                                    text=True, check=True, timeout=5).stdout
            page = re.search(r"page size of (\d+) bytes", output)
            counts = [re.search(rf"^{name}:\s+(\d+)", output, re.MULTILINE)
                      for name in ("Pages free", "Pages inactive", "Pages speculative")]
            if page and all(counts):
                return min(physical, int(page[1]) * sum(int(value[1]) for value in counts))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        with open("/proc/meminfo", encoding="ascii") as source:
            match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", source.read(), re.MULTILINE)
        if match:
            return min(physical, int(match[1]) * 1024)
    except OSError:
        pass
    try:
        return min(physical, int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE")))
    except (ValueError, OSError, AttributeError):
        return physical // 4


def os_reserve_bytes(physical: int) -> int:
    # The half-memory bound matters only on machines smaller than 8 GiB.
    return min(physical // 2, max(4 * GIB, (physical + 9) // 10))


def training_resources(device: Any = None) -> dict[str, Any]:
    physical = physical_memory_bytes()
    reserve = os_reserve_bytes(physical)
    maximum = max(1, physical - reserve)
    recommended = None
    ratio = None
    kind = getattr(device, "type", device)
    if kind in (None, "mps"):
        import torch
        if torch.backends.mps.is_available():
            recommended = int(torch.mps.recommended_max_memory())
            ratio = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7"))
            if not math.isfinite(ratio) or ratio < 0 or ratio > 2:
                raise ValueError("PYTORCH_MPS_HIGH_WATERMARK_RATIO must be finite and between 0 and 2")
            # A zero environment ratio is not introduced or relied upon here;
            # an actual finite allocator limit is installed before training.
            if recommended > 0 and ratio > 0:
                maximum = min(maximum, int(recommended * ratio))
    default = min(maximum, physical * 8 // 10)
    if recommended:
        default = min(default, recommended)
    return {"physical_bytes": physical, "os_reserve_bytes": reserve,
            "mps_recommended_bytes": recommended, "mps_environment_high_watermark_ratio": ratio,
            "maximum_training_bytes": maximum, "default_training_bytes": default,
            "display_unit": "GiB", "bytes_per_gib": GIB}


def resolve_training_budget(requested: int | None, device: Any) -> tuple[int, dict[str, Any]]:
    report = training_resources(device)
    budget = report["default_training_bytes"] if requested is None else requested
    if type(budget) is not int or not 0 < budget <= report["maximum_training_bytes"]:
        maximum = report["maximum_training_bytes"] / GIB
        raise ValueError(f"Training memory must be a positive integer byte count up to this machine's {maximum:.2f} GiB practical limit")
    report["selected_training_bytes"] = budget
    return budget, report


def configure_training_resources(budget: int, device: Any) -> dict[str, Any]:
    """Use CPU cores for native preparation and enforce the selected GPU limit."""
    import torch
    cores = os.cpu_count() or 1
    torch.set_num_threads(cores)
    report = {"cpu_logical_cores": cores, "torch_cpu_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "mps_allocator_fraction": None}
    if getattr(device, "type", device) == "mps":
        recommended = int(torch.mps.recommended_max_memory())
        if recommended <= 0:
            raise ValueError("Metal did not report a usable recommended working set")
        fraction = budget / recommended
        if not 0 < fraction <= 2:
            raise ValueError("Selected memory exceeds the supported Metal allocator fraction")
        torch.mps.set_per_process_memory_fraction(float(fraction))
        report["mps_allocator_fraction"] = fraction
    return report


def decoded_cache_budget(budget: int, *, available: int | None = None,
                         physical: int | None = None) -> int:
    """Keep a bounded CPU cache while leaving most selected memory for training."""
    physical = physical_memory_bytes() if physical is None else physical
    available = available_memory_bytes() if available is None else available
    reclaimable = max(0, available - os_reserve_bytes(physical))
    return min(int(budget * 0.35), reclaimable // 2)
