#!/usr/bin/env python3
"""Native material training resource policy and the supported training CLI.

The executable delegates to the same LoRA bridge used by Texture Studio. There
are no readers for discarded development checkpoint formats or hidden crops.
"""
from __future__ import annotations

import math
import os


def training_memory_plan(size: int, cache_gib: float, scope: str = "final-map") -> dict:
    """Measured complete-grid allowance, independent of the chosen budget."""
    if scope not in ("final-map", "map-decoder"):
        raise ValueError("Choose final-map or map-decoder refinement scope")
    if size < 256 or size % 64:
        raise ValueError("Choose a complete training grid of at least 256 pixels, divisible by 64; no padding or resizing")
    if not math.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("CPU cache budget must be finite and nonnegative")
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    if scope == "final-map":
        # Genuine float32 rank-8 LoRA, complete2048 height forward/backward/
        # AdamW on this64GiB M2 Max: peak driver42.3448GiB, successful under
        #45GiB positive Metal cap. Outer RRDB checkpoints retain exact math.
        # Three consecutive1024 updates sampled19.7359GiB driver peak; reserve
        #21GiB. Driver accounting includes cached/private Metal workspaces and
        # can slightly exceed the allocator cap itself. Neither RSS nor the
        # cache should be added to driver usage as a measured physical peak.
        driver = (2 + 19 * (size / 1024) ** 2) if size <= 1024 else 2 + 43 * (size / 2048) ** 2
        basis = ("Measured float32 final-map LoRA with RRDB checkpoints: three1024 updates sampled19.7359GiB driver peak (21GiB allowance); one2048 forward/backward/AdamW sampled42.3448GiB (45GiB allowance). Other grids are estimates, not peak guarantees.")
    else:
        driver = 2 + 34 * (size / 1024) ** 2
        basis = "Measured1024 native map-decoder envelope34.28GiB;2048 map-decoder LoRA has not been qualified and remains conservatively excluded on64GiB hardware."
    required = driver + cache_gib + 2
    maximum = max(0, physical - 4)
    return {"required_memory_gib": required, "recommended_memory_gib": min(maximum, math.ceil(required)),
            "driver_estimate_gib": driver, "cpu_cache_gib": cache_gib, "runtime_reserve_gib": 2,
            "physical_gib": physical, "maximum_memory_gib": maximum,
            "hardware_supported": required <= maximum, "refinement_scope": scope,
            "qualified_native_2048": scope == "final-map" and size == 2048,
            "measured_native_2048_peak_driver_gib": 42.34480285644531 if scope == "final-map" else None,
            "measured_native_1024_peak_driver_gib": 19.735946655273438 if scope == "final-map" else None,
            "training_activation_checkpointing": "rrdb-block-v1", "estimate_basis": basis}


def resource_plan(size: int, memory_gib: float, cache_gib: float, scope: str = "final-map", *, inference: bool = False) -> dict:
    if scope not in ("final-map", "map-decoder"):
        raise ValueError("Choose final-map or map-decoder refinement scope")
    if size < 256 or size % 64:
        raise ValueError("Choose a complete training grid of at least 256 pixels, divisible by 64; no padding or resizing")
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    if not math.isfinite(memory_gib) or memory_gib <= 0 or memory_gib > physical - 4:
        operation = "Inference" if inference else "Training"
        raise ValueError(f"{operation} budget must fit this Mac ({physical:.0f}GiB total,4GiB for macOS)")
    if not math.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("CPU cache budget must be finite and nonnegative")
    if inference:
        # A no-gradient forward does not retain training activations or allocate
        # optimizer state. Do not reject it using a measured backward-pass peak,
        # or invent a lower peak before measuring this architecture at this grid.
        # The positive MPS allocation cap remains authoritative for a real probe.
        driver_budget = memory_gib - cache_gib - 2
        if driver_budget < 1:
            raise ValueError("Inference budget must leave at least 1 GiB for Metal after the CPU cache and 2 GiB runtime reserve")
        return {"physical_gib": physical, "budget_gib": memory_gib, "cpu_cache_gib": cache_gib,
                "driver_budget_gib": driver_budget, "runtime_reserve_gib": 2,
                "driver_estimate_gib": None, "combined_estimate_gib": None,
                "refinement_scope": None, "operation": "inference", "peak_measured": False,
                "estimate_basis": "No-gradient inference; peak memory is unmeasured. The positive Metal allocation cap is retained; a native probe can still exhaust its budget."}
    estimate = training_memory_plan(size, cache_gib, scope)
    driver_estimate = estimate["driver_estimate_gib"]
    combined = estimate["required_memory_gib"]
    if combined > memory_gib:
        raise ValueError(f"Native{size} training estimates{combined:.1f}GiB above the{memory_gib:.1f}GiB budget. "
                         "Choose a smaller complete training grid and prepare that grid from the originals. No padding/resizing is used.")
    return {**estimate, "physical_gib": physical, "budget_gib": memory_gib, "cpu_cache_gib": cache_gib,
            "driver_estimate_gib": driver_estimate, "combined_estimate_gib": combined,
            "refinement_scope": scope,
            "estimate_basis": estimate["estimate_basis"]}



def main(argv=None):
    from material_model_workbench import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
