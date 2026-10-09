#!/usr/bin/env python3
"""Native material training diagnostics and the supported training CLI.

The executable delegates to the same LoRA bridge used by Texture Studio. There
are no readers for discarded development checkpoint formats or hidden crops.
"""
from __future__ import annotations

import math
import os


def validate_training_configuration(size: int, cache_gib: float, scope: str = "final-map") -> None:
    """Validate map and cache contracts without inspecting machine memory."""
    if scope not in ("final-map", "map-decoder"):
        raise ValueError("Choose final-map or map-decoder refinement scope")
    if size < 256 or size % 64:
        raise ValueError("Choose a complete training grid of at least 256 pixels, divisible by 64; no padding or resizing")
    if not math.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("CPU cache budget must be finite and nonnegative")


def training_memory_plan(size: int, cache_gib: float, scope: str = "final-map") -> dict:
    """Optional historical diagnostics; these estimates never admit a run."""
    validate_training_configuration(size, cache_gib, scope)
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
        basis = "Measured1024 native map-decoder envelope34.28GiB; larger map-decoder grids are estimates without measured peaks."
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


def resource_plan(size: int, memory_gib: float | None, cache_gib: float, scope: str = "final-map", *, inference: bool = False) -> dict:
    """Describe a run without making memory estimates an admission policy.

    The legacy memory argument is retained for callers. Allocations are governed
    by the runtime allocator; cache size only controls optional decoded reuse.
    """
    validate_training_configuration(size, cache_gib, scope)
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    if inference:
        return {"physical_gib": physical, "cpu_cache_gib": cache_gib,
                "driver_estimate_gib": None, "combined_estimate_gib": None,
                "refinement_scope": None, "operation": "inference", "peak_measured": False,
                "memory_admission_enabled": False,
                "estimate_basis": "No-gradient inference; peak memory is unmeasured. Runtime allocation failures are reported by the selected device."}
    estimate = training_memory_plan(size, cache_gib, scope)
    driver_estimate = estimate["driver_estimate_gib"]
    combined = estimate["required_memory_gib"]
    return {**estimate, "physical_gib": physical, "cpu_cache_gib": cache_gib,
            "driver_estimate_gib": driver_estimate, "combined_estimate_gib": combined,
            "memory_admission_enabled": False,
            "refinement_scope": scope,
            "estimate_basis": estimate["estimate_basis"]}



def main(argv=None):
    from material_model_workbench import main as run
    return run(argv)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
