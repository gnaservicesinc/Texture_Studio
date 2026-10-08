"""Hardware budgets and actual native-pair reuse keep full numeric precision."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_resources as resources
import material_training_cycle as cycle


def mocked_machine(monkeypatch, physical_gib=64, recommended_gib=51.84):
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda: physical_gib * resources.GIB)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: int(recommended_gib * resources.GIB))
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)


def test_64_gib_machine_default_and_maximum_use_most_unified_memory(monkeypatch):
    mocked_machine(monkeypatch)
    report = resources.training_resources(torch.device("mps"))
    assert report["maximum_training_bytes"] == int(64 * resources.GIB) - __import__("math").ceil(6.4 * resources.GIB)
    assert report["default_training_bytes"] == int(51.2 * resources.GIB)
    requested = 56 * resources.GIB
    selected, report = resources.resolve_training_budget(requested, torch.device("mps"))
    assert selected == requested and selected > 30_000_000_000
    assert report["display_unit"] == "GiB"


@pytest.mark.parametrize("gib", (16, 32, 128))
def test_limits_scale_with_physical_memory_and_leave_os_headroom(monkeypatch, gib):
    mocked_machine(monkeypatch, gib, gib * 0.81)
    report = resources.training_resources(torch.device("mps"))
    assert report["maximum_training_bytes"] == gib * resources.GIB - resources.os_reserve_bytes(gib * resources.GIB)
    assert report["default_training_bytes"] <= report["maximum_training_bytes"]
    assert report["maximum_training_bytes"] >= gib * resources.GIB * 0.75


def test_configured_allocator_watermark_is_respected_and_never_disabled(monkeypatch):
    mocked_machine(monkeypatch)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
    report = resources.training_resources(torch.device("mps"))
    assert report["maximum_training_bytes"] == int(int(51.84 * resources.GIB) * .8)
    with pytest.raises(ValueError, match="practical limit"):
        resources.resolve_training_budget(50 * resources.GIB, torch.device("mps"))
    seen = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", seen.append)
    selected = report["default_training_bytes"]
    configured = resources.configure_training_resources(selected, torch.device("mps"))
    assert seen == [selected / int(51.84 * resources.GIB)]
    assert 0 < seen[0] <= 0.8
    assert configured["torch_cpu_threads"] == __import__("os").cpu_count()


def test_zero_environment_watermark_still_gets_finite_selected_allocator_limit(monkeypatch):
    mocked_machine(monkeypatch)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0")
    seen = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", seen.append)
    selected, _ = resources.resolve_training_budget(None, torch.device("mps"))
    resources.configure_training_resources(selected, torch.device("mps"))
    assert 0 < seen[0] < 2


@pytest.mark.parametrize("requested", (-1, 0, True, 1.2, 64 * resources.GIB))
def test_bad_or_overphysical_explicit_budget_rejected_before_allocation(monkeypatch, requested):
    mocked_machine(monkeypatch)
    with pytest.raises(ValueError, match="practical limit"):
        resources.resolve_training_budget(requested, torch.device("mps"))


def test_cpu_budget_does_not_initialize_metal(monkeypatch):
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda: 64 * resources.GIB)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: (_ for _ in ()).throw(AssertionError("No GPU query for CPU")))
    selected, report = resources.resolve_training_budget(None, torch.device("cpu"))
    assert selected == int(51.2 * resources.GIB)
    assert report["mps_recommended_bytes"] is None


def test_decoded_cache_scales_but_yields_when_other_apps_need_memory():
    assert resources.decoded_cache_budget(50 * resources.GIB, physical=64 * resources.GIB,
                                          available=50 * resources.GIB) == int(17.5 * resources.GIB)
    assert resources.decoded_cache_budget(50 * resources.GIB, physical=64 * resources.GIB,
                                          available=6 * resources.GIB) == 0
    assert resources.decoded_cache_budget(2 * resources.GIB, physical=64 * resources.GIB,
                                          available=40 * resources.GIB) == int(.7 * resources.GIB)


def test_native_pair_cache_reuses_exact_values_without_extra_decodes_and_evicts():
    decoded = Counter()
    def loader(sample):
        identity = sample["metadata"]["sample_id"]
        decoded[identity] += 1
        raw = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 65535
        return raw, raw[:, :1].clone(), None
    sample = lambda name: {"metadata": {"sample_id": name}}
    cache = cycle.NativePairCache(loader, 128)
    first = cache.get(sample("one"), torch.device("cpu"))
    second = cache.get(sample("one"), torch.device("cpu"))
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert first[0].data_ptr() == second[0].data_ptr()
    cache.get(sample("two"), torch.device("cpu"))
    cache.get(sample("one"), torch.device("cpu"))  # Keep one most recent.
    cache.get(sample("three"), torch.device("cpu"))
    assert list(cache.pairs) == ["one", "three"]
    assert cache.bytes == 128 and decoded == {"one": 1, "two": 1, "three": 1}
    cache.get(sample("two"), torch.device("cpu"))
    assert decoded["two"] == 2 and cache.bytes <= cache.maximum_bytes
    report = cache.report()
    assert not report["prefilled"] and report["native_values_unchanged"]


def test_small_budget_does_not_cache_oversized_pair():
    source = torch.zeros(1, 3, 32, 32)
    cache = cycle.NativePairCache(lambda _sample: (source, source[:, :1], None), 1)
    sample = {"metadata": {"sample_id": "large"}}
    cache.get(sample, torch.device("cpu"))
    cache.get(sample, torch.device("cpu"))
    assert cache.bytes == 0 and cache.misses == 2


def test_soft_guard_accounts_cpu_caches_plus_metal_driver(monkeypatch):
    monkeypatch.setattr(cycle, "memory", lambda _device: {"mps_driver_bytes": 60})
    guard = cycle.Guard(torch.device("mps"), 1, 100)
    guard.retained_cpu_bytes = lambda: 41
    with pytest.raises(cycle.CycleLimit, match="memory soft guard"):
        guard.check("test")
    assert guard.peak["tracked_unified_bytes"] == 101


def test_cache_is_discarded_before_memory_guard_stops_useful_training(monkeypatch):
    pair = (torch.zeros(16, dtype=torch.float32), None, None)
    cache = cycle.NativePairCache(lambda _sample: pair, 128)
    cache.get({"metadata": {"sample_id": "one"}}, torch.device("cpu"))
    cache.get({"metadata": {"sample_id": "two"}}, torch.device("cpu"))
    monkeypatch.setattr(cycle, "memory", lambda _device: {"mps_driver_bytes": 100})
    guard = cycle.Guard(torch.device("mps"), 1, 180)
    guard.retained_cpu_bytes = lambda: cache.bytes
    guard.reclaim_cpu_bytes = cache.reclaim
    guard.check("growing GPU activations")
    assert cache.bytes == 64 and cache.pressure_evictions == 1
    assert guard.peak["tracked_unified_bytes"] == 164


def test_actual_cycle_decodes_once_per_crop_and_keeps_schedule(tmp_path, monkeypatch):
    from test_material_training_cycle import arguments, run
    args = arguments(tmp_path)
    monkeypatch.setattr(cycle, "decoded_cache_budget", lambda _budget: resources.GIB)
    original = cycle.load_target_pair
    calls = Counter()
    def checked_decode(sample, *settings):
        calls[sample["metadata"]["sample_id"]] += 1
        return original(sample, *settings)
    monkeypatch.setattr(cycle, "load_target_pair", checked_decode)
    report = run(args)
    assert report["complete_schedule"] and report["completed_steps"] == 4
    assert report["decoded_pair_cache"]["decodes"] == 3
    assert report["decoded_pair_cache"]["hits"] == 12
    # Preflight checks each crop once, then the feature pass lazily decodes it
    # once; repeated evaluation, updates, and exports all reuse exact tensors.
    assert calls == {"white_stucco_02_auto_001": 2, "white_stucco_02_auto_002": 2, "white_stucco_02_auto_003": 2}
    latest = torch.load(args.output / "checkpoint.latest.pt", weights_only=True)
    assert latest["schedule"] == cycle.balanced_schedule(2, 2, args.seed)
    assert report["target_precision_bits"] == {name: 16 for name in calls}
