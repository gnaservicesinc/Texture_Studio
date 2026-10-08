"""Native PNG precision, bounded scheduling and small automatic check policy."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_native_size as native
import material_dataset as dataset
import material_resources as resources
from material_dataset import read_png, write_png, rectangles_overlap
from material_validation_policy import POLICY, plan


@pytest.mark.parametrize("sample_type", [np.uint8, np.dtype("<u2"), np.dtype(">u2")])
def test_pixel_hash_remains_canonical_for_strided_and_big_endian_maps(sample_type):
    original = np.arange(600, dtype=np.uint16).reshape(10, 20, 3).astype(sample_type)
    for array in (original, original[1:8, 2:14, :], original[:, :, ::-1]):
        expected = hashlib.sha256(np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False)).tobytes()).hexdigest()
        assert dataset.pixel_sha256(array) == expected

@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("compression", [0, 3, 6])
def test_light_compression_preserves_every_16bit_code_and_tags(tmp_path, channels, compression):
    array = np.random.default_rng(23).integers(0, 65536, (96, 128, channels), dtype=np.uint16)
    path = tmp_path / "numeric.png"
    write_png(path, array, compression=compression)
    decoded, header = read_png(path)
    np.testing.assert_array_equal(decoded, array)
    assert header["sample_bits"] == 16

def materials(count=97, rectangular=False):
    return {f"material_{i:03d}": {"dimensions": [4096, 2048 if rectangular else 4096],
            "samples": [], "maps": {"height": {"source": {"width":4096,"height":4096,"channels":1,"sample_bits":16}}}}
            for i in range(count)}

def test_automatic_five_percent_keeps_every_material_for_training():
    first = materials()
    policy = plan(first, 2048)
    second = materials()
    assert plan(second, 2048) == policy
    assert policy["policy"] == POLICY and len(policy["material_ids"]) == 5
    assert len(first) == 97
    assert sum(r["split"]=="train" for m in first.values() for r in m["regions"]) == 194
    assert sum(r["split"]=="validation" for m in first.values() for r in m["regions"]) == 5
    for material in first.values():
        for a in material["regions"]:
            for b in material["regions"]:
                if a["split"] != b["split"]: assert not rectangles_overlap(a["rectangle"], b["rectangle"])

def test_quick_fit_has_own_check_and_retains_other_materials():
    source = materials(20, rectangular=True)
    policy = plan(source, 2048, "material_019")
    assert policy["material_ids"] == ["material_019"]
    assert len(source) == 20
    assert [r["split"] for r in source["material_019"]["regions"]] == ["train", "validation"]
    assert all(r["split"] == "train" for r in source["material_000"]["regions"])

def test_excluded_check_region_is_not_randomly_reintroduced():
    source = materials(20)
    original = plan(copy.deepcopy(source), 1024)["material_ids"][0]
    source[original]["samples"] = [{"status":"excluded", "crop_rectangle_top_left_xywh":[1536,1536,1024,1024]}]
    assert original not in plan(source, 1024)["material_ids"]

def test_parallelism_is_bounded_by_cpu_and_available_memory(monkeypatch):
    source = materials(97)
    monkeypatch.setattr(native.os, "cpu_count", lambda:12)
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda:64*1024**3)
    monkeypatch.setattr(native, "available_memory", lambda:64*1024**3)
    assert native.worker_count(source, 2048) == 12
    assert native.worker_count(source, 2048, 2) == 2
    monkeypatch.setattr(native, "available_memory", lambda:512*1024**2)
    assert native.worker_count(source, 2048) == 1


def test_larger_hardware_uses_more_than_the_old_eight_workers_and_twelve_gib(monkeypatch):
    source = materials(97)
    for material in source.values():
        # The larger parent working set requires over 12 GiB for 24 workers.
        material["maps"]["normal"] = {"source": {"width":8192,"height":8192,"channels":4,"sample_bits":16}}
    monkeypatch.setattr(native.os, "cpu_count", lambda:24)
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda:128*1024**3)
    monkeypatch.setattr(native, "available_memory", lambda:100*1024**3)
    assert native.worker_count(source, 2048) == 24
    report = native.preparation_resources(source, 2048)
    assert report["working_memory_budget_bytes"] > 12*1024**3
    assert 24*report["estimated_worker_peak_bytes"] <= report["working_memory_budget_bytes"]
    assert native.worker_count(materials(3), 2048) == 3


def test_source_cache_uses_detected_headroom_without_a_512_mib_ceiling(monkeypatch):
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda:64*1024**3)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda:48*1024**3)
    automatic = dataset.SourceDecodeCache()
    assert automatic.max_bytes == 48*1024**3 - resources.os_reserve_bytes(64*1024**3)
    assert automatic.max_bytes > 512*1024**2
    assert dataset.SourceDecodeCache(2*1024**3).max_bytes == 2*1024**3
    assert dataset.SourceDecodeCache(100*1024**3).max_bytes == automatic.max_bytes
    assert dataset.SourceDecodeCache(0).max_bytes == 0
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            dataset.SourceDecodeCache(invalid)


def test_source_cache_cannot_retain_arrays_when_other_apps_consume_headroom(monkeypatch):
    monkeypatch.setattr(resources, "physical_memory_bytes", lambda:64*1024**3)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda:2*1024**3)
    assert dataset.SourceDecodeCache().max_bytes == 0
