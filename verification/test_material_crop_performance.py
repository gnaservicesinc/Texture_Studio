"""Native PNG precision, bounded scheduling and small automatic check policy."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_native_size as native
from material_dataset import read_png, write_png, rectangles_overlap
from material_validation_policy import POLICY, plan

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
    monkeypatch.setattr(native, "available_memory", lambda:64*1024**3)
    assert native.worker_count(source, 2048) == 8
    assert native.worker_count(source, 2048, 2) == 2
    monkeypatch.setattr(native, "available_memory", lambda:512*1024**2)
    assert native.worker_count(source, 2048) == 1
