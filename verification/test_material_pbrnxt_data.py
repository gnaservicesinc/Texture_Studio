"""Native paired pixels, source precision, cache identity and relief detail."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import material_pbrnxt_data as data
from material_dataset import read_png, write_png


def add_sample(dataset, sample_id="surface_001", material="surface", split="train", status="approved",
               dimensions=(128, 128), origin=(0, 0), base=16384):
    width, height = dimensions
    yy, xx = np.indices((height, width))
    rgb = np.stack((xx * 257, yy * 257, (xx + yy) * 128), axis=-1).astype(np.uint16)
    codes = (base + yy * width + xx).astype(np.uint16)[..., None]
    folder = dataset / "samples" / sample_id
    folder.mkdir(parents=True)
    maps, map_metadata = {}, {}
    for role, values in (("input", rgb), ("height", codes)):
        path = folder / f"{role}.png"
        write_png(path, values)
        maps[role] = path.name
        map_metadata[role] = {"sample_sha256": data.digest(path), "sample_bits": 16,
                              "channels": values.shape[-1],
                              "encoding": "source_srgb_assumed" if role == "input" else "linear_data"}
    record = {"sample_id": sample_id, "material_id": material, "split": split, "status": status,
              "source_precision_verified": True, "crop_values_verified": True,
              "sample_pixel_dimensions": [width, height], "source_pixel_dimensions": [512, 512],
              "crop_rectangle_top_left_xywh": [*origin, width, height],
              "maps": maps, "map_metadata": map_metadata}
    (folder / "sample.json").write_text(json.dumps(record))
    index_path = dataset / "dataset.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"samples": []}
    index["samples"].append({key: record[key] for key in ("sample_id", "material_id", "split", "status")}
                            | {"path": str(folder.relative_to(dataset))})
    index_path.write_text(json.dumps(index))
    return folder, rgb, codes


def mutate_metadata(folder, **updates):
    path = folder / "sample.json"
    payload = json.loads(path.read_text())
    payload.update(updates)
    path.write_text(json.dumps(payload))
    return payload


def test_adjacent_uint16_height_codes_survive_numeric_transfer_and_exact_native_crop(tmp_path):
    folder, native_rgb, native_height = add_sample(tmp_path)
    before = {path: data.digest(path) for path in folder.iterdir()}
    training, checks, identity = data.select_pairs(tmp_path)
    assert checks == [] and identity["source_bits"] == 16
    cache = data.PairCache(1024 * 1024)
    rgb, height = cache.load(training[0])
    assert rgb.dtype == height.dtype == torch.float32
    np.testing.assert_array_equal(height.numpy()[0, 0], native_height[..., 0].astype(np.float32) / np.float32(65535))
    assert height[0, 0, 0, 1] > height[0, 0, 0, 0], "Adjacent codes must not collapse to an 8-bit display grid"
    rng = random.Random(43)
    expected_rng = random.Random(43)
    x, y = expected_rng.randrange(65), expected_rng.randrange(65)
    crop_rgb, crop_height, rectangle = data.crop_pair(rgb, height, 64, rng)
    assert rectangle == [x, y, 64, 64]
    np.testing.assert_array_equal(crop_height.numpy()[0, 0], native_height[y:y+64, x:x+64, 0].astype(np.float32) / np.float32(65535))
    np.testing.assert_array_equal(crop_rgb.numpy()[0], (native_rgb[y:y+64, x:x+64].astype(np.float32) / np.float32(65535)).transpose(2, 0, 1))
    assert {path: data.digest(path) for path in folder.iterdir()} == before


def test_numeric_target_ignores_color_transfer_and_preserves_relative_range():
    codes = np.array([[0, 1, 32767, 32768, 65534, 65535]], dtype=np.uint16)
    values = data.numeric_height(codes)
    np.testing.assert_array_equal(values, codes.astype(np.float32) / np.float32(65535))
    assert values[0, 3] == np.float32(32768 / 65535)
    narrow = np.array([[30000, 30001]], dtype=np.uint16)
    np.testing.assert_array_equal(data.numeric_height(narrow), narrow.astype(np.float32) / np.float32(65535))
    linear_rgb = np.repeat(codes[..., None], 3, axis=-1)
    transferred_rgb = data.input_rgb(linear_rgb, "linear")
    assert transferred_rgb[0, 0, 3] > .7, "Only explicitly linear color inputs use the model's sRGB transfer"
    assert values[0, 3] < .51
    np.testing.assert_array_equal(data.input_rgb(linear_rgb, "source_srgb_assumed")[0], values)


def test_scalar_height_allows_opaque_gray_alpha_but_never_transparent_or_colored_targets():
    values = np.array([[0, 32768, 65535]], dtype=np.uint16)
    opaque = np.full(values.shape, 65535, dtype=np.uint16)
    np.testing.assert_array_equal(data.numeric_height(np.stack((values, opaque), axis=-1)), data.numeric_height(values))
    transparent = opaque.copy()
    transparent[0, 0] = 65534
    with pytest.raises(ValueError, match="Transparent height"):
        data.numeric_height(np.stack((values, transparent), axis=-1))
    rgb = np.repeat(values[..., None], 3, axis=-1)
    rgb[0, 0, 1] = 1
    with pytest.raises(ValueError, match="scalar"):
        data.numeric_height(rgb)
    with pytest.raises(ValueError, match="UInt16"):
        data.numeric_height(values.astype(np.uint8))


@pytest.mark.parametrize("problem", ["actual-size", "reported-size", "precision", "channels", "hash", "encoding", "rectangle", "source-bounds", "identity"])
def test_pair_selection_rejects_inconsistent_actual_source_identity(tmp_path, problem):
    folder, _, codes = add_sample(tmp_path)
    metadata = json.loads((folder / "sample.json").read_text())
    if problem == "actual-size":
        path = folder / "height.png"
        path.unlink()
        write_png(path, codes[:64, :64])
        metadata["map_metadata"]["height"]["sample_sha256"] = data.digest(path)
    elif problem == "reported-size":
        metadata["sample_pixel_dimensions"] = [64, 64]
    elif problem == "precision":
        metadata["map_metadata"]["height"]["sample_bits"] = 8
    elif problem == "channels":
        metadata["map_metadata"]["input"]["channels"] = 1
    elif problem == "hash":
        metadata["map_metadata"]["height"]["sample_sha256"] = "0" * 64
    elif problem == "encoding":
        metadata["map_metadata"]["height"]["encoding"] = "srgb"
    elif problem == "rectangle":
        metadata["crop_rectangle_top_left_xywh"] = [0, 0, 64, 64]
    elif problem == "source-bounds":
        metadata["crop_rectangle_top_left_xywh"] = [450, 450, 128, 128]
    elif problem == "identity":
        metadata["material_id"] = "another-surface"
    (folder / "sample.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        data.select_pairs(tmp_path)


def test_excluded_materials_and_crops_are_omitted_without_blocking_other_training(tmp_path):
    included, _, _ = add_sample(tmp_path)
    excluded, _, _ = add_sample(tmp_path, "bad_001", "bad-surface", status="excluded")
    # Excluded data need not be readable, because it must never enter training.
    shutil.rmtree(excluded)
    training, checks, identity = data.select_pairs(tmp_path)
    assert [pair["metadata"]["sample_id"] for pair in training] == [included.name]
    assert checks == [] and identity["materials"] == 1
    with pytest.raises(ValueError, match="bad-surface"):
        data.select_pairs(tmp_path, ["bad-surface"])
    with pytest.raises(ValueError, match="missing-surface"):
        data.select_pairs(tmp_path, ["missing-surface"])


def test_check_crop_must_use_distinct_source_pixels_and_report_known_material_scope(tmp_path):
    add_sample(tmp_path)
    check, _, _ = add_sample(tmp_path, "surface_002", split="validation", origin=(128, 128))
    training, checks, identity = data.select_pairs(tmp_path)
    assert len(training) == len(checks) == 1
    assert "does not establish unseen-material generalization" in identity["check_scope"]
    mutate_metadata(check, crop_rectangle_top_left_xywh=[64, 64, 128, 128])
    with pytest.raises(ValueError, match="overlaps"):
        data.select_pairs(tmp_path)


def test_duplicate_sample_and_path_escape_are_rejected(tmp_path):
    add_sample(tmp_path)
    index_path = tmp_path / "dataset.json"
    index = json.loads(index_path.read_text())
    index["samples"].append(index["samples"][0])
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Duplicate"):
        data.select_pairs(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        data.contained(tmp_path, "../another-dataset/source.png")


def test_cache_key_distinguishes_same_sample_id_from_other_source_and_transfer(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    add_sample(first, base=12000)
    folder, _, _ = add_sample(second, base=40000)
    pair_a = data.select_pairs(first)[0][0]
    pair_b = data.select_pairs(second)[0][0]
    cache = data.PairCache(1024 * 1024)
    first_values = cache.load(pair_a)
    second_values = cache.load(pair_b)
    assert first_values[1][0, 0, 0, 0] < second_values[1][0, 0, 0, 0]
    assert cache.load(pair_a)[0] is first_values[0]
    metadata = json.loads((folder / "sample.json").read_text())
    metadata["map_metadata"]["input"]["encoding"] = "linear"
    (folder / "sample.json").write_text(json.dumps(metadata))
    pair_c = data.select_pairs(second)[0][0]
    linear_values = cache.load(pair_c)
    assert linear_values[0][0, 0, 0, 64] > second_values[0][0, 0, 0, 64]
    assert len(cache.entries) == 3


def test_cache_detects_changed_source_even_after_cache_hit(tmp_path):
    folder, _, height = add_sample(tmp_path)
    pair = data.select_pairs(tmp_path)[0][0]
    cache = data.PairCache(1024 * 1024)
    cache.load(pair)
    path = folder / "height.png"
    path.unlink()
    write_png(path, height + np.uint16(1))
    with pytest.raises(ValueError, match="changed during training"):
        cache.load(pair)
    assert cache.bytes == 0 and not cache.entries and not cache.file_states


def test_cache_decodes_once_on_hit_and_obeys_byte_budget(tmp_path, monkeypatch):
    add_sample(tmp_path, "surface_001")
    add_sample(tmp_path, "surface_002", origin=(128, 0))
    pairs = data.select_pairs(tmp_path)[0]
    one_pair_bytes = 128 * 128 * 4 * 4
    cache = data.PairCache(one_pair_bytes)
    calls = []
    original = data.read_png

    def counting_read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(data, "read_png", counting_read)
    cache.load(pairs[0])
    cache.load(pairs[0])
    assert len(calls) == 2
    cache.load(pairs[1])
    assert cache.bytes == one_pair_bytes and len(cache.entries) == 1 and len(cache.file_states) == 1
    cache.load(pairs[0])
    assert len(calls) == 6 and cache.bytes == one_pair_bytes
    disabled = data.PairCache(one_pair_bytes - 1)
    disabled.load(pairs[0])
    assert disabled.bytes == 0 and not disabled.entries


@pytest.mark.parametrize("size", [192, 63, 65])
def test_native_crop_rejects_undersized_source_or_non_native_grid(size):
    rgb = torch.zeros(1, 3, 128, 128)
    height = torch.zeros(1, 1, 128, 128)
    with pytest.raises(ValueError):
        data.crop_pair(rgb, height, size, random.Random(7))
    with pytest.raises(ValueError, match="matching dimensions"):
        data.crop_pair(rgb, height[..., :64, :64], 64, random.Random(7))


def test_geometric_augmentation_only_reorders_real_target_codes():
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    height = (yy * 128 + xx).to(torch.float32)[None, None] / 65535
    rgb = torch.cat((height, height, height), dim=1)
    _, augmented, rectangle = data.crop_pair(rgb, height, 64, random.Random(12), augment=True)
    x, y, _, _ = rectangle
    expected = height[..., y:y+64, x:x+64]
    assert torch.equal(augmented.flatten().sort().values, expected.flatten().sort().values)


def test_whole_rectangular_maps_retain_every_real_pixel_without_padding_or_resizing():
    codes = np.arange(512 * 1024, dtype=np.uint32).reshape(512, 1024).astype(np.uint16)
    height = torch.from_numpy(data.numeric_height(codes)[None, None])
    rgb = height.repeat(1, 3, 1, 1)
    full_rgb, full_height, rectangle = data.crop_pair(rgb, height, 1024, random.Random(7), whole_maps=True)
    assert rectangle == [0, 0, 1024, 512]
    assert torch.equal(full_rgb, rgb) and torch.equal(full_height, height)
    shapes = set()
    for seed in range(8):
        transformed_rgb, transformed_height, source_rectangle = data.crop_pair(
            rgb, height, 1024, random.Random(seed), augment=True, whole_maps=True)
        assert source_rectangle == rectangle
        assert transformed_rgb.shape[-2:] == transformed_height.shape[-2:]
        assert torch.equal(torch.sort(transformed_height.flatten()).values, torch.sort(height.flatten()).values)
        shapes.add(tuple(transformed_height.shape[-2:]))
    assert shapes == {(512, 1024), (1024, 512)}


@pytest.mark.parametrize("dimensions,size,message", [
    ((512, 1024), 512, "maximum edge"),
    ((128, 1024), 1024, "at least256"),
    ((480, 1024), 1024, "divisible by64"),
])
def test_whole_maps_reject_unusable_native_dimensions_without_repair(dimensions, size, message):
    height = torch.zeros(1, 1, *dimensions)
    rgb = height.repeat(1, 3, 1, 1)
    with pytest.raises(ValueError, match=message):
        data.crop_pair(rgb, height, size, random.Random(7), whole_maps=True)


def test_relief_loss_penalizes_missing_fine_detail_and_inversion_with_gradients():
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    grain = ((xx + yy) % 2).to(torch.float32) * .08 - .04
    target = (.5 + grain)[None, None]
    exact, exact_metrics = data.height_loss(target, target)
    flat = torch.full_like(target, .5, requires_grad=True)
    flattened, flat_metrics = data.height_loss(flat, target)
    inverted, inverted_metrics = data.height_loss(1 - target, target)
    assert exact.item() == 0 and exact_metrics["detail_l1"] == 0
    assert flat_metrics["detail_l1"] > 0 and flattened > 0
    assert inverted_metrics["detail_l1"] > flat_metrics["detail_l1"] and inverted > flattened
    flattened.backward()
    assert torch.isfinite(flat.grad).all() and flat.grad.abs().sum() > 0
    assert torch.all(flat.grad[..., :16, :] == 0), "Only real interior pixels contribute to fitting"
    with pytest.raises(ValueError, match="nonnegative"):
        data.height_loss(target, target, margin=-1)


def test_display_png_preserves_16bit_codes_and_exr_retains_unclipped_float32(tmp_path):
    pytest.importorskip("OpenEXR")
    import OpenEXR
    codes = np.array([[0, 1, 16384, 16385, 32767, 32768, 65534, 65535]], dtype=np.uint16)
    values = codes.astype(np.float32) / np.float32(65535)
    png = tmp_path / "display.png"
    data.write_display(png, values)
    decoded, metadata = read_png(png)
    assert metadata["sample_bits"] == 16 and metadata["channels"] == 1
    np.testing.assert_array_equal(decoded[..., 0], codes)
    raw = np.concatenate((values, np.array([[-.125, 1.125]], dtype=np.float32)), axis=1)
    before = raw.copy()
    exr = tmp_path / "raw.exr"
    data.write_exr(exr, raw)
    reopened = OpenEXR.File(str(exr), separate_channels=True)
    assert reopened.channels()["Y"].pixels.dtype == np.float32
    np.testing.assert_array_equal(reopened.channels()["Y"].pixels, before)
    assert reopened.header()["textureStudioEncoding"] == "linear_data"
    np.testing.assert_array_equal(raw, before)
    data.write_display(tmp_path / "clipped-display.png", raw)
    preview, _ = read_png(tmp_path / "clipped-display.png")
    np.testing.assert_array_equal(preview[0, -2:, 0], np.array([0, 65535], dtype=np.uint16))
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("writer", [data.write_display, data.write_exr])
def test_export_rejects_nonfinite_values_instead_of_burning_invalid_pixels(tmp_path, writer):
    with pytest.raises(ValueError, match="finite"):
        writer(tmp_path / "invalid", np.array([[np.nan, np.inf]], dtype=np.float32))
    assert not (tmp_path / "invalid").exists()
