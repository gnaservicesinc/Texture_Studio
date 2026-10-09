"""Registered resolution families and old color alternatives retain raw codes."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_dataset as dataset
import material_native_size as native


def write_set(root, resolution="4k", edge=16, family="fabric", colors=("diff",), height=True):
    folder = root / "Original Fabric"
    folder.mkdir(parents=True, exist_ok=True)
    roles = [*colors, "rough", "nor_dx", "nor_gl", *( ["disp"] if height else [])]
    for number, suffix in enumerate(roles):
        channels = 3 if suffix in colors or suffix.startswith("nor_") else 1
        values = (np.arange(edge * edge * channels, dtype=np.uint16).reshape(edge, edge, channels) + number * 170)
        dataset.write_png(folder / f"{family}_{suffix}_{resolution}.png", values)
    return folder


@pytest.mark.parametrize("filename, expected", [
    ("fabric_pattern_05_col_01_4k.png", ("fabric_pattern_05", "col_01", "4k")),
    ("fabric_pattern_07_col_1_4k (1).png", ("fabric_pattern_07", "col_1", "4k")),
    ("leather_red_02_coll1_4k.png", ("leather_red_02", "col_1", "4k")),
    ("fabric_color2_2k.png", ("fabric", "color2", "2k")),
    ("fabric_albedo_8k.png", ("fabric", "albedo", "8k")),
])
def test_legacy_color_names(filename, expected):
    assert dataset.parse_source_filename(filename) == expected
    assert dataset.source_role(expected[1]) == "input"


def test_resolution_sets_share_family_and_never_mix_maps(tmp_path):
    write_set(tmp_path, "2k", 8)
    write_set(tmp_path, "4k", 16)
    write_set(tmp_path, "8k", 32)
    materials = native._source_maps(tmp_path)
    assert [m["material_id"] for m in materials] == ["fabric_2k", "fabric_4k", "fabric_8k"]
    assert {m["source_family_id"] for m in materials} == {"fabric"}
    assert len({m["source_set_id"] for m in materials}) == 3
    for material in materials:
        assert {(m["width"], m["height"]) for m in material["maps"].values()} == {tuple(material["common_pixel_dimensions"])}
        assert material["maps"]["normal"]["suffix"] == "nor_gl"
    offline = dataset.discover_materials(tmp_path, 4)
    assert len(offline) == 3 and all(m["ready"] for m in offline)


def test_color_alternatives_bind_one_geometry_without_duplicate_images(tmp_path):
    root = tmp_path / "sources"
    folder = write_set(root, colors=("col_01", "col_02", "col_03"))
    original_files = {p: p.read_bytes() for p in folder.iterdir()}
    material = dataset.discover_materials(root, 4)[0]
    assert material["ready"]
    assert [v["variant_id"] for v in material["input_variants"]] == ["color_1", "color_2", "color_3"]
    records = dataset.prepare_material(material, tmp_path / "dataset", {}, True, 0)
    for record in records:
        variants = record["input_variants"]
        assert len(variants) == 3
        assert len({variant["path"] for variant in variants}) == 3
        assert len(list((tmp_path / "dataset" / "samples" / record["sample_id"]).glob("*.png"))) == 6
        x, y, width, height = record["crop_rectangle_top_left_xywh"]
        for variant in variants:
            expected, _ = dataset.read_png(variant["source"]["path"])
            actual, _ = dataset.read_png(tmp_path / "dataset" / "samples" / record["sample_id"] / variant["path"])
            np.testing.assert_array_equal(actual, expected[y:y + height, x:x + width])
    assert original_files == {p: p.read_bytes() for p in folder.iterdir()}


def test_identical_duplicate_download_is_not_another_color_variant(tmp_path):
    folder = write_set(tmp_path, colors=("col_1", "col_2"))
    original = folder / "fabric_col_1_4k.png"
    (folder / "fabric_col_1_4k (1).png").write_bytes(original.read_bytes())
    material = native._source_maps(tmp_path)[0]
    assert len(material["input_variants"]) == 2
    assert len(material["ignored_files"]) == 2  # Duplicate color and alternate DX normal.


def test_different_duplicate_color_identity_requires_resolution(tmp_path):
    folder = write_set(tmp_path, colors=("col_1", "col_2"))
    dataset.write_png(folder / "fabric_col_1_4k (1).png", np.zeros((16, 16, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="same color variant"):
        native._source_maps(tmp_path)


def test_missing_geometry_is_not_borrowed_from_different_resolution(tmp_path):
    folder = write_set(tmp_path, "4k", 16, height=False)
    dataset.write_png(folder / "fabric_disp_2k.png", np.ones((8, 8, 1), dtype=np.uint16))
    (folder / "fabric_spec_4k.png").write_bytes(b"An ignored map need not be decoded")
    materials = native._source_maps(tmp_path)
    assert len(materials) == 1
    assert "height" not in materials[0]["maps"]
    assert "extra:spec" not in materials[0]["maps"]


def test_source_index_refresh_keeps_reviews_and_references_only(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    folder = write_set(root / "sources", "4k", 16, colors=("col_1", "col_2"))
    index = native.ensure_source_index(root, {"samples": []})
    record_path = root / index["samples"][0]["path"] / "sample.json"
    record = json.loads(record_path.read_text())
    record.update(status="approved", review_status="approved", curation_note="Keep the fine weave")
    dataset.write_json(record_path, record)
    index["samples"][0]["status"] = "approved"
    dataset.write_json(root / "dataset.json", index)
    assert native.ensure_source_index(root, index) == index
    write_set(root / "sources", "2k", 8, colors=("col_1", "col_2"))
    # Old file summaries are cached; only the new four maps are inspected.
    inspected = []
    original_summary = dataset.source_summary
    monkeypatch.setattr(dataset, "source_summary", lambda p: (inspected.append(p.name), original_summary(p))[1])
    updated = native.ensure_source_index(root, index)
    assert {entry["material_id"] for entry in updated["samples"]} == {"fabric_2k", "fabric_4k"}
    assert len(inspected) == 6  # Two colors, roughness, DX/GL normals, height.
    original_record = json.loads((root / "samples" / "fabric_4k_full" / "sample.json").read_text())
    new_record = json.loads((root / "samples" / "fabric_2k_full" / "sample.json").read_text())
    assert original_record["status"] == "approved"
    assert original_record["curation_note"] == "Keep the fine weave"
    assert new_record["status"] == "unreviewed"
    for record in (original_record, new_record):
        assert record["source_family_id"] == record["asset_family_id"] == "fabric"
        assert len(record["input_variants"]) == 2
        assert all(Path(v["path"]).is_absolute() and v["storage"] == "source_reference" for v in record["input_variants"])
    assert not list((root / "samples").rglob("*.png"))
    assert all(path.parent == folder for path in folder.iterdir())


def test_low_bit_height_is_recorded_but_never_declared_trainable(tmp_path):
    root = tmp_path / "dataset"
    folder = write_set(root / "sources", height=False)
    dataset.write_png(folder / "fabric_disp_4k.png", np.ones((16, 16, 1), dtype=np.uint8))
    index = native.ensure_source_index(root, {"samples": []})
    record = json.loads((root / index["samples"][0]["path"] / "sample.json").read_text())
    assert record["available_targets"] == ["roughness", "normal"]
    assert record["map_metadata"]["height"]["sample_bits"] == 8


def test_exact_provider_sidecar_claim_is_bound_to_source_bytes(tmp_path):
    folder = write_set(tmp_path, "2k", 16)
    color = folder / "fabric_diff_2k.png"
    inspected = dataset.source_summary(color)
    url = "https://dl.polyhaven.org/file/ph-assets/Textures/png/2k/fabric/fabric_diff_2k.png"
    manifest = {"material_id": "fabric", "resolution": "2k", "provider": "Poly Haven", "license": "CC0-1.0",
        "api": {"api_url": "https://api.polyhaven.com/files/fabric"},
        "maps": {"input": {"published_bytes": inspected["file_bytes"], "published_md5": inspected["file_md5"], "url": url}},
        "downloaded_maps": {"input": {"path": str(color), "sha256": inspected["file_sha256"], "bytes": inspected["file_bytes"]}}}
    sidecar = folder / "material-source-2k.json"
    dataset.write_json(sidecar, manifest)
    source = native._source_maps(tmp_path)[0]["maps"]["input"]
    assert source["provider"] == "Poly Haven" and source["published_url"] == url
    manifest["downloaded_maps"]["input"]["sha256"] = "0" * 64
    dataset.write_json(sidecar, manifest)
    assert "provider" not in native._source_maps(tmp_path)[0]["maps"]["input"]


def test_source_refresh_never_rebinds_changed_or_missing_maps(tmp_path):
    root = tmp_path / "dataset"
    folder = write_set(root / "sources", "2k", 16)
    index = native.ensure_source_index(root, {"samples": []})
    metadata_path = root / index["samples"][0]["path"] / "sample.json"
    before = json.loads(metadata_path.read_text())
    (folder / "fabric_disp_2k.png").unlink()
    (folder / "fabric_diff_2k.png").unlink()
    dataset.write_png(folder / "fabric_diff_2k.png", np.zeros((16, 16, 3), dtype=np.uint16))
    refreshed = native.ensure_source_index(root, index)
    after = json.loads(metadata_path.read_text())
    assert refreshed == index
    assert after["map_metadata"]["height"] == before["map_metadata"]["height"]
    assert after["map_metadata"]["input"]["sample_sha256"] == before["map_metadata"]["input"]["sample_sha256"]


def test_missing_source_tombstone_only_restores_bound_original(tmp_path):
    root = tmp_path / "dataset"
    folder = write_set(root / "sources", "2k", 16)
    index = native.ensure_source_index(root, {"samples": []})
    record = json.loads((root / index["samples"][0]["path"] / "sample.json").read_text())
    source = record["map_metadata"]["height"]["source"]
    height = Path(source["path"])
    original_bytes = height.read_bytes()
    index["samples"] = []
    index["removed_missing_sources"] = [{"source_set_id": record["source_set_id"],
        "material_id": record["material_id"], "path": str(height), "file_sha256": source["file_sha256"]}]
    height.unlink()
    dataset.write_json(root / "dataset.json", index)
    assert not native.ensure_source_index(root, index)["samples"]
    dataset.write_png(height, np.zeros((16, 16, 1), dtype=np.uint16))
    assert not native.ensure_source_index(root, index)["samples"]
    height.write_bytes(original_bytes)
    assert native.ensure_source_index(root, index)["samples"]


def test_missing_color_variant_tombstone_keeps_geometry_and_other_colors(tmp_path):
    root = tmp_path / "dataset"
    folder = write_set(root / "sources", "2k", 16, colors=("col_1", "col_2"))
    index = native.ensure_source_index(root, {"samples": []})
    metadata_path = root / index["samples"][0]["path"] / "sample.json"
    record = json.loads(metadata_path.read_text())
    removed = record["input_variants"].pop()
    color = Path(removed["path"])
    original_bytes = color.read_bytes()
    index["removed_missing_color_variants"] = [{"source_set_id": record["source_set_id"],
        "material_id": record["material_id"], "variant_id": removed["variant_id"],
        "path": str(color), "file_sha256": removed["sample_sha256"]}]
    color.unlink()
    dataset.write_json(metadata_path, record)
    dataset.write_json(root / "dataset.json", index)
    native.ensure_source_index(root, index)
    assert len(json.loads(metadata_path.read_text())["input_variants"]) == 1
    dataset.write_png(color, np.zeros((16, 16, 3), dtype=np.uint16))
    native.ensure_source_index(root, index)
    assert len(json.loads(metadata_path.read_text())["input_variants"]) == 1
    color.write_bytes(original_bytes)
    updated = native.ensure_source_index(root, index)
    restored = json.loads(metadata_path.read_text())
    assert len(restored["input_variants"]) == 2
    assert "height" in restored["maps"]
    assert updated["removed_missing_color_variants"] == index["removed_missing_color_variants"]
