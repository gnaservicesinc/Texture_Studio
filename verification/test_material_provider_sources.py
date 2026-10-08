"""Original provider filenames and byte-bound provenance stay distinct."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import material_dataset as dataset
import material_recreation as recreation
import audit_material_sources as source_audit


@pytest.mark.parametrize("name, expected", [
    ("Snow013_4K-PNG_Color.png", ("Snow013", "diff", "4k")),
    ("Snow013_4K-PNG_Displacement.png", ("Snow013", "disp", "4k")),
    ("Snow014_4K-PNG_NormalGL.png", ("Snow014", "nor_gl", "4k")),
    ("Snow015_4K-PNG_NormalDX.png", ("Snow015", "nor_dx", "4k")),
    ("Snow015_4K-PNG_Roughness.png", ("Snow015", "rough", "4k")),
    ("rocks_ground_01_rough_ao_4k.png", ("rocks_ground_01", "rough_ao", "4k")),
    ("snow_01_translucent_4k.png", ("snow_01", "translucent", "4k")),
])
def test_original_provider_names(name, expected):
    assert dataset.parse_source_filename(name) == expected


def test_compound_polyhaven_map_suffix_keeps_asset_identity():
    match = source_audit.PARENT.fullmatch("rocks_ground_01_rough_ao_4k.png")
    assert match["asset"] == "rocks_ground_01" and match["role"] == "rough_ao"


def fixture(tmp_path):
    root = tmp_path / "sources"
    folder = root / "Snow013"
    folder.mkdir(parents=True)
    for suffix, dtype, channels in (("Color", np.uint8, 3), ("Displacement", np.uint16, 1),
                                    ("NormalGL", np.uint16, 3), ("Roughness", np.uint8, 1)):
        values = np.arange(16 * 16 * channels, dtype=dtype).reshape(16, 16, channels)
        dataset.write_png(folder / f"Snow013_4K-PNG_{suffix}.png", values)
    material = dataset.discover_materials(root, 4)[0]
    package = {"schema": recreation.PACKAGE_AUDIT_SCHEMA, "provider": "ambientCG", "material_id": "snow013",
        "asset_id": "Snow013", "asset_url": "https://ambientcg.com/view?id=Snow013", "completed_utc": "2026-10-07T22:00:00Z",
        "package": {"url": "https://ambientcg.com/get?file=Snow013_4K-PNG.zip", "filename": "Snow013_4K-PNG.zip", "file_bytes": 999, "sha256": "d" * 64},
        "license": {"spdx": "CC0-1.0", "url": "https://docs.ambientcg.com/license/", "snapshot_sha256": "e" * 64},
        "creation_method": {"id": "PBRPhotogrammetry", "asset_page_label": "Surface Photogrammetry"},
        "files": [{"source_filename": s["filename"], "source_sha256": s["file_sha256"], "member_sha256": s["file_sha256"],
            "source_bytes": s["file_bytes"], "member_bytes": s["file_bytes"], "exact_full_file_match": True, "source_stable": True,
            "archive_member": s["filename"]} for s in material["maps"].values()]}
    return root, folder, material, {"files": [], "package_audits": [package]}


def test_provider_identity_and_original_samples_survive_preparation(tmp_path):
    _, folder, material, audit = fixture(tmp_path)
    assert material["ready"] and material["material_id"] == "snow013"
    assert dataset.slugify("Snow013") != dataset.slugify("snow_01")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    records = dataset.prepare_material(material, tmp_path / "dataset", {}, True, .2, audit)
    for record in records:
        assert record["source_provider"] == "ambientCG"
        assert record["source_asset_id"] == "Snow013"
        assert record["source_url"] == "https://ambientcg.com/view?id=Snow013"
        assert record["source_creation_method"]["id"] == "PBRPhotogrammetry"
        assert record["map_metadata"]["height"]["sample_bits"] == 16
        assert record["map_metadata"]["roughness"]["sample_bits"] == 8
        assert dataset.verify_sample(tmp_path / "dataset" / "samples" / record["sample_id"], True) == []
        for item in record["map_metadata"].values():
            assert item["transforms"] == []
            assert item["source"]["filename"] in before
            assert item["source"]["published_archive_member"] == item["source"]["filename"]
            assert "published_url" not in item["source"]
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("change", [{"file_sha256": "f" * 64}, {"file_bytes": 1}])
def test_changed_member_cannot_receive_provider_claim(tmp_path, change):
    _, _, material, audit = fixture(tmp_path)
    source = material["maps"]["height"]
    assert dataset.published_source_provenance(source, "snow013", audit)["provider"] == "ambientCG"
    assert dataset.published_source_provenance(source | change, "snow013", audit) == {}
    assert dataset.published_source_provenance(source, "snow_01", audit) == {}


def test_package_and_polyhaven_audits_load_without_relabeling(tmp_path):
    _, _, _, audit = fixture(tmp_path)
    ph = tmp_path / "ph.json"
    ph.write_text(json.dumps({"files": [{"asset_id": "snow_01"}]}))
    pkg = tmp_path / "snow013.json"
    pkg.write_text(json.dumps(audit["package_audits"][0]))
    loaded = dataset.load_source_audits(ph, [pkg])
    assert loaded["files"] == [{"asset_id": "snow_01"}]
    assert loaded["package_audits"][0]["asset_id"] == "Snow013"
    assert dataset.load_source_audits(None, []) is None


def test_recreation_keeps_original_archive_names(tmp_path):
    _, _, material, audit = fixture(tmp_path)
    for source in material["maps"].values():
        evidence = recreation.package_download_evidence(source, "snow013", audit["package_audits"])
        assert evidence["download_kind"] == "zip_archive_member"
        assert evidence["archive_filename"] == "Snow013_4K-PNG.zip"
        assert evidence["restore_filename"] == source["filename"] == evidence["archive_member"]
        assert evidence["provider"] == "ambientCG"
        assert "filenames are explicitly recorded" in evidence["verification_basis"]
        assert "only source filename changed" not in evidence["verification_basis"]
