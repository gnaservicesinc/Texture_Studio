"""CPU-only native ambientCG download, original-byte and provenance contracts."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_ambientcg_material_resolution as download
from material_dataset import write_png
from material_recreation import package_download_evidence


def metadata(asset="Snow013", size=123, method="PBRPhotogrammetry"):
    return {"foundAssets": [{"assetId": asset, "dataType": "Material", "creationMethod": method,
                             "dimensionX": 1.2, "dimensionY": .6,
                             "downloadFolders": {"default": {"downloadFiletypeCategories": {"zip": {"downloads": [
                                 {"fileName": f"{asset}_1K-PNG.zip", "downloadLink": f"https://ambientcg.com/get?file={asset}_1K-PNG.zip", "size": size}
                             ]}}}}}]}


def plan(asset="Snow013", size=123):
    raw = json.dumps(metadata(asset, size)).encode()
    license_bytes = b"Creative Commons CC0 1.0 Universal License"
    return {"asset_id": asset, "material_id": asset.lower(), "provider": "ambientCG", "resolution": "1k",
            "asset_url": f"https://ambientcg.com/view?id={asset}", "license": "CC0-1.0", "license_url": download.audit.LICENSE_URL,
            "creation_method": {"id": "PBRPhotogrammetry", "asset_page_label": "Surface Photogrammetry"},
            "physical_width_meters": 1.2, "physical_height_meters": .6,
            "package": {"filename": f"{asset}_1K-PNG.zip", "url": f"https://ambientcg.com/get?file={asset}_1K-PNG.zip", "published_api_bytes": size},
            "api": {"api_url": f"https://ambientcg.com/api/v2/full_json?include=downloadData&id={asset}", "snapshot_sha256": hashlib.sha256(raw).hexdigest()},
            "license_evidence": {"snapshot_sha256": hashlib.sha256(license_bytes).hexdigest(), "retrieved_utc": "test"},
            "_api_bytes": raw, "_license_bytes": license_bytes}


def archive_fixture(root, asset="Snow013", bits=16, mismatched=False):
    source = root / "original"; source.mkdir(parents=True)
    for role, suffix in download.ROLE_SUFFIXES.items():
        size = (512, 1024) if role == "roughness" and mismatched else (1024, 1024)
        channels = 3 if role in ("input", "normal") else 1
        role_bits = bits if role == "height" else (16 if role == "normal" else 8)
        yy, xx = np.indices(size)
        array = ((xx * 61 + yy * 47) % (65536 if role_bits == 16 else 256)).astype(np.uint16 if role_bits == 16 else np.uint8)
        array = np.repeat(array[..., None], channels, axis=2)
        write_png(source / f"{asset}_1K-PNG_{suffix}.png", array)
    archive = root / f"{asset}_1K-PNG.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for path in source.iterdir(): package.write(path, path.name)
        package.writestr("../../unrelated.txt", "never extracted")
        package.writestr(f"{asset}_1K-PNG_NormalDX.png", b"ignored DirectX normal")
    return archive, {path.name: path.read_bytes() for path in source.iterdir()}


def mock_archive(monkeypatch, archive, bad_sha=False):
    def fetch(_url, path, maximum, filename=None):
        assert filename == archive.name and maximum == archive.stat().st_size
        shutil.copyfile(archive, path)
        return {"headers": {"x-bz-content-sha1": "0" * 40 if bad_sha else hashlib.sha1(path.read_bytes()).hexdigest()}, "completed_utc": "test"}
    monkeypatch.setattr(download.audit, "bounded_download", fetch)


def test_official_plan_records_photogrammetry_and_current_license(monkeypatch):
    raw = json.dumps(metadata()).encode()
    calls = []
    def fetch(url, path, maximum, filename=None):
        calls.append(url)
        path.write_bytes(b"Creative Commons CC0 1.0 Universal License" if url == download.audit.LICENSE_URL else raw)
        return {"headers": {}, "completed_utc": "test"}
    monkeypatch.setattr(download.audit, "bounded_download", fetch)
    result = download.fetch_plan("Snow013", "1K")
    assert result["creation_method"]["asset_page_label"] == "Surface Photogrammetry"
    assert result["api"]["snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["_api_bytes"] == raw and result["resolution"] == "1k"
    assert result["provider_dimensions_raw"] == {"dimensionX": 1.2, "dimensionY": .6, "dimensionZ": None}
    assert result["physical_width_meters"] is None and result["physical_height_meters"] is None
    assert not any(key.startswith("_") for key in download.public_plan(result))
    assert len(calls) == 2 and all(download.audit.official_url(url) for url in calls)


@pytest.mark.parametrize("method,license_text,message", [("PBRProcedural", "Creative Commons CC0 1.0 Universal License", "Photogrammetry"),
                                                       ("PBRPhotogrammetry", "unknown license", "license")])
def test_unverified_method_or_license_rejected(monkeypatch, method, license_text, message):
    def fetch(url, path, *_args):
        path.write_bytes(license_text.encode() if url == download.audit.LICENSE_URL else json.dumps(metadata(method=method)).encode())
        return {"completed_utc": "test"}
    monkeypatch.setattr(download.audit, "bounded_download", fetch)
    with pytest.raises(ValueError, match=message): download.fetch_plan("Snow013", "1K")


def test_original_1k_maps_publish_exact_bytes_and_compatible_audit(tmp_path, monkeypatch):
    archive, originals = archive_fixture(tmp_path)
    mock_archive(monkeypatch, archive)
    result = download.download_asset(plan(size=archive.stat().st_size), tmp_path / "sources-1k", tmp_path / "audits")
    assert result["actual_native_pixel_dimensions"] == [1024, 1024]
    assert set(result["downloaded_maps"]) == set(download.ROLE_SUFFIXES)
    directory = tmp_path / "sources-1k/snow013"
    assert {path.name: path.read_bytes() for path in directory.glob("*.png")} == originals
    assert not result["temporary_archive_retained"] and not list((tmp_path / "sources-1k").glob(".ambientcg-download-*"))
    report = json.loads(Path(result["package_audit_path"]).read_text())
    assert report["schema"] == download.audit.SCHEMA
    assert report["creation_method"]["id"] == "PBRPhotogrammetry"
    assert report["package"]["sha1_matches_provider_header"] is True
    assert {entry["role"]: entry["source_sample_bits"] for entry in report["files"]} == {"input": 8, "height": 16, "normal": 16, "roughness": 8}
    entry = next(entry for entry in report["files"] if entry["role"] == "height")
    source = {"filename": entry["source_filename"], "file_bytes": entry["source_bytes"], "file_sha256": entry["source_sha256"]}
    assert package_download_evidence(source, "snow013", [report])["creation_method"]["id"] == "PBRPhotogrammetry"
    before = (directory / "material-source-1k.json").read_bytes()
    again = download.download_asset(plan(size=archive.stat().st_size), tmp_path / "sources-1k", tmp_path / "audits")
    assert again["existing_manifest_reused"] and (directory / "material-source-1k.json").read_bytes() == before
    assert all(entry["action"] == "reused_verified" for entry in again["downloaded_maps"].values())


@pytest.mark.parametrize("bits,mismatched,message", [(8, False, "8-bit"), (16, True, "dimensions differ")])
def test_bad_precision_or_pair_dimensions_publish_no_pngs(tmp_path, monkeypatch, bits, mismatched, message):
    archive, _ = archive_fixture(tmp_path, bits=bits, mismatched=mismatched)
    mock_archive(monkeypatch, archive)
    with pytest.raises(ValueError, match=message):
        download.download_asset(plan(size=archive.stat().st_size), tmp_path / "sources-1k", tmp_path / "audits")
    assert not list((tmp_path / "sources-1k").rglob("*.png"))
    assert not list((tmp_path / "sources-1k").glob(".ambientcg-download-*"))


def test_changed_existing_file_is_not_overwritten_or_new_maps_published(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path)
    mock_archive(monkeypatch, archive)
    source = tmp_path / "sources-1k/snow013"; source.mkdir(parents=True)
    changed = source / "Snow013_1K-PNG_Roughness.png"; changed.write_bytes(b"existing changed data")
    with pytest.raises(ValueError, match="left untouched"):
        download.download_asset(plan(size=archive.stat().st_size), source.parent, tmp_path / "audits")
    assert changed.read_bytes() == b"existing changed data"
    assert list(source.iterdir()) == [changed]


def test_provider_sha_mismatch_publish_no_originals(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path)
    mock_archive(monkeypatch, archive, bad_sha=True)
    with pytest.raises(ValueError, match="SHA1 differs"):
        download.download_asset(plan(size=archive.stat().st_size), tmp_path / "sources-1k", tmp_path / "audits")
    assert not list((tmp_path / "sources-1k").rglob("*.png"))


def test_failed_publication_rolls_back_only_its_new_maps(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path)
    mock_archive(monkeypatch, archive)
    original_link = download.os.link
    calls = []
    def link(source, target):
        calls.append(target)
        if len(calls) == 2: raise PermissionError("publication blocked")
        return original_link(source, target)
    monkeypatch.setattr(download.os, "link", link)
    unrelated = tmp_path / "sources-1k/keep.txt"; unrelated.parent.mkdir(); unrelated.write_bytes(b"keep")
    with pytest.raises(PermissionError, match="publication blocked"):
        download.download_asset(plan(size=archive.stat().st_size), unrelated.parent, tmp_path / "audits")
    assert unrelated.read_bytes() == b"keep" and not list(unrelated.parent.rglob("*.png"))


@pytest.mark.parametrize("mode", ("duplicate", "symlink", "missing"))
def test_ambiguous_or_nonregular_original_members_rejected(tmp_path, mode):
    archive, _ = archive_fixture(tmp_path)
    if mode == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(archive, "a") as package:
            package.writestr("Snow013_1K-PNG_Displacement.png", b"duplicate")
    else:
        changed = tmp_path / "changed.zip"
        with zipfile.ZipFile(archive) as original, zipfile.ZipFile(changed, "w") as package:
            for member in original.infolist():
                if member.filename.endswith("_Displacement.png"):
                    if mode == "missing": continue
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
                package.writestr(member, original.read(member))
        archive = changed
    with pytest.raises(ValueError, match="ZIP member"):
        download.stage_members(archive, tmp_path / "stage", "Snow013", "1K")
    assert not (tmp_path / "stage").exists()


def test_total_budget_prevents_package_download(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "fetch_plan", lambda asset, _resolution: plan(asset))
    monkeypatch.setattr(download, "download_asset", lambda *_args: pytest.fail("No package download permitted"))
    assert download.main(["--assets", "Snow013", "--download", "--max-bytes", "1", "--destination", str(tmp_path / "sources"),
                          "--evidence", str(tmp_path / "evidence")]) == 1
    report = json.loads((tmp_path / "evidence/download-report.json").read_text())
    assert not report["within_requested_byte_budget"] and not (tmp_path / "sources").exists()


def test_concurrent_download_failure_does_not_hide_success(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    monkeypatch.setattr(download, "fetch_plan", lambda asset, _resolution: plan(asset))
    def fetch(one, *_args):
        barrier.wait(timeout=5)
        if one["asset_id"] == "Snow013": raise ValueError("simulated package failure")
        return {"asset_id": one["asset_id"], "status": "downloaded_or_reused_verified"}
    monkeypatch.setattr(download, "download_asset", fetch)
    assert download.main(["--assets", "Snow013,Snow014", "--download", "--workers", "2", "--destination", str(tmp_path / "sources"),
                          "--evidence", str(tmp_path / "evidence")]) == 1
    report = json.loads((tmp_path / "evidence/download-report.json").read_text())
    assert [item["asset_id"] for item in report["downloaded_materials"]] == ["Snow014"]
    assert report["errors"] == [{"asset_id": "Snow013", "error": "simulated package failure"}]


@pytest.mark.parametrize("value", ("", "Snow013,", "../Snow013", "Snow013,snow013", "Snow013/Snow014"))
def test_unsafe_or_duplicate_ids_rejected(value):
    with pytest.raises(ValueError, match="unique official"): download.asset_ids(value)
