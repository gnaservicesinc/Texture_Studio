"""Usable dataset creation/import/edit/removal preserve exact original bytes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import material_workbench as workbench
from material_dataset import file_sha256, read_png, write_json, write_png
REAL_TRAINING_ACTIVE = workbench.training_active


def args(**values):
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)


def create(path, name="My first dataset", description="Original detail"):
    return workbench.create_dataset(args(dataset=path, name=name, description=description))


def maps(path, size=256, provider=False, family="surface", input_bits=16):
    path.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    codes = ((xx + yy * 37) % 65536).astype(np.uint16)
    arrays = {"input": np.stack((codes, codes // 2, 65535 - codes), -1),
              "height": codes[..., None], "roughness": (65535 - codes)[..., None],
              "normal": np.repeat(codes[..., None], 3, -1)}
    if input_bits == 8:
        arrays["input"] = (arrays["input"] % 256).astype(np.uint8)
    suffix = {"input": "diff", "height": "disp", "roughness": "rough", "normal": "nor_gl"}
    result = {}
    for role, array in arrays.items():
        filename = f"{family}_{suffix[role]}_2k.png" if provider else role + ".png"
        result[role] = path / filename
        write_png(result[role], array)
    return result


def snapshot(path):
    return {str(item): item.read_bytes() for item in path.rglob("*") if item.is_file()}


def add(path, source_maps, name="Surface", expected=None, **options):
    return workbench.add_material(args(dataset=path, name=name, expected_index_sha256=expected,
                                       **source_maps, **options))


def test_create_empty_named_dataset_and_open_folder_or_manifest(tmp_path):
    path = tmp_path / "Datasets" / "My first dataset"
    result = create(path, name="  新しい dataset  ", description="Line one\nLine two")
    assert result["created"] and result["materials"] == []
    assert result["name"] == "新しい dataset" and result["description"] == "Line one\nLine two"
    assert result["material_count"] == result["sample_count"] == 0
    assert result["supported_training_sizes"] == []
    assert result["dataset_id"] and result["created_utc"] == result["updated_utc"]
    assert workbench.dataset_info(args(dataset=path / "dataset.json"))["dataset_id"] == result["dataset_id"]
    assert not (path / "sources").exists()
    assert result["dataset_management"]["owns_directory"]


def test_create_can_use_empty_folder_but_never_overwrites_existing_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    create(empty)
    before = snapshot(empty)
    with pytest.raises(ValueError, match="already contains"):
        create(empty, name="Different")
    assert snapshot(empty) == before
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "photo.png").write_bytes(b"untouched")
    with pytest.raises(ValueError, match="already contains"):
        create(other)
    assert (other / "photo.png").read_bytes() == b"untouched"


@pytest.mark.parametrize("name", ["", "   ", "a\nb", "x" * 201])
def test_invalid_dataset_name_is_rejected_before_creating_files(tmp_path, name):
    with pytest.raises(ValueError, match="name"):
        create(tmp_path / "dataset", name=name)
    assert not (tmp_path / "dataset").exists()


def test_dataset_info_accepts_empty_sources_folder(tmp_path):
    (tmp_path / "sources").mkdir()
    write_json(tmp_path / "dataset.json", {"schema_version": 2, "samples": []})
    assert workbench.dataset_info(args(dataset=tmp_path))["materials"] == []


def test_edit_dataset_changes_display_name_description_with_stale_hash_protection(tmp_path):
    path = tmp_path / "dataset"
    before = create(path)
    result = workbench.edit_dataset(args(dataset=path / "dataset.json", name="Renamed dataset",
        description="Useful notes\nStill original data", expected_index_sha256=before["index_sha256"]))
    assert result["dataset_path"] == str(path) and result["name"] == "Renamed dataset"
    assert result["description"] == "Useful notes\nStill original data"
    assert result["dataset_id"] == before["dataset_id"] and result["created_utc"] == before["created_utc"]
    assert result["updated_utc"] >= before["updated_utc"]
    content = (path / "dataset.json").read_bytes()
    with pytest.raises(ValueError, match="changed since"):
        workbench.edit_dataset(args(dataset=path, name="Lost edit", description="stale",
                                    expected_index_sha256=before["index_sha256"]))
    assert (path / "dataset.json").read_bytes() == content


def test_add_material_keeps_full_native_dimensions_precision_and_exact_external_files(tmp_path):
    source = tmp_path / "original maps"
    source_maps = maps(source, input_bits=8)
    source_bytes = snapshot(source)
    path = tmp_path / "dataset"
    before = create(path)
    result = add(path, source_maps, name="精密 surface", expected=before["index_sha256"])
    assert result["added_material_count"] == result["material_count"] == 1
    sample = result["materials"][0]["samples"][0]
    assert sample["width"] == sample["height"] == 256
    assert sample["maps"]["input"]["source_bits"] == 8
    assert sample["maps"]["height"]["source_bits"] == 16
    assert all(Path(item["path"]).is_relative_to(source) for item in sample["maps"].values())
    assert sample["available_targets"] == ["height", "roughness", "normal"]
    assert not list(path.rglob("*.png")) and snapshot(source) == source_bytes
    record = json.loads(Path(sample["metadata_path"]).read_text())
    assert all(details["transforms"] == [] for details in record["map_metadata"].values())
    assert result["source_bytes_modified"] is False


def test_add_requires_registered_native_map_dimensions_and_does_not_publish_partial_dataset(tmp_path):
    source_maps = maps(tmp_path / "originals")
    source_maps["height"].unlink()
    write_png(source_maps["height"], np.zeros((128, 128, 1), np.uint16))
    path = tmp_path / "dataset"
    before = create(path)
    with pytest.raises(ValueError, match="same native dimensions"):
        add(path, source_maps, expected=before["index_sha256"])
    assert not (path / "samples").exists()
    assert file_sha256(path / "dataset.json") == before["index_sha256"]


def test_add_requires_color_and_target_and_native_rgb_normals(tmp_path):
    source_maps = maps(tmp_path / "originals")
    path = tmp_path / "dataset"
    create(path)
    with pytest.raises(ValueError, match="at least one"):
        add(path, {"input": source_maps["input"]})
    source_maps["normal"].unlink()
    write_png(source_maps["normal"], np.zeros((256, 256, 1), np.uint16))
    with pytest.raises(ValueError, match="RGB"):
        add(path, source_maps)
    assert workbench.dataset_info(args(dataset=path))["sample_count"] == 0


def test_duplicate_add_skips_same_bytes_and_conflicting_identity_is_actionable(tmp_path):
    source_maps = maps(tmp_path / "originals")
    path = tmp_path / "dataset"
    create(path)
    first = add(path, source_maps)
    repeated = add(path, source_maps, expected=first["index_sha256"])
    assert repeated["added_material_count"] == 0 and repeated["duplicate_material_count"] == 1
    assert repeated["index_sha256"] == first["index_sha256"]
    other = maps(tmp_path / "other")
    other["height"].unlink()
    write_png(other["height"], np.ones((256, 256, 1), np.uint16))
    with pytest.raises(ValueError, match="another material name"):
        add(path, other)
    assert workbench.dataset_info(args(dataset=path))["material_count"] == 1


def test_add_stale_selection_preserves_all_metadata(tmp_path):
    path = tmp_path / "dataset"
    first = create(path)
    workbench.edit_dataset(args(dataset=path, name="New name", description=None,
                                expected_index_sha256=first["index_sha256"]))
    before = snapshot(path)
    with pytest.raises(ValueError, match="changed since"):
        add(path, maps(tmp_path / "originals"), expected=first["index_sha256"])
    assert snapshot(path) == before


def test_import_provider_asset_folder_and_enclosing_folder_are_idempotent(tmp_path):
    sources = tmp_path / "provider downloads"
    maps(sources / "stone", provider=True, family="stone")
    maps(sources / "wood", provider=True, family="wood")
    before = snapshot(sources)
    path = tmp_path / "dataset"
    first = create(path)
    result = workbench.import_folder(args(dataset=path, folder=sources / "stone", expected_index_sha256=first["index_sha256"]))
    assert result["added_material_count"] == 1
    result = workbench.import_folder(args(dataset=path, folder=sources, expected_index_sha256=result["index_sha256"]))
    assert result["added_material_count"] == 1 and result["duplicate_material_count"] == 1
    assert result["material_count"] == 2 and snapshot(sources) == before


def test_import_generic_material_folder_and_normal_convention(tmp_path):
    originals = tmp_path / "My material"
    source_maps = maps(originals)
    source_maps["normal"].rename(originals / "NormalDX.PNG")
    path = tmp_path / "dataset"
    create(path)
    result = workbench.import_folder(args(dataset=path, folder=originals, expected_index_sha256=None))
    normal = result["materials"][0]["samples"][0]["maps"]["normal"]
    assert normal["source_normal_convention"] == "directx"
    assert result["name"] == "My first dataset"


def test_import_rejects_entire_batch_before_writing_when_generic_set_is_invalid(tmp_path):
    sources = tmp_path / "imports"
    maps(sources / "good")
    bad = maps(sources / "bad")
    bad["roughness"].unlink()
    write_png(bad["roughness"], np.zeros((128, 128, 1), np.uint8))
    path = tmp_path / "dataset"
    first = create(path)
    before = snapshot(path)
    with pytest.raises(ValueError, match="same native dimensions"):
        workbench.import_folder(args(dataset=path, folder=sources, expected_index_sha256=first["index_sha256"]))
    assert snapshot(path) == before


def test_import_existing_dataset_refers_to_originals_and_changed_original_fails(tmp_path):
    original = tmp_path / "first"
    create(original)
    source_maps = maps(tmp_path / "images")
    first = add(original, source_maps)
    path = tmp_path / "second"
    before = create(path)
    imported = workbench.import_folder(args(dataset=path, folder=original, expected_index_sha256=before["index_sha256"]))
    assert imported["materials"][0]["samples"][0]["maps"] == first["materials"][0]["samples"][0]["maps"]
    source_maps["height"].unlink()
    write_png(source_maps["height"], np.ones((256, 256, 1), np.uint16))
    with pytest.raises(ValueError, match="changed since it was registered"):
        workbench.import_folder(args(dataset=path, folder=original, expected_index_sha256=imported["index_sha256"]))


def test_remove_material_unlists_all_samples_without_modifying_any_originals(tmp_path):
    original_maps = maps(tmp_path / "images")
    path = tmp_path / "dataset"
    create(path)
    added = add(path, original_maps)
    material_id = added["materials"][0]["material_id"]
    before = snapshot(tmp_path / "images")
    removed = workbench.remove_material(args(dataset=path, material=material_id, expected_index_sha256=added["index_sha256"]))
    assert removed["materials"] == [] and removed["removed_sample_count"] == 1
    assert snapshot(tmp_path / "images") == before
    assert workbench.dataset_info(args(dataset=path))["materials"] == []
    restored = add(path, original_maps, expected=removed["index_sha256"])
    assert restored["material_count"] == 1


def test_removed_legacy_material_does_not_reappear_on_source_discovery_refresh(tmp_path):
    path = tmp_path / "dataset"
    path.mkdir()
    write_json(path / "dataset.json", {"schema_version": 2, "name": "My legacy data", "description": "Retain this", "samples": []})
    source_maps = maps(path / "sources" / "surface", provider=True)
    first = workbench.dataset_info(args(dataset=path))
    assert first["name"] == "My legacy data" and first["description"] == "Retain this"
    material_id = first["materials"][0]["material_id"]
    workbench.remove_material(args(dataset=path, material=material_id, expected_index_sha256=first["index_sha256"]))
    maps(path / "sources" / "wood", provider=True, family="wood")
    result = workbench.dataset_info(args(dataset=path))
    assert [material["material_id"] for material in result["materials"]] == ["wood_2k"]
    assert result["name"] == "My legacy data" and result["description"] == "Retain this"
    assert all(source.is_file() for source in source_maps.values())
    restored = workbench.import_folder(args(dataset=path, folder=path / "sources" / "surface", expected_index_sha256=result["index_sha256"]))
    assert restored["material_count"] == 2


def test_grid_inspector_retains_materials_too_small_to_crop_and_can_edit_them(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    first = add(path, maps(tmp_path / "small", size=128), name="Small")
    add(path, maps(tmp_path / "big", size=256), name="Big", expected=first["index_sha256"])
    result = workbench.dataset_info(args(dataset=path, review_size=256))
    assert result["material_count"] == 2
    small = next(material["samples"][0] for material in result["materials"] if material["name"] == "Small")
    assert small["width"] == 128
    workbench.curate(args(dataset=path, sample=small["sample_id"], status="excluded", split=None,
        note="Original detail too small", review_size=256, expected_index_sha256=result["index_sha256"]))
    selected = next(material["samples"][0] for material in workbench.dataset_info(args(dataset=path, review_size=256))["materials"] if material["name"] == "Small")
    assert selected["status"] == "excluded" and selected["note"] == "Original detail too small"


def test_managed_dataset_trash_scope_is_whole_folder_only_when_all_originals_are_external(tmp_path):
    path = tmp_path / "dataset"
    first = create(path)
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=first["index_sha256"]))
    assert deletion["safe_to_trash_folder"] and deletion["trash_paths"] == [str(path)]
    added = add(path, maps(tmp_path / "originals"))
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=added["index_sha256"]))
    assert deletion["safe_to_trash_folder"]
    (path / "unrelated.txt").write_bytes(b"preserve unrelated data")
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=added["index_sha256"]))
    assert not deletion["safe_to_trash_folder"] and deletion["trash_paths"] == [str(path / "dataset.json")]


def test_legacy_dataset_trash_scope_preserves_internal_originals(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    write_json(path / "dataset.json", {"schema_version": 2, "samples": []})
    maps(path / "sources" / "surface", provider=True)
    first = workbench.dataset_info(args(dataset=path))
    source_bytes = snapshot(path / "sources")
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=first["index_sha256"]))
    assert not deletion["safe_to_trash_folder"] and deletion["trash_paths"] == [str(path / "dataset.json")]
    assert deletion["original_sources_preserved"] and snapshot(path / "sources") == source_bytes


def test_active_training_and_preparation_block_dataset_management(tmp_path, monkeypatch):
    path = tmp_path / "dataset"
    first = create(path)
    monkeypatch.setattr(workbench, "training_active", lambda _path: True)
    for action, fields in ((workbench.edit_dataset, {"name": "Rename", "description": ""}),
                           (workbench.remove_material, {"material": "surface_2k"}),
                           (workbench.validate_delete, {})):
        with pytest.raises(ValueError, match="Stop active"):
            action(args(dataset=path, expected_index_sha256=first["index_sha256"], **fields))
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)
    (path / ".training-data" / ".preparing-test").mkdir(parents=True)
    write_json(path / ".training-data" / ".preparing-test" / ".native-crop-preparation.json",
               {"source_dataset_path": str(path), "pid": os.getpid()})
    with pytest.raises(ValueError, match="cancel active"):
        workbench.validate_delete(args(dataset=path, expected_index_sha256=first["index_sha256"]))


def test_validate_delete_stale_hash_and_workspace_root_are_rejected(tmp_path):
    path = tmp_path / "dataset"
    first = create(path)
    workbench.edit_dataset(args(dataset=path, name="Renamed", description=None, expected_index_sha256=first["index_sha256"]))
    with pytest.raises(ValueError, match="changed since"):
        workbench.validate_delete(args(dataset=path, expected_index_sha256=first["index_sha256"]))
    with pytest.raises(ValueError, match="protected workspace"):
        workbench.validate_delete(args(dataset=ROOT, expected_index_sha256=None))


def test_prepared_dataset_inherits_original_information_and_cannot_be_managed_as_original(tmp_path):
    path = tmp_path / "dataset"
    create(path, name="Useful dataset", description="Keep full source precision")
    first = add(path, maps(tmp_path / "originals"))
    prepared = workbench.prepare_size(args(dataset=path, size=256, expected_index_sha256=first["index_sha256"], automatic_validation=False))
    stage = Path(prepared["dataset_path"])
    assert prepared["name"] == "Useful dataset" and prepared["description"] == "Keep full source precision"
    assert prepared["original_dataset_path"] == str(path)
    with pytest.raises(ValueError, match="Open the original"):
        workbench.edit_dataset(args(dataset=stage, name="Wrong dataset", description=None, expected_index_sha256=prepared["index_sha256"]))


def test_interrupted_new_material_publication_recovers_missing_sample_file_transaction(tmp_path, monkeypatch):
    path = tmp_path / "dataset"
    first = create(path)
    source_maps = maps(tmp_path / "originals")
    source_bytes = snapshot(tmp_path / "originals")
    actual = workbench.atomic_bytes
    calls = []
    def interrupted(target, content):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("Interrupted before new sample publication")
        return actual(target, content)
    monkeypatch.setattr(workbench, "atomic_bytes", interrupted)
    with pytest.raises(OSError, match="Interrupted"):
        add(path, source_maps, expected=first["index_sha256"])
    assert (path / workbench.JOURNAL).is_file()
    monkeypatch.setattr(workbench, "atomic_bytes", actual)
    recovered = workbench.dataset_info(args(dataset=path))
    assert recovered["material_count"] == 1 and not (path / workbench.JOURNAL).exists()
    assert snapshot(tmp_path / "originals") == source_bytes


def test_management_cli_is_one_json_object_for_create_edit_import_remove_and_delete(tmp_path):
    script = ROOT / "scripts" / "material_workbench.py"
    path = tmp_path / "dataset"
    def run(command, *extra):
        result = subprocess.run([sys.executable, "-B", str(script), command, "--dataset", str(path), *extra],
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        decoded = json.loads(result.stdout)
        assert decoded["ok"] and decoded["command"] == command
        return decoded
    first = run("create-dataset", "--name", "Dataset", "--description", "Description")
    edited = run("edit-dataset", "--name", "Rename", "--expected-index-sha256", first["index_sha256"], "--review-size", "1024")
    originals = tmp_path / "maps"
    maps(originals)
    imported = run("import-folder", "--folder", str(originals), "--expected-index-sha256", edited["index_sha256"], "--review-size", "1024")
    assert imported["material_count"] == 1
    removed = run("remove-material", "--material", imported["materials"][0]["material_id"], "--expected-index-sha256", imported["index_sha256"], "--review-size", "1024")
    deletion = run("validate-delete", "--expected-index-sha256", removed["index_sha256"])
    assert deletion["trash_paths"] == [str(path)]


def test_manual_validation_choice_survives_automatic_preparation_and_cache_cleanup(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    first = add(path, maps(tmp_path / "originals"))
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note="Manual holdout", review_size=256, expected_index_sha256=first["index_sha256"],
        expected_review_sha256=selected["review_sha256"]))
    after = workbench.dataset_info(args(dataset=path, review_size=256))
    prepared = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
        expected_index_sha256=after["index_sha256"], expected_review_sha256=after["review_sha256"]))
    assert prepared["materials"][0]["samples"][0]["split"] == "validation"
    assert prepared["automatic_validation"]["source_family_ids"] == ["surface"]
    workbench.cleanup_size(args(dataset=Path(prepared["dataset_path"])))
    repeated = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
                                         expected_index_sha256=after["index_sha256"]))
    assert repeated["materials"][0]["samples"][0]["split"] == "validation"


def test_manual_assignment_propagates_across_resolution_siblings_to_prevent_leakage(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    add(path, maps(tmp_path / "low", size=256), name="Same family")
    add(path, maps(tmp_path / "high", size=512), name="Same family")
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note=None, review_size=256, expected_index_sha256=selected["index_sha256"],
        expected_review_sha256=selected["review_sha256"]))
    after = workbench.dataset_info(args(dataset=path, review_size=256))
    assert all(sample["split"] == "validation" for material in after["materials"] for sample in material["samples"])
    prepared = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
        expected_index_sha256=after["index_sha256"], expected_review_sha256=after["review_sha256"]))
    assert all(sample["split"] == "validation" for material in prepared["materials"] for sample in material["samples"])


def test_manual_region_assignment_survives_automatic_assignment_without_overlapping_pixels(tmp_path, monkeypatch):
    import material_native_size as native
    original_layout = native.crop_layout
    def small_disjoint_layout(width, height, size):
        if (width, height, size) == (512, 512, 256):
            return [("top_left", [0, 0, 256, 256]), ("top_right", [256, 0, 256, 256]),
                    ("bottom_right", [256, 256, 256, 256])]
        return original_layout(width, height, size)
    monkeypatch.setattr(native, "crop_layout", small_disjoint_layout)
    path = tmp_path / "dataset"
    create(path)
    add(path, maps(tmp_path / "originals", size=512))
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    region = next(sample for sample in selected["materials"][0]["samples"] if sample["sample_id"].endswith("bottom_right"))
    workbench.curate(args(dataset=path, sample=region["sample_id"], status="approved", split="train", note=None,
        review_size=256, expected_index_sha256=selected["index_sha256"], expected_review_sha256=selected["review_sha256"]))
    result = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True, expected_index_sha256=None))
    assert all(sample["split"] == "train" for sample in result["materials"][0]["samples"])
    assert result["automatic_validation"]["source_family_ids"] == []


def test_stale_size_review_and_prepare_selection_reject_lost_update_even_when_index_is_same(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    add(path, maps(tmp_path / "originals"))
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    review_args = dict(dataset=path, sample=sample["sample_id"], status="approved", split=None,
        review_size=256, expected_index_sha256=selected["index_sha256"], expected_review_sha256=selected["review_sha256"])
    workbench.curate(args(note="First review", **review_args))
    with pytest.raises(ValueError, match="reviews changed since"):
        workbench.curate(args(note="Stale review", **review_args))
    with pytest.raises(ValueError, match="reviews changed since"):
        workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
            expected_index_sha256=selected["index_sha256"], expected_review_sha256=selected["review_sha256"]))
    after = workbench.dataset_info(args(dataset=path, review_size=256))
    assert after["index_sha256"] == selected["index_sha256"]
    assert after["materials"][0]["samples"][0]["note"] == "First review"


def test_reusing_removed_material_name_with_different_originals_drops_review_approval(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    original_maps = maps(tmp_path / "first sources")
    add(path, original_maps)
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note="This approval belongs to the first pixels", review_size=256, expected_index_sha256=selected["index_sha256"]))
    # A cached prepared dataset holds old review metadata as well.
    workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True, expected_index_sha256=None))
    workbench.remove_material(args(dataset=path, material=selected["materials"][0]["material_id"],
                                   expected_index_sha256=selected["index_sha256"]))
    different = maps(tmp_path / "different sources")
    different["height"].unlink()
    write_png(different["height"], np.ones((256, 256, 1), np.uint16))
    added = add(path, different, review_size=256)
    new_sample = added["materials"][0]["samples"][0]
    assert new_sample["status"] == "unreviewed" and new_sample["note"] is None and new_sample["split"] == "train"
    prepared = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True, expected_index_sha256=None))
    fresh = prepared["materials"][0]["samples"][0]
    assert fresh["status"] == "unreviewed" and fresh["note"] is None and fresh["split"] == "train"


def test_restoring_exact_originals_preserves_their_review(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    original_maps = maps(tmp_path / "originals")
    add(path, original_maps)
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note="Verified original", review_size=256, expected_index_sha256=selected["index_sha256"]))
    workbench.remove_material(args(dataset=path, material=selected["materials"][0]["material_id"], expected_index_sha256=None))
    restored = add(path, original_maps, review_size=256)
    sample = restored["materials"][0]["samples"][0]
    assert sample["status"] == "approved" and sample["note"] == "Verified original" and sample["split"] == "validation"


def test_dead_preparation_stage_does_not_trap_dataset_deletion(tmp_path):
    path = tmp_path / "dataset"
    selected = create(path)
    stage = path / ".training-data" / ".preparing-abandoned"
    stage.mkdir(parents=True)
    write_json(stage / ".native-crop-preparation.json", {"source_dataset_path": str(path), "pid": 999999999})
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=selected["index_sha256"]))
    assert not deletion["safe_to_trash_folder"] and deletion["trash_paths"] == [str(path / "dataset.json")]
    assert stage.is_dir()


def test_dataset_management_operates_without_pytorch_installed(tmp_path):
    script = ROOT / "scripts" / "material_workbench.py"
    dataset = tmp_path / "dataset"
    bootstrap = """import importlib.abc, runpy, sys
class MissingTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('PyTorch is intentionally unavailable for dataset management')
sys.meta_path.insert(0, MissingTorch())
sys.path.insert(0, sys.argv[1])
script = sys.argv[2]
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(ROOT / "scripts"), str(script),
                             "create-dataset", "--dataset", str(dataset), "--name", "No training framework"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["name"] == "No training framework"
    assert "torch" not in result.stderr


def test_editing_original_below_selected_grid_persists_to_eligible_training_size(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    add(path, maps(tmp_path / "originals", size=256))
    selected = workbench.dataset_info(args(dataset=path, review_size=1024))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note="User edited the original material", review_size=1024, expected_index_sha256=selected["index_sha256"],
        expected_review_sha256=selected["review_sha256"]))
    eligible = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = eligible["materials"][0]["samples"][0]
    assert sample["split"] == "validation" and sample["status"] == "approved"
    assert sample["note"] == "User edited the original material"
    prepared = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
                                          expected_index_sha256=eligible["index_sha256"]))
    assert prepared["materials"][0]["samples"][0]["split"] == "validation"


def test_new_source_review_survives_cleanup_of_older_unchanged_prepared_cache(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    add(path, maps(tmp_path / "originals"))
    old = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True, expected_index_sha256=None))
    selected = workbench.dataset_info(args(dataset=path, review_size=256))
    sample = selected["materials"][0]["samples"][0]
    workbench.curate(args(dataset=path, sample=sample["sample_id"], status="approved", split="validation",
        note="New source-side review wins", review_size=256, expected_index_sha256=selected["index_sha256"],
        expected_review_sha256=selected["review_sha256"]))
    current = workbench.dataset_info(args(dataset=path, review_size=256))
    prepared = workbench.prepare_size(args(dataset=path, size=256, automatic_validation=True,
        expected_index_sha256=current["index_sha256"], expected_review_sha256=current["review_sha256"]))
    sample = prepared["materials"][0]["samples"][0]
    assert sample["status"] == "approved" and sample["split"] == "validation"
    assert sample["note"] == "New source-side review wins"
    assert not Path(old["dataset_path"]).exists()


def test_finder_metadata_does_not_prevent_owned_dataset_folder_trash(tmp_path):
    path = tmp_path / "dataset"
    selected = create(path)
    (path / ".DS_Store").write_bytes(b"Finder window metadata")
    deletion = workbench.validate_delete(args(dataset=path, expected_index_sha256=selected["index_sha256"]))
    assert deletion["safe_to_trash_folder"] and deletion["trash_paths"] == [str(path)]


@pytest.mark.parametrize("alias", ["dataset.json", "sources", ".training-data/256-cached"])
def test_active_process_check_recognizes_dataset_manifest_source_and_cache_aliases(tmp_path, monkeypatch, alias):
    path = tmp_path / "dataset"
    create(path)
    (path / "sources").mkdir()
    selected = path / alias
    command = f"{os.getpid() + 1000} python /runtime/train_material_pbrnxt.py --dataset '{selected}'"
    monkeypatch.setattr(workbench.subprocess, "run", lambda *_a, **_kw: args(stdout=command))
    assert REAL_TRAINING_ACTIVE(path)


def test_same_originals_added_under_another_name_are_not_duplicated_or_split_as_independent_family(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    original_maps = maps(tmp_path / "originals")
    first = add(path, original_maps, name="Original name")
    second = add(path, original_maps, name="Another name", expected=first["index_sha256"])
    assert second["added_material_count"] == 0 and second["duplicate_material_count"] == 1
    assert second["material_count"] == 1 and second["index_sha256"] == first["index_sha256"]
    # Independent file bindings remain separate even when all numeric codes
    # happen to be identical (for example two uniformly flat references).
    independent = add(path, maps(tmp_path / "independent originals"), name="Independent material")
    assert independent["material_count"] == 2 and independent["added_material_count"] == 1


def test_duplicate_original_set_detection_handles_older_records_without_color_variant_list(tmp_path):
    path = tmp_path / "dataset"
    create(path)
    original_maps = maps(tmp_path / "originals")
    first = add(path, original_maps)
    sample_path = Path(first["materials"][0]["samples"][0]["metadata_path"])
    sample = json.loads(sample_path.read_text())
    sample.pop("input_variants")
    write_json(sample_path, sample)
    duplicated = add(path, original_maps, name="Another name")
    assert duplicated["material_count"] == 1 and duplicated["duplicate_material_count"] == 1
