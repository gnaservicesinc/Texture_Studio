"""Native JSON bridge selection, numeric data and safe metadata ownership."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "verification"))
import material_workbench as workbench
from frozen_dino_height import MODEL_SHA256, CODE_REVISION
from material_dataset import write_png
from material_training_cycle import MaterialMapHead
from test_material_training_cycle import dataset_with_two_corners
from train_material_height import digest


def args(**values):
    return SimpleNamespace(**values)


def checkpoint(root, role="height"):
    model = MaterialMapHead(4, 768, 4, role)
    with torch.no_grad():
        model.head.weight.fill_(0.01)
    path = root / "checkpoint.pt"
    torch.save({"schema": workbench.CYCLE_SCHEMA, "head_config": {"base_channels": 4, "feature_channels": 768, "projection_channels": 4, "target": role}, "head_state": model.state_dict(), "target": role, "encoder": {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION}, "step": 7, "encoder_size": 28, "optimizer_state": {"large-unused-state": torch.ones(100)}}, path)
    return path


def test_dataset_selection_exact_ids_paths_precision_including_excluded(tmp_path):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    result = workbench.dataset_info(args(dataset=dataset))
    assert result["dataset_path"] == str(dataset.resolve())
    assert len(result["materials"]) == 4
    assert all(len(item["samples"]) == 3 for item in result["materials"])
    crop = result["materials"][0]["samples"][0]
    assert crop["width"] == crop["height"] == 32
    assert crop["maps"]["height"]["sha256"] == digest(Path(crop["maps"]["height"]["path"]))
    assert workbench.dataset_info(args(dataset=dataset / "dataset.json"))["dataset_path"] == str(dataset.resolve())


def test_curate_exact_selected_crop_preserves_all_image_bytes(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)
    images = {str(p): digest(p) for p in dataset.rglob("*.png")}
    before = workbench.dataset_info(args(dataset=dataset))
    chosen = before["materials"][2]["samples"][1]["sample_id"]
    result = workbench.curate(args(dataset=dataset, sample=chosen, status="excluded", split=None,
        note="Review this crop, leave other crops unchanged", expected_index_sha256=before["index_sha256"]))
    after = workbench.dataset_info(args(dataset=dataset))
    statuses = {s["sample_id"]: s["status"] for m in after["materials"] for s in m["samples"]}
    assert statuses[chosen] == "excluded" and list(statuses.values()).count("excluded") == 1
    assert result["source_bytes_modified"] is False
    assert images == {str(p): digest(p) for p in dataset.rglob("*.png")}
    assert not (dataset / workbench.JOURNAL).exists()
    with pytest.raises(ValueError, match="changed since"):
        workbench.curate(args(dataset=dataset, sample=chosen, status="approved", split=None,
            note=None, expected_index_sha256=before["index_sha256"]))
    assert workbench.curate(args(dataset=dataset, sample=chosen, status="approved", split=None,
        note=None, expected_index_sha256=after["index_sha256"]))["status"] == "approved"


def test_curate_refuses_active_training_and_invalid_split_without_writes(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    choice = args(dataset=dataset, sample="white_stucco_02_auto_003", status="approved", split="train", note=None, expected_index_sha256=None)
    files = {str(p): digest(p) for p in dataset.rglob("sample.json")}
    monkeypatch.setattr(workbench, "training_active", lambda _path: True)
    with pytest.raises(ValueError, match="Stop active"):
        workbench.curate(choice)
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)
    with pytest.raises(ValueError, match="validation regions"):
        workbench.curate(choice)
    assert files == {str(p): digest(p) for p in dataset.rglob("sample.json")}


def test_interrupted_metadata_transaction_recovers_only_json(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)
    actual = workbench.atomic_bytes
    calls = {"count": 0}
    def interrupted(path, value):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("interrupted before dataset index publication")
        return actual(path, value)
    monkeypatch.setattr(workbench, "atomic_bytes", interrupted)
    with pytest.raises(OSError):
        workbench.curate(args(dataset=dataset, sample="white_stucco_02_auto_001", status="approved",
            split=None, note="retained", expected_index_sha256=None))
    assert (dataset / workbench.JOURNAL).is_file()
    monkeypatch.setattr(workbench, "atomic_bytes", actual)
    recovered = workbench.dataset_info(args(dataset=dataset))
    sample = next(s for m in recovered["materials"] for s in m["samples"] if s["sample_id"] == "white_stucco_02_auto_001")
    assert sample["status"] == "approved" and sample["note"] == "retained"
    assert not (dataset / workbench.JOURNAL).exists()


def test_checkpoint_schema_hash_shapes_no_silent_fallback(tmp_path):
    path = checkpoint(tmp_path)
    info = workbench.checkpoint_info(args(checkpoint=path, expected_sha256=digest(path)))
    assert info["step"] == 7 and info["target"] == "height" and info["compatible"]
    with pytest.raises(ValueError, match="SHA256"):
        workbench.checkpoint_snapshot(path, "0" * 64)
    payload = torch.load(path, weights_only=True)
    payload["schema"] = "unknown-depth-model"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported"):
        workbench.checkpoint_snapshot(path)


def test_historical_frozen_diagnostic_package_remains_loadable(tmp_path):
    path = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=workbench.DIAGNOSTIC_SCHEMA, architecture=workbench.ARCHITECTURE,
        model_config={key: value for key, value in payload.pop("head_config").items() if key != "target"},
        model_state=payload.pop("head_state"), variant="frozen_features")
    torch.save(payload, path)
    result = workbench.package(args(checkpoint=path, expected_sha256=digest(path), output=tmp_path / "package"))
    normalized, _ = workbench.checkpoint_snapshot(Path(result["checkpoint_path"]))
    assert normalized["schema"] == workbench.CYCLE_SCHEMA
    assert normalized["source_checkpoint_schema"] == workbench.DIAGNOSTIC_SCHEMA
    assert normalized["target"] == "height" and "model_state" not in normalized


def fake_encoder_loader(_model, _code, device):
    return torch.nn.Linear(1, 1).requires_grad_(False).to(device), {"checkpoint_sha256": MODEL_SHA256}


def test_inference_native_float_exr_and_compact_package_roundtrip(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    image = tmp_path / "input.png"
    codes = (np.indices((32, 40))[0] * 300 + 8000).astype(np.uint16)
    write_png(image, np.repeat(codes[..., None], 3, axis=-1))
    original = image.read_bytes()
    monkeypatch.setattr(workbench, "extract_features", lambda _encoder, source, _size: torch.ones(1, 768, 2, 2, device=source.device))
    result = workbench.infer(args(checkpoint=path, expected_sha256=digest(path), image=image,
        input_encoding="linear", output=tmp_path / "inference", device="cpu", model_directory=tmp_path,
        code_directory=tmp_path), encoder_loader=fake_encoder_loader)
    assert result["native_dimensions"] == [40, 32]
    assert set(result["outputs"]) == {"height", "normal"}
    import OpenEXR
    height = OpenEXR.File(result["outputs"]["height"]["path"], separate_channels=True).channels()["Y"].pixels
    assert height.shape == (32, 40) and height.dtype == np.float32
    assert image.read_bytes() == original and not list((tmp_path / "inference").glob("*.npy"))
    packed = workbench.package(args(checkpoint=path, expected_sha256=digest(path), output=tmp_path / "package"))
    decoded, _ = workbench.checkpoint_snapshot(Path(packed["checkpoint_path"]))
    assert decoded["target"] == "height" and "optimizer_state" not in decoded
    assert not packed["encoder_included"]
    assert set(p.name for p in (tmp_path / "package").iterdir()) == {"material-head.pt", "config.json", "README.md", "LICENSE", "SHA256SUMS.json"}
    assert (tmp_path / "package/LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    from material_training_cycle import load_cycle_head
    reloaded, provenance = load_cycle_head(Path(packed["checkpoint_path"]), torch.device("cpu"), packed["checkpoint_sha256"])
    assert reloaded.target == "height" and provenance["checkpoint_sha256"] == packed["checkpoint_sha256"]
    assert provenance["optimizer_reset"]


def test_photo_size_never_silently_reduced(tmp_path):
    path = tmp_path / "large.png"
    write_png(path, np.zeros((16, 2049, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="never silently"):
        workbench.load_photo(path, "srgb")


def test_upload_requires_exact_private_package_and_saved_auth_without_network(tmp_path):
    path = checkpoint(tmp_path)
    package = tmp_path / "package"
    workbench.package(args(checkpoint=path, expected_sha256=None, output=package))
    calls = []
    class FakeAPI:
        def whoami(self):
            calls.append("saved-auth")
        def create_repo(self, **kwargs):
            calls.append(kwargs)
        def repo_info(self, **kwargs):
            return args(private=True)
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return args(commit_url="https://huggingface.co/test/head/commit/verified")
    result = workbench.upload_package(args(package=package, repo="test/head", public=False), FakeAPI)
    assert result["private"] and not result["source_photos_uploaded"]
    assert calls[1]["private"] and "folder_path" in calls[2]
    (package / "extra-photo.png").write_bytes(b"Never upload this file")
    with pytest.raises(ValueError, match="bounded"):
        workbench.upload_package(args(package=package, repo="test/head", public=False), FakeAPI)


def test_owned_encoder_remove_refuses_external_and_added_files(tmp_path, monkeypatch):
    # Tiny pinned-fixture download tests never fetch model weights or network.
    files = {"model/model.safetensors": b"fake model", "model/config.json": b"{}", "code/LICENSE": b"fake Apache license"}
    descriptions = {name: {"url": "https://test.invalid/" + name, "sha256": __import__("hashlib").sha256(value).hexdigest(), "bytes": len(value)} for name, value in files.items()}
    monkeypatch.setattr(workbench, "encoder_files", lambda: descriptions)
    def download(url, path, checksum, maximum, exact):
        relative = url.removeprefix("https://test.invalid/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[relative])
    monkeypatch.setattr(workbench, "download_checked", download)
    destination = tmp_path / "managed"
    result = workbench.install_encoder(args(destination=destination))
    assert not result["reused"] and workbench.install_encoder(args(destination=destination))["reused"]
    extra = destination / "my-own-unique-model.pt"
    extra.write_bytes(b"protect")
    with pytest.raises(ValueError, match="unrecognized"):
        workbench.remove_encoder(args(directory=destination))
    assert extra.read_bytes() == b"protect"
    extra.unlink()
    assert workbench.remove_encoder(args(directory=destination))["removed"] and not destination.exists()
    external = tmp_path / "external"
    external.mkdir()
    with pytest.raises(FileNotFoundError):
        workbench.remove_encoder(args(directory=external))


def test_cli_has_one_json_object_for_success_and_failure(tmp_path):
    path = checkpoint(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/material_workbench.py"), "checkpoint", "--checkpoint", str(path)], text=True, capture_output=True)
    response = json.loads(result.stdout)
    assert result.returncode == 0 and response["ok"] and response["protocol_schema"] == workbench.SCHEMA
    bad = subprocess.run([sys.executable, str(ROOT / "scripts/material_workbench.py"), "checkpoint", "--checkpoint", str(path), "--expected-sha256", "0" * 64], text=True, capture_output=True)
    response = json.loads(bad.stdout)
    assert bad.returncode == 1 and not response["ok"] and response["error"]["code"] == "ValueError"


def test_full_rank8_lora_package_identity_and_frozen_training_refusal(tmp_path):
    path = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=workbench.ADAPTATION_SCHEMA, variant="lora", adapter_state={})
    payload["head_config"].pop("target")
    for index in range(12):
        for part, outgoing in (("qkv", 2304), ("proj", 768)):
            payload["adapter_state"][f"blocks.{index}.attn.{part}"] = {"lora_A": torch.full((8, 768), 0.01), "lora_B": torch.full((outgoing, 8), 0.02)}
    torch.save(payload, path)
    exported = workbench.package(args(checkpoint=path, expected_sha256=None, output=tmp_path / "lora"))
    inspected, _ = workbench.checkpoint_snapshot(Path(exported["checkpoint_path"]))
    assert inspected["variant"] == "lora" and len(inspected["adapter_state"]) == 24
    assert json.loads((tmp_path / "lora/config.json").read_text())["has_rank8_attention_adapters"]
    from material_training_cycle import load_cycle_head
    with pytest.raises(ValueError, match="adapted encoder"):
        load_cycle_head(Path(exported["checkpoint_path"]), torch.device("cpu"))
    broken = copy.deepcopy(payload)
    broken["adapter_state"].pop("blocks.0.attn.qkv")
    torch.save(broken, path)
    with pytest.raises(ValueError, match="LoRA layer"):
        workbench.checkpoint_snapshot(path)


def test_curation_journal_never_writes_map_or_outside_path(tmp_path):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    image = next(dataset.rglob("*.png"))
    raw = b"invalid replacement"
    import base64,hashlib
    journal = {"schema": workbench.SCHEMA, "changes": [{"path": str(image.relative_to(dataset)), "old_sha256": digest(image), "new_sha256": hashlib.sha256(raw).hexdigest(), "new_base64": base64.b64encode(raw).decode()}]}
    (dataset / workbench.JOURNAL).write_text(json.dumps(journal))
    original = image.read_bytes()
    with pytest.raises(ValueError, match="only edit"):
        workbench.recover_journal(dataset)
    assert image.read_bytes() == original


def test_recovery_refuses_third_party_json_and_preserves_partial_transaction(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset_with_two_corners(dataset)
    monkeypatch.setattr(workbench, "training_active", lambda _path: False)
    actual = workbench.atomic_bytes
    calls = {"count": 0}
    def interrupted(path, value):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("power-loss boundary")
        actual(path, value)
    monkeypatch.setattr(workbench, "atomic_bytes", interrupted)
    with pytest.raises(OSError):
        workbench.curate(args(dataset=dataset, sample="white_stucco_02_auto_001", status="approved", split=None, note=None, expected_index_sha256=None))
    transaction = (dataset / workbench.JOURNAL).read_bytes()
    index = dataset / "dataset.json"
    old = index.read_bytes()
    index.write_bytes(old + b" ")
    outsider = index.read_bytes()
    monkeypatch.setattr(workbench, "atomic_bytes", actual)
    with pytest.raises(ValueError, match="identity changed"):
        workbench.recover_journal(dataset)
    assert index.read_bytes() == outsider and (dataset / workbench.JOURNAL).read_bytes() == transaction


def test_atomic_metadata_publication_syncs_destination_directory(tmp_path, monkeypatch):
    observed = []
    actual = workbench.sync_directory
    monkeypatch.setattr(workbench, "sync_directory", lambda path: (observed.append(path), actual(path))[1])
    path = tmp_path / "sample.json"
    workbench.atomic_bytes(path, b"{}")
    assert observed == [tmp_path] and path.read_bytes() == b"{}"


def test_install_existing_unknown_empty_directory_is_never_replaced(tmp_path, monkeypatch):
    destination = tmp_path / "existing"
    destination.mkdir()
    monkeypatch.setattr(workbench, "download_checked", lambda *_args: pytest.fail("Existing unowned directory must not download"))
    with pytest.raises(FileNotFoundError):
        workbench.install_encoder(args(destination=destination))
    assert destination.is_dir() and list(destination.iterdir()) == []


def test_install_cancellation_after_first_move_rolls_back_owned_reservation(tmp_path, monkeypatch):
    import hashlib
    files = {"model/model.safetensors": b"fake weights", "model/config.json": b"{}", "code/LICENSE": b"fake license"}
    descriptions = {name: {"url": "https://test.invalid/" + name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in files.items()}
    monkeypatch.setattr(workbench, "encoder_files", lambda: descriptions)
    def download(url, path, *_arguments):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[url.removeprefix("https://test.invalid/")])
    monkeypatch.setattr(workbench, "download_checked", download)
    actual_rename = workbench.os.rename
    destination = tmp_path / "managed"
    def cancelled(source, target):
        actual_rename(source, target)
        if Path(target).parent == destination:
            raise KeyboardInterrupt("native Stop requested")
    monkeypatch.setattr(workbench.os, "rename", cancelled)
    with pytest.raises(KeyboardInterrupt, match="native Stop"):
        workbench.install_encoder(args(destination=destination))
    assert not destination.exists() and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(workbench.os, "rename", actual_rename)
    assert not workbench.install_encoder(args(destination=destination))["reused"]
    assert workbench.remove_encoder(args(directory=destination))["removed"]


def test_encoder_symlink_and_modified_owned_data_are_not_removed(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        workbench.remove_encoder(args(directory=link))
    assert real.is_dir()


def test_uploaded_visibility_conflict_is_not_silently_overridden(tmp_path):
    path = checkpoint(tmp_path)
    package = tmp_path / "package"
    workbench.package(args(checkpoint=path, expected_sha256=None, output=package))
    class ExistingPublic:
        def whoami(self):
            return {"name": "user"}
        def create_repo(self, **_kwargs):
            pass
        def repo_info(self, **_kwargs):
            return args(private=False)
        def upload_folder(self, **_kwargs):
            pytest.fail("Private upload choice must never write to an existing public repo")
    with pytest.raises(ValueError, match="visibility"):
        workbench.upload_package(args(package=package, repo="user/existing", public=False), ExistingPublic)


def test_legacy_adaptation_resolves_actual_nondefault_size_with_matching_run(tmp_path):
    variant = tmp_path / "legacy" / "frozen"
    variant.mkdir(parents=True)
    source = checkpoint(tmp_path)
    payload = torch.load(source, weights_only=True)
    payload.update(schema=workbench.ADAPTATION_SCHEMA, variant="frozen", schedule_sha256="schedule-proof", selected_files_sha256={"crop.png": "crop-proof"})
    payload.pop("encoder_size")
    payload["head_config"].pop("target")
    path = variant / "checkpoint.final.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="matching run.json"):
        workbench.checkpoint_snapshot(path)
    run = {"schema": workbench.ADAPTATION_SCHEMA, "encoder_dimensions": [280, 280], "schedule_sha256": "schedule-proof", "selected_files_sha256": {"crop.png": "crop-proof"}}
    (variant.parent / "run.json").write_text(json.dumps(run))
    normalized, _ = workbench.checkpoint_snapshot(path)
    assert normalized["encoder_size"] == 280
    assert normalized["encoder_size_provenance"]["sha256"] == digest(variant.parent / "run.json")
    packed = workbench.package(args(checkpoint=path, expected_sha256=None, output=tmp_path / "package"))
    moved, _ = workbench.checkpoint_snapshot(Path(packed["checkpoint_path"]))
    assert moved["encoder_size"] == 280 and moved["encoder_size_provenance"] == normalized["encoder_size_provenance"]
    run["schedule_sha256"] = "other-run"
    (variant.parent / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="does not match"):
        workbench.checkpoint_snapshot(path)


@pytest.mark.parametrize("field,value", (("base_channels", 400000), ("projection_channels", 400000), ("base_channels", True), ("projection_channels", "12")))
def test_unbounded_malformed_configuration_refused_before_model_allocation(tmp_path, monkeypatch, field, value):
    path = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["head_config"][field] = value
    torch.save(payload, path)
    monkeypatch.setattr(workbench, "MaterialMapHead", lambda **_values: pytest.fail("Invalid configuration must fail before constructing a head"))
    with pytest.raises(ValueError, match="bounded configuration"):
        workbench.checkpoint_snapshot(path)
