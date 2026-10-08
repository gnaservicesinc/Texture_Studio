"""Source-only Xcode resources stay complete after local Python inference."""
from pathlib import Path
import hashlib
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage_material_backend import stage, stage_da3


def write(root, name, content=b"source fixture\n"):
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(content)
    return file


def da3_source(root):
    names = (
        "worker.py", "setup_runtime.py", "requirements.txt", "UPSTREAM_LICENSE", "UPSTREAM_REVISION",
        "upstream/depth_anything_3/api.py", "upstream/depth_anything_3/configs/da3-giant.yaml",
        "upstream/depth_anything_3/configs/da3mono-large.yaml",
        "upstream/depth_anything_3/bench/configs/eval_bench.yml",
        "upstream/depth_anything_3/model/dinov2/layers/attention.py",
    )
    for index, name in enumerate(names):
        write(root, name, f"fixture {index}\n".encode())
    return names


def contents(root):
    return {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in root.rglob("*") if file.is_file()}


def test_da3_stage_keeps_source_pins_and_configs_and_removes_all_build_payloads(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "Resources/DA3Backend"
    expected = da3_source(source)
    for name in (
        "__pycache__/worker.cpython-314.pyc",
        "upstream/depth_anything_3/model/__pycache__/da3.cpython-314.pyc",
        "upstream/depth_anything_3/model/model.safetensors",
        "upstream/depth_anything_3/model/weights.pth",
        "upstream/depth_anything_3/.venv/lib/site-packages/unrelated.py",
        "runtime/lib/python3.14/unrelated.py", "pyvenv.cfg", "diagnostic.npy", "unknown.payload",
    ):
        write(source, name, b"not distributable\n")
        write(destination, name, b"old build payload\n")
    write(destination, "upstream/depth_anything_3/obsolete.py")
    (destination / "empty-stale-folder").mkdir()
    before = contents(source)

    stage_da3(source, destination)
    assert set(contents(destination)) == set(expected)
    assert contents(destination) == {name: before[name] for name in expected}
    assert contents(source) == before, "Staging must leave vendored files and local caches untouched"
    assert not (destination / "empty-stale-folder").exists()
    assert not list(destination.rglob("__pycache__"))
    assert not list(destination.parent.glob(".da3-source-stage-*"))
    first = contents(destination)
    stage_da3(source, destination)
    assert contents(destination) == first


def test_missing_required_source_does_not_destroy_previous_resources(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "Resources/DA3Backend"
    da3_source(source)
    (source / "upstream/depth_anything_3/configs/da3-giant.yaml").unlink()
    write(destination, "previous.py", b"previous valid build\n")
    previous = contents(destination)
    with pytest.raises(FileNotFoundError, match="da3-giant.yaml"):
        stage_da3(source, destination)
    assert contents(destination) == previous


def test_symlinked_source_metadata_is_not_bundled(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "Resources/DA3Backend"
    da3_source(source)
    license_file = source / "UPSTREAM_LICENSE"
    license_file.unlink()
    license_file.symlink_to(write(tmp_path, "outside-license", b"unexpected external file\n"))
    with pytest.raises(FileNotFoundError, match="UPSTREAM_LICENSE"):
        stage_da3(source, destination)
    assert not destination.exists()


def test_replacement_failure_restores_previous_build(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "Resources/DA3Backend"
    da3_source(source)
    write(destination, "previous.py", b"previous valid build\n")
    previous = contents(destination)
    rename = Path.rename

    def fail_replacement(path, target):
        if path.name == "DA3Backend" and path.parent.name.startswith(".da3-source-stage-"):
            raise OSError("fixture replacement failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_replacement)
    with pytest.raises(OSError, match="fixture replacement failed"):
        stage_da3(source, destination)
    assert contents(destination) == previous


def test_existing_xcode_entry_point_stages_both_source_backends(tmp_path):
    root, resources = tmp_path / "repo", tmp_path / "Resources"
    da3_names = da3_source(root / "native/TextureStudio/Resources/DA3Backend")
    write(root, "scripts/material_workbench.py", b"bridge source\n")
    write(root, "src/ipde/__init__.py", b"package source\n")
    write(root, "LICENSE", b"project license\n")
    write(resources, "DA3Backend/__pycache__/worker.pyc")
    write(resources, "MaterialBackend/obsolete.pt")

    stage(root, resources / "MaterialBackend")
    assert set(contents(resources / "DA3Backend")) == set(da3_names)
    assert set(contents(resources / "MaterialBackend")) == {"material_workbench.py", "ipde/__init__.py", "LICENSE"}


def test_current_vendored_backend_preserves_every_declared_source_file(tmp_path):
    source = ROOT / "native/TextureStudio/Resources/DA3Backend"
    destination = tmp_path / "Resources/DA3Backend"
    stage_da3(source, destination)
    for file in source.rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts and file.suffix in {".py", ".yaml", ".yml"}:
            assert (destination / file.relative_to(source)).read_bytes() == file.read_bytes()
    for name in ("UPSTREAM_LICENSE", "UPSTREAM_REVISION", "requirements.txt"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()


def test_xcode_staging_is_the_only_owner_of_da3_resources():
    project = (ROOT / "native/TextureStudio/TextureStudio.xcodeproj/project.pbxproj").read_text()
    # An ordinary folder Copy Resources phase can run after source staging and
    # overwrite the clean result with probe-generated bytecode.
    assert "/* DA3Backend in Resources */" not in project
    assert "name = DA3Backend; path = Resources/DA3Backend;" in project
    assert 'stage_material_backend.py' in project
    assert 'alwaysOutOfDate = 1;' in project
