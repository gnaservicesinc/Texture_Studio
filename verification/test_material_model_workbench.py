"""Exercise adapter training/export/recovery without downloading model weights."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import material_lora as lora
import material_model_workbench as bridge
from material_dataset import read_png, write_png
from material_pbrnxt import SOURCE_FILES, sha256


class TinyMaterial(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Conv2d(3, 4, 1)
        self.ups = nn.ModuleList([nn.Conv2d(4, count, 3, padding=1) for count in (3, 3, 1, 1)])
        self.provenance = {"architecture": "fixture"}

    def map(self, rgb, target):
        return self.ups[lora.TARGET_BRANCH[target]](self.gen(rgb))


def configuration(target="height", step=1):
    return {"schema": lora.SCHEMA, "architecture": "pbrnxt-native-v1", "target": target,
            "scope": "final-map", "base": bridge.BASE, "step": step, "training_size": 256,
            "image_padding": False, "image_resizing": False}


def fixture_model(target="height", rank=2):
    torch.manual_seed(12)
    model = TinyMaterial().eval()
    original = copy.deepcopy(model)
    parameters = lora.install(model, target, rank=rank, alpha=rank)
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    rgb = torch.rand(1, 3, 64, 64)
    model.map(rgb, target).sub(.6).square().mean().backward()
    optimizer.step()
    return model, original, rgb


def sources(root):
    for name in SOURCE_FILES:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture source or exact upstream notice")
    return root


@pytest.mark.parametrize("target", ["height", "roughness", "normal"])
def test_only_adapter_parameters_update_and_fusion_reproduces_identical_predictions(target, tmp_path):
    model, original, rgb = fixture_model(target)
    state = lora.fused_state(model)
    fused = copy.deepcopy(original)
    fused.load_state_dict(state, strict=True)
    torch.testing.assert_close(model.map(rgb, target), fused.map(rgb, target), rtol=0, atol=0)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".lora_" not in name)
    assert torch.equal(state["gen.weight"], original.state_dict()["gen.weight"])
    tensors, specs = lora.adapter_state(model)
    file = tmp_path / "adapter.safetensors"
    lora.save_tensors(file, tensors, dict(configuration(target), layers=specs))
    settings, saved = lora.read(file)
    restored = copy.deepcopy(original)
    lora.restore(restored, settings, saved)
    torch.testing.assert_close(restored.map(rgb, target), fused.map(rgb, target), rtol=0, atol=0)


def test_outer_rrdb_activation_checkpointing_preserves_exact_forward_and_gradients():
    from material_pbrnxt import native_branch_forward
    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        def forward(self, rgb):
            return rgb + self.conv(rgb).tanh() * .2
    class ShortcutBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = nn.Sequential(RRDB(), RRDB(), nn.Conv2d(3, 3, 3, padding=1))
        def forward(self, rgb):
            return rgb + self.sub(rgb)
    original = nn.Module()
    original.model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), ShortcutBlock(),
                                   nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(3, 3, 3, padding=1),
                                   nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(3, 1, 3, padding=1))
    checkpointed = copy.deepcopy(original)
    checkpointed.material_gradient_checkpointing = True
    rgb = torch.rand(1, 3, 64, 64)
    first = native_branch_forward(original, rgb)
    second = native_branch_forward(checkpointed, rgb)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.square().mean().backward()
    second.square().mean().backward()
    for left, right in zip(original.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_weighted_mixture_preserves_delta_sum_without_svd_or_cross_terms():
    first, original, rgb = fixture_model(rank=2)
    second, _, _ = fixture_model(rank=4)
    a, a_specs = lora.adapter_state(first)
    b, b_specs = lora.adapter_state(second)
    conf, tensors = lora.combine([(dict(configuration(), layers=a_specs), a, .4),
                                 (dict(configuration(), layers=b_specs), b, 1.7)])
    combined = copy.deepcopy(original)
    lora.restore(combined, conf, tensors)
    module = lora.layers(combined)["ups.3"]
    expected = lora.layers(first)["ups.3"].delta() * .4 + lora.layers(second)["ups.3"].delta() * 1.7
    torch.testing.assert_close(module.delta(), expected, rtol=1e-6, atol=1e-8)
    other = dict(configuration(), base=dict(bridge.BASE, sha256="0" * 64), layers=b_specs)
    with pytest.raises(ValueError, match="same exact base"):
        lora.combine([(dict(configuration(), layers=a_specs), a, 1), (other, b, 1)])


@pytest.mark.parametrize("developer", [False, True])
def test_export_always_saves_adapter_and_full_runs_after_base_removed(tmp_path, monkeypatch, developer):
    model, original, rgb = fixture_model()
    directory = tmp_path / "export"
    result = bridge.export_model(model, configuration(), directory, developer, sources(tmp_path / "source"))
    assert (directory / "adapter.safetensors").is_file()
    assert (directory / "model.safetensors").exists() is developer
    assert not list(directory.rglob("*.pt")) and not list(directory.rglob("*.png"))
    path, conf, tensors, checksum = bridge.snapshot(directory, result["sha256"])
    assert result["variant"] == ("full" if developer else "lora")
    assert result["supports_training_warm_start"]
    if developer:
        monkeypatch.setattr(bridge, "complete_architecture", lambda _source: copy.deepcopy(original))
        monkeypatch.setattr(bridge, "source_provenance", lambda _source: [])  # Tiny architecture/license fixture.
        monkeypatch.setattr(bridge, "load_base", lambda *_args: pytest.fail("Fused checkpoint must not need its base weights"))
        loaded = bridge.load_selected(SimpleNamespace(code_directory=directory / "source"), conf, tensors, torch.device("cpu"))
        torch.testing.assert_close(loaded.map(rgb, "height"), model.map(rgb, "height"), rtol=0, atol=0)


def test_invalid_adapter_identity_or_numeric_tensors_rejected(tmp_path):
    model, _, _ = fixture_model()
    tensors, specs = lora.adapter_state(model)
    path = tmp_path / "adapter.safetensors"
    lora.save_tensors(path, tensors, dict(configuration(), layers=specs))
    with pytest.raises(ValueError, match="changed"):
        bridge.snapshot(path, "0" * 64)
    config = dict(configuration(), target="normal", layers=specs)
    lora.save_tensors(path, tensors, config)
    with pytest.raises(ValueError, match="outside"):
        bridge.snapshot(path)


def training_memory(monkeypatch, gib=64):
    """Give tiny CPU training fixtures explicit hardware for resource admission."""
    original_sysconf = bridge.os.sysconf
    def sysconf(name):
        if name == "SC_PHYS_PAGES":
            return gib * 1024**3 // 4096
        if name == "SC_PAGE_SIZE":
            return 4096
        return original_sysconf(name)
    monkeypatch.setattr(bridge.os, "sysconf", sysconf)


@pytest.mark.parametrize("scope", ["final-map", "map-decoder"])
@pytest.mark.parametrize("memory_gib", [1, 32, 48, 1000])
def test_capabilities_keep_complete_grids_independent_of_memory_predictions(monkeypatch, scope, memory_gib):
    training_memory(monkeypatch, gib=8)
    result = bridge.capabilities(SimpleNamespace(scope=scope, memory_gib=memory_gib, cache_gib=1))
    assert result["training_sizes"] == [256, 512, 1024, 2048, 4096]
    assert 8192 in result["inference_sizes"]
    assert result["memory_admission_enabled"] is False
    assert "memory_plans" not in result and "unavailable_training_sizes" not in result


def test_training_passes_memory_predictions_to_actual_dataset_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge.os, "sysconf", lambda *_args: pytest.fail("Starting training must not inspect machine memory"))
    def validate_actual_dataset(*_args, **_kwargs):
        raise ValueError("actual dataset validation reached")
    monkeypatch.setattr(bridge, "select_pairs", validate_actual_dataset)
    monkeypatch.setattr(bridge, "load_base", lambda *_args: pytest.fail("model must not load"))
    args = bridge.parser().parse_args(["train", "--dataset", str(tmp_path / "dataset"), "--output", str(tmp_path / "run"),
                                     "--size", "256", "--memory-gib", "48", "--cache-gib", "0", "--device", "cpu"])
    with pytest.raises(ValueError, match="actual dataset validation reached"):
        bridge.train(args)
    assert not args.output.exists()


def test_review_reconstruction_preserves_exact_integer_crop_and_normal_flip(tmp_path):
    yy, xx = np.indices((512, 512))
    codes = np.stack((xx * 71, yy * 37, np.full_like(xx, 40000)), axis=-1).astype(np.uint16)
    source = tmp_path / "original.png"
    output = tmp_path / "review.png"
    write_png(source, codes)
    result = bridge.review_source(SimpleNamespace(image=source, expected_sha256=sha256(source), size=256, output=output,
                                                  map_type="normal", normal_convention="directx", source_rectangle=[128, 128, 256, 256]))
    actual, _ = read_png(output)
    expected = codes[128:384, 128:384].copy()
    expected[..., 1] = 65535 - expected[..., 1]
    assert np.array_equal(actual, expected) and result["source_sha256"] == sha256(source)


def test_review_reconstruction_uses_literal_crop_without_rescaling(tmp_path):
    yy, xx = np.indices((512, 512))
    codes = (xx + yy * 91).astype(np.uint16)[..., None]
    source, output = tmp_path / "height.png", tmp_path / "crop.png"
    write_png(source, codes)
    args = SimpleNamespace(image=source, expected_sha256=sha256(source), size=256, output=output,
                           map_type="height", normal_convention="opengl", source_rectangle=[128, 128, 256, 256])
    result = bridge.review_source(args)
    actual, _ = read_png(output)
    np.testing.assert_array_equal(actual, codes[128:384, 128:384])
    assert result["source_resize_algorithm"] == "exact_native_integer_codes"
    assert result["source_crop_rectangle"] == [128, 128, 256, 256]
    args.source_rectangle = [0, 0, 512, 512]
    args.output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="exact selected training grid"):
        bridge.review_source(args)


def test_tiled_generation_preserves_all_pixels_and_bounds_each_native_forward():
    class LocalModel:
        def map(self, rgb, _target):
            assert max(rgb.shape[-2:]) <= 640
            assert all(side % 64 == 0 for side in rgb.shape[-2:])
            return rgb[:, :1] * 2 - .2
    rgb = torch.rand(1, 3, 1099, 777)
    result, provenance = bridge.tiled_prediction(LocalModel(), rgb, "height", torch.device("cpu"))
    torch.testing.assert_close(result, rgb[:, :1] * 2 - .2, rtol=0, atol=0)
    assert provenance["tiled"] and provenance["tile_count"] == 6
    assert not provenance["source_pixels_resized"] and not provenance["source_pixels_discarded"]


def test_inference_at_training_resolution_uses_complete_model_context_even_with_smaller_generation_tile():
    calls = []
    class LocalModel:
        def map(self, rgb, _target):
            calls.append(list(rgb.shape[-2:]))
            return rgb[:, :1] + rgb.mean()
    rgb = torch.rand(1, 3, 1024, 1024)
    result, provenance = bridge.inference_prediction(LocalModel(), rgb, dict(configuration(), training_size=1024), torch.device("cpu"), 512)
    assert calls == [[1024, 1024]] and not provenance["tiled"] and provenance["matches_training_grid"]
    torch.testing.assert_close(result, rgb[:, :1] + rgb.mean(), rtol=0, atol=0)


@pytest.mark.parametrize("target", ["height", "roughness", "normal"])
def test_training_uses_entire_selected_grid_exports_both_modes_and_purges_stage(tmp_path, monkeypatch, target):
    training_memory(monkeypatch)
    original = TinyMaterial().eval()
    source = sources(tmp_path / "architecture")
    input_path = tmp_path / "diffuse.png"
    target_path = tmp_path / "target.png"
    write_png(input_path, np.full((256, 256, 3), 200, dtype=np.uint8))
    channels = 3 if target == "normal" else 1
    write_png(target_path, np.full((256, 256, channels), 32000, dtype=np.uint16))
    pair = {"metadata": {"sample_id": "fine_surface", "map_metadata": {
        "input": {"sample_bits": 8, "encoding": "srgb", "source": {"path": str(input_path), "file_sha256": sha256(input_path)}},
        target: {"sample_bits": 16, "encoding": "linear_data", "source": {"path": str(target_path), "file_sha256": sha256(target_path)}}}},
        "paths": {"input": input_path, target: target_path}, "sha256": {"input": sha256(input_path), target: sha256(target_path)},
        "dimensions": (256, 256), "target": target}
    monkeypatch.setattr(bridge, "select_pairs", lambda *_args, **kwargs: ([pair], [], {"fixture": "selected exact grid"}))
    monkeypatch.setattr(bridge, "load_base", lambda *_args: copy.deepcopy(original))
    monkeypatch.setattr(bridge, "source_directory", lambda _args: source)
    import material_native_size
    cleaned = []
    monkeypatch.setattr(material_native_size, "cleanup_prepared_dataset", lambda path: cleaned.append(path))
    args = bridge.parser().parse_args(["train", "--dataset", str(tmp_path / "stage"), "--output", str(tmp_path / "run"),
                                     "--size", "256", "--memory-gib", "48", "--cache-gib", "0", "--device", "cpu",
                                     "--target", target, "--updates-per-map", "1", "--developer-mode"])
    result = bridge.train(args)
    assert result["step"] == 1 and result["variant"] == "full" and result["status"] == "completed"
    assert cleaned == [args.dataset]
    assert result["steps"][0]["model_input_dimensions"] == [256, 256]
    assert result["steps"][0]["source_rectangle"] == [0, 0, 256, 256]
    manifest = json.loads(Path(result["review_manifest"]).read_text())
    assert manifest["materials"][0]["diffuse"] == str(input_path)
    assert manifest["materials"][0]["diffuse_native_size"] == 256
    assert manifest["materials"][0]["variants"][0][target] == str(target_path)


def test_training_cycles_real_registered_colors_and_review_records_exact_input(tmp_path, monkeypatch):
    training_memory(monkeypatch)
    from test_material_pbrnxt_data import add_sample, add_color_variant
    import material_pbrnxt_data as data
    folder, rgb, _ = add_sample(tmp_path / "dataset", dimensions=(256, 256))
    add_color_variant(folder, 65535 - rgb)
    pairs = data.select_pairs(tmp_path / "dataset", expected_size=256)
    original = TinyMaterial().eval()
    monkeypatch.setattr(bridge, "load_base", lambda *_args: copy.deepcopy(original))
    monkeypatch.setattr(bridge, "source_directory", lambda _args: sources(tmp_path / "architecture"))
    args = bridge.parser().parse_args(["train", "--dataset", str(tmp_path / "dataset"), "--output", str(tmp_path / "run"),
                                     "--size", "256", "--memory-gib", "48", "--cache-gib", "0", "--device", "cpu",
                                     "--updates-per-map", "8", "--check-count", "1", "--seed", "29"])
    result = bridge.train(args)
    assert {event["diffuse_variant_id"] for event in result["steps"]} == {"col1", "col2"}
    assert {event["diffuse_sha256"] for event in result["steps"]} == {
        descriptor["descriptor"]["sample_sha256"] for descriptor in pairs[0][0]["input_variants"]}
    journal = [json.loads(line) for line in Path(result["updates_log"]).read_text().splitlines()]
    assert journal == result["steps"]
    manifest = json.loads(Path(result["review_manifest"]).read_text())
    review_input = manifest["materials"][0]
    chosen = next(item["descriptor"] for item in pairs[0][0]["input_variants"] if item["descriptor"]["variant_id"] == review_input["diffuse_variant_id"])
    assert review_input["diffuse"] == str(folder / chosen["path"])
    assert review_input["diffuse_training_sha256"] == chosen["sample_sha256"]


def train_fixture(tmp_path, monkeypatch, updates=20, minutes=10):
    training_memory(monkeypatch)
    from test_material_pbrnxt_data import add_sample
    add_sample(tmp_path / "dataset", dimensions=(256, 256))
    original = TinyMaterial().eval()
    source = sources(tmp_path / "architecture")
    monkeypatch.setattr(bridge, "load_base", lambda *_args: copy.deepcopy(original))
    monkeypatch.setattr(bridge, "source_directory", lambda _args: source)
    import material_native_size
    cleaned = []
    monkeypatch.setattr(material_native_size, "cleanup_prepared_dataset", lambda path: cleaned.append(path))
    arguments = ["train", "--dataset", str(tmp_path / "dataset"), "--output", str(tmp_path / "run"),
                 "--size", "256", "--memory-gib", "24", "--cache-gib", "0", "--device", "cpu",
                 "--updates-per-map", str(updates), "--max-minutes", str(minutes)]
    return bridge.parser().parse_args(arguments), cleaned, arguments


def test_stop_finishes_current_update_exports_adapter_and_purges_training_stage(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch)
    comparison_labels = []
    original_comparison = bridge.comparison
    def comparison(*arguments, **keywords):
        comparison_labels.append(arguments[6])
        return original_comparison(*arguments, **keywords)
    monkeypatch.setattr(bridge, "comparison", comparison)
    original_step = torch.optim.AdamW.step
    def step_and_stop(optimizer, *arguments, **keywords):
        result = original_step(optimizer, *arguments, **keywords)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", step_and_stop)
    result = bridge.train(args)
    assert result["status"] == "stopped" and result["completed_updates"] == 1
    assert bridge.snapshot(Path(result["checkpoint_path"]))[1]["step"] == 1
    assert result["review_deferred"] and len(comparison_labels) == 1
    assert cleaned == [args.dataset]


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_stop_during_setup_aborts_without_loading_model_or_saving_adapter(tmp_path, monkeypatch, capsys, number):
    args, cleaned, arguments = train_fixture(tmp_path, monkeypatch)
    original_handlers = {signal_number: signal.getsignal(signal_number) for signal_number in (signal.SIGINT, signal.SIGTERM)}
    def stop_loading(*_args):
        signal.getsignal(number)(number, None)
        pytest.fail("Setup must not continue after Stop")
    monkeypatch.setattr(bridge, "load_base", stop_loading)
    monkeypatch.setattr(bridge, "export_model", lambda *_args: pytest.fail("Setup abort cannot export an untrained adapter"))
    assert bridge.main(arguments) == 130
    output = capsys.readouterr().out
    assert '"event": "aborted"' in output and '"event": "training_started"' not in output
    assert not list(args.output.rglob("*.safetensors"))
    assert cleaned == [args.dataset]
    assert all(signal.getsignal(signal_number) == handler for signal_number, handler in original_handlers.items())


def test_stop_aborts_active_update_without_overwriting_previously_saved_checkpoint(tmp_path, monkeypatch, capsys):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch)
    original_prediction = bridge.predicted
    original_step = torch.optim.AdamW.step
    calls = 0
    def abort_second_update(model, rgb, target):
        nonlocal calls
        calls += 1
        if calls == 3:  # Baseline, completed update, next update.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            pytest.fail("Stop must not finish the next update")
        return original_prediction(model, rgb, target)
    def save_previous_checkpoint(optimizer, *arguments, **keywords):
        result = original_step(optimizer, *arguments, **keywords)
        (args.output / "checkpoint.latest.safetensors").write_bytes(b"previously saved checkpoint bytes")
        return result
    monkeypatch.setattr(bridge, "predicted", abort_second_update)
    monkeypatch.setattr(torch.optim.AdamW, "step", save_previous_checkpoint)
    monkeypatch.setattr(bridge, "export_model", lambda *_args: pytest.fail("Stop cannot begin a checkpoint export"))
    with pytest.raises(bridge.TrainingAborted):
        bridge.train(args)
    report = json.loads((args.output / "run.json").read_text())
    assert report["status"] == "aborted" and report["completed_updates"] == 1
    assert (args.output / "checkpoint.latest.safetensors").read_bytes() == b"previously saved checkpoint bytes"
    assert not (args.output / "export").exists()
    assert cleaned == [args.dataset]
    assert capsys.readouterr().out.count('"event": "training_started"') == 1


def test_time_budget_stops_between_updates_and_keeps_completed_adapter(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch, minutes=1)
    ticks = iter([0, 61, 62])
    monkeypatch.setattr(bridge.time, "monotonic", lambda: next(ticks))
    result = bridge.train(args)
    assert result["status"] == "completed" and result["completed_updates"] == 1
    assert Path(result["adapter_path"]).is_file() and cleaned == [args.dataset]


def test_failure_after_update_preserves_recovery_adapter_and_cleans_stage(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch)
    original = bridge.predicted
    calls = 0
    def fail_second_update(model, rgb, target):
        nonlocal calls
        calls += 1
        if calls == 3:  # Baseline review, completed update, failed next update.
            raise RuntimeError("fixture update interrupted")
        return original(model, rgb, target)
    monkeypatch.setattr(bridge, "predicted", fail_second_update)
    with pytest.raises(RuntimeError, match="fixture update interrupted"):
        bridge.train(args)
    report = json.loads((args.output / "run.json").read_text())
    assert report["status"] == "failed" and report["completed_updates"] == 1
    assert bridge.snapshot(args.output / "checkpoint.latest.safetensors")[1]["step"] == 1
    assert cleaned == [args.dataset]


def test_cli_setup_failure_also_cleans_prepared_training_stage(tmp_path, monkeypatch):
    args, cleaned, arguments = train_fixture(tmp_path, monkeypatch)
    def fail(*_args):
        raise ValueError("fixture base cannot load")
    monkeypatch.setattr(bridge, "load_base", fail)
    assert bridge.main(arguments) == 1
    assert cleaned == [args.dataset]


def test_hub_upload_catalog_download_and_reuse_preserve_no_photo_package(tmp_path, monkeypatch):
    model, _, _ = fixture_model()
    exported = tmp_path / "export"
    bridge.export_model(model, configuration(), exported, True, sources(tmp_path / "architecture"))
    monkeypatch.setattr(bridge, "CATALOG", tmp_path / "catalog.json")
    calls = []
    class API:
        def whoami(self):
            return {"name": "artist"}
        def create_repo(self, *args, **kwargs):
            calls.append(("repo", kwargs))
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True)
        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            assert all(not name.endswith((".png", ".exr", ".pt")) for name in kwargs["allow_patterns"])
            return SimpleNamespace(oid="1" * 40, commit_url="https://huggingface.co/artist/detail/commit/" + "1" * 40)
        def list_models(self, **kwargs):
            return []
    monkeypatch.setattr(bridge, "hub_api", API)
    uploaded = bridge.upload(SimpleNamespace(package=exported, repo="artist/detail", public=False))
    catalog = bridge.hub_models(None)
    assert catalog["authenticated"] and catalog["models"][0]["repository"] == "artist/detail"
    assert uploaded["source_photos_uploaded"] is False and uploaded["sha256"] == sha256(exported / "model.safetensors")
    import huggingface_hub
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        shutil.copytree(exported, kwargs["local_dir"], dirs_exist_ok=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = SimpleNamespace(repo="artist/detail", revision="1" * 40, destination=tmp_path / "downloaded")
    result = bridge.download_model(args)
    again = bridge.download_model(args)
    assert not result["reused"] and again["reused"]
    assert len(downloads) == 1 and result["sha256"] == again["sha256"]
    assert Path(result["code_directory"]).is_dir()
    assert not list(tmp_path.glob(".material-hub-*"))


def test_failed_hub_download_removes_only_its_partial_stage_and_keeps_previous_model(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(bridge, "CATALOG", tmp_path / "catalog.json")
    def fail_download(**kwargs):
        Path(kwargs["local_dir"], "partial.safetensors").write_text("unfinished")
        raise RuntimeError("network interrupted")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_download)
    with pytest.raises(ValueError, match="model download failed"):
        bridge.download_model(SimpleNamespace(repo="artist/detail", revision="1" * 40, destination=tmp_path / "downloaded"))
    assert not (tmp_path / "downloaded").exists() and not list(tmp_path.glob(".material-hub-*"))


def test_changed_checkpoint_or_missing_separate_adapter_cannot_be_reexported(tmp_path):
    model, _, _ = fixture_model()
    exported = tmp_path / "export"
    result = bridge.export_model(model, configuration(), exported, True, sources(tmp_path / "architecture"))
    (exported / "adapter.safetensors").unlink()
    with pytest.raises(ValueError, match="separately saved adapter"):
        bridge.package(SimpleNamespace(checkpoint=exported / "model.safetensors", expected_sha256=result["sha256"]))


@pytest.mark.parametrize("developer", [False, True])
def test_reexport_fused_checkpoint_requires_no_original_base_weights(tmp_path, monkeypatch, developer):
    model, _, _ = fixture_model()
    exported = tmp_path / "original-export"
    result = bridge.export_model(model, configuration(), exported, True, sources(tmp_path / "architecture"))
    monkeypatch.setattr(bridge, "load_base", lambda *_args: pytest.fail("A full export must remain exportable after its base was removed"))
    monkeypatch.setattr(bridge, "source_provenance", lambda _source: [])  # Tiny architecture/license fixture.
    args = bridge.parser().parse_args(["package", "--checkpoint", str(exported / "model.safetensors"),
                                      "--expected-sha256", result["sha256"], "--output", str(tmp_path / "new-export"),
                                      *(["--developer-mode"] if developer else [])])
    reexported = bridge.package(args)
    conf, hashes = bridge.verified_package(Path(reexported["package_path"]))
    assert conf["full_checkpoint"] is developer
    assert "adapter.safetensors" in hashes
    if developer:
        _, full, tensors, _ = bridge.snapshot(Path(reexported["checkpoint_path"]))
        assert full["fused_adapter_sha256"] == hashes["adapter.safetensors"]
        assert all(torch.equal(value, lora.fused_state(model)[name]) for name, value in tensors.items())


def test_quick_checks_are_capped_but_requested_and_final_checkpoints_use_full_pool(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch, updates=3)
    from test_material_pbrnxt_data import add_sample
    for number in range(6):
        add_sample(args.dataset, f'check_{number}', f'subject_{number}', split='validation', dimensions=(256, 256))
    args.validation_every = 1
    original_step = torch.optim.AdamW.step
    calls = 0
    original_signal = signal.getsignal(signal.SIGUSR1)
    def request_while_updating(optimizer, *arguments, **keywords):
        nonlocal calls
        result = original_step(optimizer, *arguments, **keywords)
        calls += 1
        if calls == 2:
            signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', request_while_updating)
    result = bridge.train(args)
    history = result['validation_history']
    assert [(item['step'], item['scope'], item['sample_count']) for item in history] == [
        (0, 'quick', 4), (1, 'quick', 4), (2, 'full', 6), (3, 'quick', 4), (3, 'full', 6)]
    assert result['completed_updates'] == 3
    assert [checkpoint['step'] for checkpoint in result['checkpoints']] == [2, 3]
    for checkpoint in result['checkpoints']:
        configuration, _ = lora.read(Path(checkpoint['checkpoint_path']))
        assert configuration['validation']['scope'] == 'full'
        assert len({sample['sample_id'] for sample in configuration['validation']['samples']}) == 6
    assert Path(result['best_checkpoint_path']).is_file()
    assert signal.getsignal(signal.SIGUSR1) == original_signal and cleaned == [args.dataset]


def test_zero_error_learning_check_is_recorded_and_restores_model_mode():
    class Model(nn.Module):
        def map(self, rgb, target):
            return rgb[:, :1]
    model = Model().train()
    pair = {'metadata': {'sample_id': 'same'}, 'paths': {}}
    class Cache:
        def load(self, pair):
            return torch.ones(1, 3, 256, 256), torch.ones(1, 1, 256, 256)
    result = bridge.validation_check(model, [pair], Cache(), 'height', 256, 17, torch.device('cpu'))
    assert result['status'] == 'checked' and result['mae'] == 0
    assert result['samples'] == [{'sample_id': 'same', 'mae': 0}]
    assert model.training


@pytest.mark.parametrize('error', [float('nan'), float('inf')])
def test_nonfinite_learning_check_still_fails_and_restores_model_mode(error):
    class Model(nn.Module):
        def map(self, rgb, target):
            return rgb[:, :1]
    model = Model().train()
    pair = {'metadata': {'sample_id': 'invalid'}, 'paths': {}}
    class Cache:
        def load(self, pair):
            return torch.ones(1, 3, 256, 256), torch.full((1, 1, 256, 256), error)
    with pytest.raises(ValueError, match='nonfinite'):
        bridge.validation_check(model, [pair], Cache(), 'height', 256, 17, torch.device('cpu'))
    assert model.training


def test_exactly_matched_training_maps_complete_updates_and_save_checkpoints(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch, updates=2)
    from test_material_pbrnxt_data import add_sample
    add_sample(args.dataset, 'check', 'check-subject', split='validation', dimensions=(256, 256))
    original = TinyMaterial().eval()
    monkeypatch.setattr(bridge, 'load_base', lambda *_args: copy.deepcopy(original))
    source_cache = bridge.PairCache
    class MatchedCache(source_cache):
        def load(self, pair):
            rgb, _reference = super().load(pair)
            with torch.no_grad():
                reference = original.map(rgb, args.target)
            return rgb, reference
    monkeypatch.setattr(bridge, 'PairCache', MatchedCache)
    args.validation_every = 1
    result = bridge.train(args)
    assert result['status'] == 'completed' and result['completed_updates'] == 2
    assert all(event['metrics']['total'] == 0 for event in result['steps'])
    assert all(check['mae'] == 0 for check in result['validation_history'])
    assert result['final_validation']['status'] == 'checked'
    assert Path(result['checkpoint_path']).is_file()
    assert Path(result['best_checkpoint_path']).is_file()
    assert bridge.snapshot(Path(result['checkpoint_path']))[1]['step'] == 2
    assert cleaned == [args.dataset]


def test_nonfinite_training_error_still_stops_before_weight_update(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch, updates=1)
    def nonfinite_loss(prediction, reference, **_kwargs):
        return prediction.sum() * float('nan'), {'total': float('nan')}
    monkeypatch.setattr(bridge, 'height_loss', nonfinite_loss)
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *_args, **_kwargs: pytest.fail('Invalid loss cannot update weights'))
    with pytest.raises(ValueError, match='Training error is nonfinite'):
        bridge.train(args)
    report = json.loads((args.output / 'run.json').read_text())
    assert report['status'] == 'failed' and report['completed_updates'] == 0
    assert report['training_performed'] is False
    assert not list(args.output.rglob('*.safetensors'))
    assert cleaned == [args.dataset]


def test_failed_initial_check_does_not_claim_training_or_publish_checkpoint(tmp_path, monkeypatch):
    args, cleaned, _ = train_fixture(tmp_path, monkeypatch, updates=1)
    def fail_initial_check(*_args, **_kwargs):
        raise ValueError('Validation error is nonfinite')
    monkeypatch.setattr(bridge, 'validation_check', fail_initial_check)
    with pytest.raises(ValueError, match='nonfinite'):
        bridge.train(args)
    report = json.loads((args.output / 'run.json').read_text())
    assert report['status'] == 'failed' and report['completed_updates'] == 0
    assert report['training_performed'] is False
    assert not list(args.output.rglob('*.safetensors'))
    assert cleaned == [args.dataset]
