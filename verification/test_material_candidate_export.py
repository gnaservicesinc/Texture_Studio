"""Candidate routing rejects mismatched encoder/adapter identities."""
import hashlib
import io
import json
import base64
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_material_candidate import checked_candidate, SCHEMA
from frozen_dino_height import CODE_REVISION, MODEL_SHA256
import export_material_candidate as exporter
from frozen_dino_height import ConditionedHeightNet
from material_dataset import read_png, write_png
from train_material_height import digest, find_samples


class CandidateExportTests(unittest.TestCase):
    def candidate(self):
        return {"schema": SCHEMA, "variant": "frozen", "head_config": {"base_channels": 12, "feature_channels": 768, "projection_channels": 12}, "encoder": {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION}, "head_state": {"probe": torch.zeros(1)}, "adapter_state": {}}

    def check(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            torch.save(value, path)
            return checked_candidate(path)

    def test_unknown_variant_never_silently_uses_frozen_encoder(self):
        value = self.candidate()
        value["variant"] = "unrecognized"
        with self.assertRaisesRegex(ValueError, "frozen/LoRA"):
            self.check(value)

    def test_frozen_candidate_cannot_discard_nonempty_adapters(self):
        value = self.candidate()
        value["adapter_state"] = {"layer": {"lora_A": torch.zeros(1)}}
        with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
            self.check(value)

    def test_encoder_code_and_nonfinite_weights_are_rejected(self):
        value = self.candidate()
        value["encoder"]["official_meta_code_revision"] = "other"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.check(value)
        value = self.candidate()
        value["head_state"]["probe"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite Float32"):
            self.check(value)


if __name__ == "__main__":
    unittest.main()


def export_fixture(tmp_path, cycle=False, encoder_size=518):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_material_curriculum import make_dataset
    dataset = tmp_path / "dataset"
    make_dataset(dataset)
    for path in dataset.glob("samples/*/sample.json"):
        metadata = json.loads(path.read_text())
        metadata["map_metadata"]["height"]["sample_bits"] = 16
        rough = np.arange(1024, dtype=np.uint16).reshape(32, 32, 1) + 30000
        write_png(path.parent / "roughness.png", rough)
        metadata["maps"]["roughness"] = "roughness.png"
        metadata["map_metadata"]["roughness"] = {"sample_sha256": digest(path.parent / "roughness.png"), "encoding": "linear_data", "sample_bits": 16}
        path.write_text(json.dumps(metadata))
    config = {"base_channels": 12, "feature_channels": 768, "projection_channels": 12}
    if cycle:
        from material_training_cycle import MaterialMapHead
        config["target"] = "height"
        head = MaterialMapHead(**config)
    else:
        head = ConditionedHeightNet(**config)
    payload = {"schema": exporter.CYCLE_SCHEMA if cycle else exporter.SCHEMA,
        "encoder": {"checkpoint_sha256": MODEL_SHA256, "official_meta_code_revision": CODE_REVISION},
        "head_config": config, "head_state": head.state_dict(), "step": 7, "encoder_size": encoder_size}
    payload.update({"target": "height"} if cycle else {"variant": "frozen", "adapter_state": {}})
    checkpoint = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint)
    args = SimpleNamespace(candidate=["checked=" + str(checkpoint)], dataset=dataset,
        material=["white_stucco_02"], split="validation", output=tmp_path / "export",
        device="cpu", allow_unreviewed=True, model_directory=tmp_path, code_directory=tmp_path)
    return args, checkpoint, payload


def fake_models(monkeypatch, feature_hook=None):
    def features(_encoder, source, size):
        if feature_hook is not None:
            feature_hook(size)
        return torch.zeros(1, 768, 2, 2, device=source.device)
    monkeypatch.setattr(exporter, "load_frozen_encoder", lambda *_args: (None, {}))
    monkeypatch.setattr(exporter, "extract_features", features)


def test_checkpoint_payload_and_checksum_use_the_same_single_byte_snapshot(tmp_path, monkeypatch):
    _, checkpoint, payload = export_fixture(tmp_path)
    original = checkpoint.read_bytes()
    replacement = dict(payload, step=9)
    load = torch.load
    seen = []
    def replace_after_snapshot(stream, **kwargs):
        assert isinstance(stream, io.BytesIO)
        seen.append(stream.getvalue())
        staged = checkpoint.with_suffix(".replacement")
        torch.save(replacement, staged)
        staged.replace(checkpoint)
        return load(stream, **kwargs)
    monkeypatch.setattr(exporter.torch, "load", replace_after_snapshot)
    actual, checksum = exporter.candidate_snapshot(checkpoint)
    assert actual["step"] == 7 and seen == [original]
    assert checksum == hashlib.sha256(original).hexdigest()
    assert digest(checkpoint) != checksum


def test_checkpoint_replacement_during_preflight_refuses_output(tmp_path, monkeypatch):
    args, checkpoint, payload = export_fixture(tmp_path)
    load = torch.load
    def replace_after_snapshot(stream, **kwargs):
        actual = load(stream, **kwargs)
        replacement = checkpoint.with_suffix(".replacement")
        torch.save(dict(payload, step=9), replacement)
        replacement.replace(checkpoint)
        return actual
    monkeypatch.setattr(exporter.torch, "load", replace_after_snapshot)
    with pytest.raises(ValueError, match="Selected export file changed"):
        exporter.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("changed_role", ("checkpoint", "metadata", "roughness", "input", "height"))
def test_changes_during_inference_never_publish_completed_manifest(tmp_path, monkeypatch, changed_role):
    args, checkpoint, payload = export_fixture(tmp_path)
    sample = next(s for s in find_samples(args.dataset, True) if s["metadata"]["material_id"] == "white_stucco_02" and s["metadata"]["split"] == "validation")
    def change(_size):
        if changed_role == "checkpoint":
            staged = checkpoint.with_suffix(".replacement")
            torch.save(dict(payload, step=9), staged)
            staged.replace(checkpoint)
        elif changed_role == "metadata":
            sample["metadata_path"].write_bytes(sample["metadata_path"].read_bytes() + b"\n")
        else:
            path = sample["metadata_path"].parent / sample["metadata"]["maps"][changed_role]
            path.write_bytes(path.read_bytes() + b"changed after decode")
    fake_models(monkeypatch, change)
    with pytest.raises(ValueError, match="Selected export file changed"):
        exporter.run(args)
    assert not (args.output / "review-manifest.json").exists()


def test_manifest_change_between_selection_and_snapshot_is_rejected(tmp_path, monkeypatch):
    args, _, _ = export_fixture(tmp_path)
    original = exporter.find_samples
    def change(dataset, reviewed):
        found = original(dataset, reviewed)
        sample = next(s for s in found if s["metadata"]["material_id"] == "white_stucco_02" and s["metadata"]["split"] == "validation")
        metadata = json.loads(sample["metadata_path"].read_text())
        metadata["notes"] = "changed during selection"
        sample["metadata_path"].write_text(json.dumps(metadata))
        return found
    monkeypatch.setattr(exporter, "find_samples", change)
    with pytest.raises(ValueError, match="manifest changed during selection"):
        exporter.run(args)
    assert not args.output.exists()


def test_cycle_encoder_size_native_exports_and_all_source_bindings(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    args, checkpoint, _ = export_fixture(tmp_path, cycle=True, encoder_size=280)
    originals = {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}
    seen = []
    fake_models(monkeypatch, lambda size: seen.append(size))
    report = exporter.run(args)
    assert seen == [280]
    material = report["materials"][0]
    assert material["target_original_bits"] == 16
    assert len(report["selected_files_sha256"]) == 5  # checkpoint, manifest, three maps
    assert report["source_files_verified_unchanged"] and report["checkpoint_files_verified_unchanged"]
    snapshot = args.output / material["sample_metadata_snapshot"]
    assert digest(snapshot) == material["sample_metadata_sha256"]
    target = next(item for item in material["variants"] if item["name"] == "target")
    assert target["height_sha256"] == material["source_map_sha256"]["height"]
    candidate = material["variants"][-1]
    assert candidate["contextual_encoder_size"] == 280 and candidate["native_dimensions"] == [32, 32]
    assert candidate["checkpoint_sha256"] == digest(checkpoint)
    import OpenEXR
    values = OpenEXR.File(candidate["height"], separate_channels=True).channels()["Y"].pixels
    assert values.dtype == np.float32 and values.shape == (32, 32)
    assert originals == {str(path): digest(path) for path in args.dataset.rglob("*") if path.is_file()}


def test_numeric_gamma_tags_and_prepared_normal_green_flip_are_not_reapplied(tmp_path, monkeypatch):
    args, _, _ = export_fixture(tmp_path)
    sample = next(s for s in find_samples(args.dataset, True) if s["metadata"]["material_id"] == "white_stucco_02" and s["metadata"]["split"] == "validation")
    metadata = sample["metadata"]
    original_codes, _ = read_png(sample["height_path"])
    sample["height_path"].unlink()
    write_png(sample["height_path"], original_codes, {"color_chunks": [{"type": "gAMA", "data_base64": base64.b64encode(struct.pack(">I", 45455)).decode()}]})
    metadata["map_metadata"]["height"]["sample_sha256"] = digest(sample["height_path"])
    normal = np.zeros((32, 32, 3), dtype=np.uint16)
    normal[..., 0], normal[..., 1], normal[..., 2] = 32768, 65535 - 40000, 65535
    normal_path = sample["metadata_path"].parent / "normal.png"
    write_png(normal_path, normal)
    metadata["maps"]["normal"] = "normal.png"
    metadata["map_metadata"]["normal"] = {"sample_sha256": digest(normal_path), "encoding": "linear_data", "source": {"suffix": "nor_dx"}, "transforms": [{"type": "directx_to_opengl"}]}
    sample["metadata_path"].write_text(json.dumps(metadata))
    original_normal_hash = digest(normal_path)
    fake_models(monkeypatch)
    report = exporter.run(args)
    material = report["materials"][0]
    assert material["surface_mode"] == "geometric_displacement"
    assert all("normal" not in variant for variant in material["variants"])
    assert digest(normal_path) == original_normal_hash
    np.testing.assert_array_equal(read_png(normal_path)[0], normal)
    # Independent cross-path check: exported original numeric target -> shader
    # preparation retains integer-code / 65535, ignoring its display gamma tag.
    import review_material_quality as quality
    prepared = quality.prepare_manifest(report, args.output, tmp_path / "quality", 32, 1)
    target = next(v for v in prepared["materials"][0]["variants"] if v["name"] == "target")
    values, _ = quality.load_numeric(tmp_path / "quality" / target["height"])
    np.testing.assert_array_equal(values, original_codes[..., 0].astype(np.float32) / np.float32(65535))
