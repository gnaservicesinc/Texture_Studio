"""Visual A/B of baseline and project RAFT models on the same spatial grid."""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
from .array_storage import array_record
from .dataset import DatasetError
from .extractor import discover_file
from .spatial import RaftStereoOptions, run_raft_stereo


def compare_models(source: Path | str, output_dir: Path | str, *, baseline_model: Path,
                   candidate_model: Path, raft_root: Path | None = None, device: str = "auto",
                   iterations: int = 32) -> dict:
    source = Path(source).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DatasetError("Choose a new model-comparison directory")
    discovery = discover_file(source)
    calibration = discovery.spatial_photo
    if not calibration or not calibration.get("raft_stereo_ready"):
        raise DatasetError("Model comparison requires a calibrated Apple spatial stereo pair")
    left = next((asset for asset in discovery.assets if asset.semantic_name == "spatial_left"), None)
    right = next((asset for asset in discovery.assets if asset.semantic_name == "spatial_right"), None)
    display = next((asset for asset in discovery.assets if asset.semantic_name == "display"), None)
    if left is None or right is None:
        raise DatasetError("The photo lacks usable native stereo views")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".compare-", dir=destination.parent))
    preview_root = Path(tempfile.mkdtemp(prefix=".compare-previews-", dir=destination.parent))
    try:
        rgb = {**array_record(temporary, temporary / "left.npy", left.array), "source_bit_depth": left.source_bit_depth}
        other = {**array_record(temporary, temporary / "right.npy", right.array), "source_bit_depth": right.source_bit_depth}
        source_hash = discovery.source_sha256
        samples = []
        for role, model in (("baseline", baseline_model), ("candidate", candidate_model)):
            import torch
            payload = torch.load(model, map_location="cpu", weights_only=True)
            student = isinstance(payload, dict) and payload.get("schema") == "ipde-display-depth-v1"
            del payload
            if student:
                if display is None:
                    raise DatasetError("Display student comparison requires full display dimensions")
                from .display_student import predict_display_depth
                depth, details = predict_display_depth(left.array, right.array, display.array.shape[:2], model,
                    raft_root=raft_root, device=device, left_record={"source_bit_depth": left.source_bit_depth or 8},
                    right_record={"source_bit_depth": right.source_bit_depth or 8})
                units, reference = details["units"], "display"
            else:
                result = run_raft_stereo(left.array, right.array, calibration,
                    RaftStereoOptions(root=raft_root, model=model, device=device, iterations=iterations))
                depth, details, units, reference = result.depth_meters, result.details, "meters", "left"
            if depth.dtype != np.float32 or depth.shape != (display.array.shape[:2] if student else left.array.shape[:2]):
                raise DatasetError(f"The {role} model returned an invalid depth dtype or grid")
            valid = np.isfinite(depth) & (depth > 0)
            if not valid.any():
                raise DatasetError(f"The {role} model returned no positive finite depth")
            target_record = array_record(temporary, temporary / f"{role}-depth.npy", depth)
            label = {"label_kind": "stereo_model_estimate", "units": units, "reference_role": reference,
                "coordinate_reference": "display" if student else "native_left",
                "target": target_record,
                "native_target": target_record,
                "valid_mask": array_record(temporary, temporary / f"{role}-valid.npy", valid),
                "metadata": {**details, "units": units, "model_id": role + (" display student" if student else " RAFT"), "input_rgb_sha256": rgb["array_sha256"]}}
            sample = {"id": role, "source_id": source_hash[:24], "source_path": str(source), "source_sha256": source_hash,
                "source_bytes": discovery.source_size, "teacher_id": role, "rgb": rgb, "right_rgb": other,
                "raw_assets": [], "calibration": calibration, "group_id": source_hash[:24], "split": "validation",
                "requested_group": None, "burst_ids": [], "teacher": label, "training_target_choice": "teacher"}
            samples.append(sample)
            if student:
                sample["display_rgb"] = {**array_record(temporary, temporary / f"{role}-display.npy", display.array), "source_bit_depth": display.source_bit_depth}
        manifest = {"schema": "ipde-depth-dataset-v1", "generation_state": "complete", "samples": samples,
            "group_ids": [source_hash[:24]], "warnings": ["Model agreement does not establish accuracy. Inspect unseen captures and measured references."]}
        (temporary / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        if destination.exists():
            raise DatasetError("The comparison destination appeared during inference")
        from .dataset_review import compare_samples, preview_sample, _publish_new_directory
        if samples[0]["teacher"]["coordinate_reference"] != samples[1]["teacher"]["coordinate_reference"]:
            report = {"samples": [dict(preview_sample(temporary, sample["id"], "training", preview_root, max_dimension=0), teacher_id=sample["teacher"]["metadata"]["model_id"]) for sample in samples],
                "comparison": {"mode": "separate-camera-grids", "legend": "Stock RAFT uses the left stereo grid; the student uses the display grid. Each overlay uses its own photo. Coordinates and values across these grids are not directly comparable; no warp or pixel difference is applied."}}
        else:
            report = compare_samples(temporary, ["baseline", "candidate"], preview_root, max_dimension=0)
        preview_root.rename(temporary / "previews")
        _publish_new_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(preview_root, ignore_errors=True)
        raise
    def published_paths(value):
        if isinstance(value, dict):
            return {key: published_paths(child) for key, child in value.items()}
        if isinstance(value, list):
            return [published_paths(child) for child in value]
        if isinstance(value, str) and (value == str(preview_root) or value.startswith(str(preview_root) + "/")):
            return str(destination / "previews") + value[len(str(preview_root)):]
        if isinstance(value, str) and (value == str(temporary) or value.startswith(str(temporary) + "/")):
            return str(destination) + value[len(str(temporary)):]
        return value
    report = published_paths(report)
    return {**report, "comparison_path": str(destination), "source_path": str(source), "baseline_model": str(baseline_model),
            "candidate_model": str(candidate_model), "warnings": manifest["warnings"]}
