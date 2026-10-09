#!/usr/bin/env python3
"""Material-model JSON bridge: native training, safe export and Hub recovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

# Both bridge entrypoints run directly from the installed source-only bundle.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors import SafetensorError

import material_lora as lora
from material_pbrnxt import (REVISION, SOURCE_FILES, WEIGHTS_BYTES, WEIGHTS_NAME, WEIGHTS_SHA256,
                            complete_architecture, load_complete_pretrained, obtain_pretrained, obtain_source, sha256, source_provenance)
from material_pbrnxt_data import PairCache, crop_pair, height_loss, input_rgb, select_pairs, select_input_variant, write_exr
from train_material_pbrnxt import validate_training_configuration

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "texture-studio-material-workbench-v1"
CATALOG = Path.home() / "Library/Application Support/Texture Studio/material-hub-catalog.json"
BASE = {"architecture": "pbrnxt-native-v1", "revision": REVISION, "sha256": WEIGHTS_SHA256,
        "name": "PBRnxt material mapping"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        path = path / ("model.safetensors" if (path / "model.safetensors").is_file() else "adapter.safetensors")
    if path.suffix != ".safetensors":
        raise ValueError("Select a material .safetensors checkpoint or its export directory")
    return path.resolve()


def snapshot(path: Path, expected: str | None = None) -> tuple[Path, dict, dict, str]:
    path = checkpoint_path(path)
    # Both metadata and tensors belong to the same immutable in-memory bytes.
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected and checksum != expected:
        raise ValueError("Selected material checkpoint changed; reload it before use")
    configuration, tensors = lora.read_bytes(raw)
    return path, configuration, tensors, checksum


def checkpoint_information(path: Path, configuration: dict, checksum: str) -> dict:
    return {"checkpoint_path": str(path), "sha256": checksum, "schema": configuration["schema"],
            "architecture": configuration["architecture"], "target": configuration["target"],
            "step": configuration["step"], "compatible": True,
            "variant": "full" if configuration["schema"] == lora.FULL_SCHEMA else "lora",
            "supports_training_warm_start": True, "supports_studio_inference": True,
            "refinement_policy": "native_material_lora", "base": configuration["base"],
            "training_size": configuration.get("training_size"), "scope": configuration.get("scope"),
            "validation": configuration.get("validation"),
            "model_directory": str(path.parent) if (path.parent / "source").is_dir() else None,
            "code_directory": str(path.parent / "source") if (path.parent / "source").is_dir() else None}


def checkpoint_info(args) -> dict:
    path, configuration, _, checksum = snapshot(args.checkpoint, args.expected_sha256)
    return checkpoint_information(path, configuration, checksum)


def capabilities(args) -> dict:
    return {"training_sizes": [256, 512, 1024, 2048, 4096], "inference_sizes": [256, 512, 1024, 2048, 4096, 8192],
            "targets": list(lora.TARGET_BRANCH), "scope": args.scope,
            "image_size_matches_training_size": True, "hidden_encoder_resize": False,
            "memory_admission_enabled": False}


def source_directory(args, *, download: bool = False) -> Path:
    directory = args.code_directory or obtain_source(args.model_directory, download)
    source_provenance(directory)
    return directory


def load_base(args, device: torch.device, base: dict = BASE):
    if base["sha256"] == WEIGHTS_SHA256:
        source, weights = obtain_pretrained(args.model_directory, getattr(args, "download", False))
        return load_complete_pretrained(source, weights, device=device)
    base_path = getattr(args, "base_checkpoint", None)
    if base_path is None:
        # A custom base stays recoverable through the package metadata or Hub
        # catalog. Never apply an adapter to an approximately matching model.
        local = base.get("checkpoint_path")
        base_path = Path(local) if local else None
    if base_path is None or not base_path.is_file():
        recoverable = next((item for item in read_catalog() if item.get("sha256") == base["sha256"]
                            and item.get("checkpoint_filename") == "model.safetensors" and item.get("revision")), None)
        if recoverable is None and base.get("repository") and base.get("revision"):
            recoverable = {"repository": base["repository"], "revision": base["revision"]}
        if recoverable:
            destination = args.model_directory / "recovered-bases" / base["sha256"]
            if destination.exists():
                base_path = destination / "model.safetensors"
            else:
                recovered = download_model(SimpleNamespace(repo=recoverable["repository"], revision=recoverable["revision"],
                                                           destination=destination))
                base_path = Path(recovered["checkpoint_path"])
    if base_path is None or not base_path.is_file():
        raise ValueError("This LoRA needs its exact custom base; download that checkpoint from the Model Library")
    _, configuration, tensors, checksum = snapshot(base_path, base["sha256"])
    if configuration["schema"] != lora.FULL_SCHEMA:
        raise ValueError("A custom base must be a complete material checkpoint")
    model = complete_architecture(source_directory(args)).to(device).eval()
    load_full_state(model, tensors)
    model.provenance = dict(configuration["base"], custom_base_sha256=checksum)
    return model


def load_full_state(model, tensors):
    expected = model.state_dict()
    if set(expected) != set(tensors) or any(expected[name].shape != tensor.shape or expected[name].dtype != tensor.dtype
                                           for name, tensor in tensors.items()):
        raise ValueError("Complete checkpoint tensors differ from the recorded material architecture")
    model.load_state_dict(tensors, strict=True)


def load_selected(args, configuration: dict, tensors: dict, device: torch.device):
    if configuration["schema"] == lora.FULL_SCHEMA:
        model = complete_architecture(source_directory(args)).to(device).eval()
        load_full_state(model, tensors)
        model.provenance = configuration["base"]
    else:
        model = load_base(args, device, configuration["base"])
        lora.restore(model, configuration, tensors)
    return model


def export_model(model, configuration: dict, output: Path, developer: bool, source: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    try:
        tensors, specs = lora.adapter_state(model)
        expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if developer:
            expected_bytes += sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
        if shutil.disk_usage(output).free < expected_bytes + 64 * 1024 ** 2:
            raise ValueError("Free disk space is insufficient for the selected model export")
        adapter_configuration = dict(configuration, schema=lora.SCHEMA, layers=specs)
        lora.save_tensors(output / "adapter.safetensors", tensors, adapter_configuration)
        if developer:
            lora.save_tensors(output / "model.safetensors", lora.fused_state(model),
                              dict(configuration, schema=lora.FULL_SCHEMA, fused_adapter_sha256=sha256(output / "adapter.safetensors")))
        return finish_package(configuration, output, developer, source)
    except BaseException:
        shutil.rmtree(output)
        raise


def finish_package(configuration, output, developer, source):
    metadata = {**configuration, "schema": lora.FULL_SCHEMA if developer else lora.SCHEMA,
                "adapter_filename": "adapter.safetensors", "checkpoint_filename": "model.safetensors" if developer else "adapter.safetensors",
                "full_checkpoint": developer, "optimizer_included": False, "source_images_included": False}
    write_json(output / "config.json", metadata)
    project_license = ROOT / "LICENSE"
    if not project_license.is_file():
        project_license = Path(__file__).with_name("LICENSE")
    shutil.copyfile(project_license, output / "LICENSE")
    # Preserve all verified upstream architecture and notices. A complete
    # checkpoint then loads after its original base weights are deleted.
    for name in SOURCE_FILES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    (output / "README.md").write_text(
        "---\nlicense: gpl-3.0\ntags:\n- texture-studio-material\n- material-maps\n- safetensors\n---\n\n"
        "# Texture Studio material refinement\n\n"
        f"Target: {configuration['target']}. Training maps: {configuration.get('training_size')} pixels, with every source pixel presented to the model.\n\n"
        "The LoRA is always included. A developer export also contains the complete network with the same update fused into its weights. "
        "Both use the pinned native-scale material architecture and linear numeric map targets. "
        "The originating PBRnxt/SCUNet/Swin/ESRGAN notices are included under source. "
        "Visual review is required before choosing a material model.\n")
    hashes = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
    write_json(output / "SHA256SUMS.json", hashes)
    checkpoint = output / metadata["checkpoint_filename"]
    return dict(checkpoint_information(checkpoint.resolve(), dict(configuration, schema=metadata["schema"]), sha256(checkpoint)),
                package_path=str(output.resolve()), adapter_path=str((output / "adapter.safetensors").resolve()))

def configuration_for(model, args, base: dict, step: int) -> dict:
    return {"schema": lora.SCHEMA, "architecture": "pbrnxt-native-v1", "target": args.target,
            "scope": args.scope, "base": base, "step": step, "training_size": args.size,
            "input_transfer": "sRGB diffuse codes / code maximum", "target_transfer": "linear numeric source codes / code maximum",
            "image_padding": False, "image_resizing": False,
            "training_activation_checkpointing": "rrdb-block-v1",
            "trained_utc": datetime.now(timezone.utc).isoformat()}


def predicted(model, rgb, target):
    output = model.map(rgb, target)
    channels = 3 if target == "normal" else 1
    if output.shape != (rgb.shape[0], channels, *rgb.shape[-2:]) or not torch.isfinite(output).all():
        raise ValueError("Material model returned invalid numeric map dimensions or values")
    return output


def comparison(model, pairs, cache, output, target, size, label, seed, device) -> list:
    records = []
    for pair in pairs:
        sample = pair["metadata"]["sample_id"]
        if not isinstance(sample, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sample) or sample in (".", ".."):
            raise ValueError("Comparison sample needs a safe identifier")
        directory = output / sample
        directory.mkdir(parents=True, exist_ok=True)
        pair = select_input_variant(pair, random.Random(f"review/{seed}/{sample}"))
        rgb, reference, rectangle = crop_pair(*cache.load(pair), size, random.Random(seed), whole_maps=True)
        with torch.no_grad():
            result = predicted(model, rgb.to(device), target).cpu().numpy()[0]
        result = result.transpose(1, 2, 0) if result.shape[0] == 3 else result[0]
        path = directory / (label + "." + target + ".exr")
        write_exr(path, result)
        def original(role):
            details = pair["metadata"]["map_metadata"][role]
            source = details.get("original_source") or details.get("source", {})
            return str(source.get("path") or pair["paths"][role]), source.get("file_sha256") or source.get("sha256") or pair["sha256"][role]
        diffuse_path, diffuse_hash = original("input")
        reference_path, reference_hash = original(target)
        source_rectangle = pair["metadata"].get("crop_rectangle_top_left_xywh", rectangle)
        exact_crop = not pair["metadata"].get("native_size_preparation", {}).get("target_resized", False)
        resize_algorithm = "exact_native_integer_codes" if exact_crop else "area_average_native_integer_codes"
        records.append({"sample": sample, "diffuse": diffuse_path, "reference": reference_path,
                        "diffuse_source_sha256": diffuse_hash, "reference_source_sha256": reference_hash,
                        "source_resize_algorithm": resize_algorithm, "source_crop_rectangle": source_rectangle if exact_crop else None,
                        "diffuse_variant_id": pair.get("input_variant", {}).get("variant_id"),
                        "diffuse_training_sha256": pair["sha256"]["input"],
                        "reference_normal_convention": ("directx" if pair["metadata"]["map_metadata"][target].get("source", {}).get("suffix") == "nor_dx" else "opengl"),
                        "output": str(path.resolve()), "target": target, "source_rectangle": rectangle,
                        "native_dimensions": [size, size], "label": label,
                        "reference_bits": pair["metadata"]["map_metadata"][target]["sample_bits"]})
    return records


def save_review(records, report, output):
    groups = {}
    for record in records:
        group = groups.setdefault(record["sample"], {"material_id": record["sample"], "diffuse": record["diffuse"],
            "diffuse_encoding": "sRGB", "diffuse_source_sha256": record["diffuse_source_sha256"],
            "diffuse_native_size": record["native_dimensions"][0],
            "diffuse_resize_algorithm": record["source_resize_algorithm"],
            "diffuse_source_crop_rectangle": record["source_crop_rectangle"],
            "diffuse_variant_id": record["diffuse_variant_id"],
            "diffuse_training_sha256": record["diffuse_training_sha256"],
            "variants": [{"name": "target", "role": "target", record["target"]: record["reference"],
                "map_type": record["target"], "source_bits": record["reference_bits"],
                "source_sha256": record["reference_source_sha256"], "native_dimensions": record["native_dimensions"],
                "source_resize_algorithm": record["source_resize_algorithm"],
                "source_crop_rectangle": record["source_crop_rectangle"],
                "source_normal_convention": record["reference_normal_convention"]}]})
        variant = {"name": record["label"], "role": "base" if record["label"] in ("pretrained-base", "starting-base") else "checkpoint",
                   record["target"]: record["output"], "map_type": record["target"], "sample_label": record["sample"],
                   "model_name": "Texture Studio material refinement", "model_architecture": "pbrnxt-native-v1"}
        if variant["role"] == "checkpoint":
            variant.update(checkpoint=report["checkpoint_path"], checkpoint_sha256=report["sha256"], checkpoint_step=report["step"])
        group["variants"].append(variant)
    write_json(output, {"schema": "texture-studio-material-quality-review-v1", "comparison_target": report["target"],
                        "materials": list(groups.values()), "automatic_model_promotion": False})


class TrainingAborted(Exception):
    """An immediate stop must not produce a new checkpoint."""


def validation_check(model, checks, cache, target, size, seed, device, *, limit=0, step=0) -> dict:
    """Evaluate the configured extra crops; never substitute training images."""
    selected = list(checks)
    if limit and len(selected) > limit:
        selected = random.Random(f"validation/{seed}/{step}").sample(selected, limit)
    result = {"step": step, "scope": "full" if len(selected) == len(checks) else "quick",
              "sample_count": len(selected), "pool_count": len(checks), "samples": [],
              "metric": "mean_absolute_error_native_code_fraction", "validation_scope": "known_subject_diagnostic" if checks and checks[0]["metadata"].get("validation_scope") == "known_subject_diagnostic" else "held_out", "status": "unavailable"}
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for pair in selected:
                sample = pair["metadata"]["sample_id"]
                pair = select_input_variant(pair, random.Random(f"validation-input/{seed}/{sample}"))
                rgb, reference, _rectangle = crop_pair(*cache.load(pair), size, random.Random(seed), whole_maps=True)
                prediction = predicted(model, rgb.to(device), target)
                error = float((prediction - reference.to(device)).abs().mean().cpu())
                if not math.isfinite(error):
                    raise ValueError(f"Validation error for {sample} is nonfinite; inspect the model and reference data")
                result["samples"].append({"sample_id": sample, "mae": error})
        if selected:
            result.update(status="checked", mae=sum(item["mae"] for item in result["samples"]) / len(selected))
        return result
    finally:
        model.train(was_training)


def train(args) -> dict:
    selected_path = selected_config = selected_tensors = checksum = None
    if args.checkpoint:
        selected_path, selected_config, selected_tensors, checksum = snapshot(args.checkpoint, args.expected_sha256)
        if (selected_path.parent / "source").is_dir():
            args.code_directory = selected_path.parent / "source"
    args.scope = args.scope or (selected_config.get("scope") if selected_config and selected_config["schema"] == lora.SCHEMA else "final-map")
    args.target = args.target or (selected_config["target"] if selected_config else "height")
    validate_training_configuration(args.size, args.cache_gib, args.scope)
    if args.updates_per_map < 1 or not math.isfinite(args.max_minutes) or args.max_minutes <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Updates, time limit and learning rate must be positive")
    training, checks, identity = select_pairs(args.dataset, args.material, expected_size=args.size, target=args.target)
    from material_native_size import validation_settings
    validation = validation_settings(identity.get("validation_settings"))
    if args.validation_every < 1 or args.checkpoint_every < 0:
        raise ValueError("Quick-check frequency must be positive; checkpoint frequency must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new run folder")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base, step = dict(BASE), 0
    if args.checkpoint:
        if selected_config["schema"] == lora.SCHEMA:
            if selected_config["target"] != args.target or selected_config["scope"] != args.scope:
                raise ValueError("Warm-start LoRA target and scope must match this run")
            base, step = selected_config["base"], selected_config["step"]
        else:
            base = {"architecture": "pbrnxt-native-v1", "sha256": checksum, "name": "Texture Studio custom material base",
                    "checkpoint_path": str(checkpoint_path(args.checkpoint))}
            args.base_checkpoint = args.checkpoint
            recoverable = next((item for item in read_catalog() if item.get("sha256") == checksum), None)
            if recoverable:
                base.update(repository=recoverable["repository"], revision=recoverable["revision"])
    model = load_base(args, device, base)
    source = source_directory(args)
    cache = PairCache(int(args.cache_gib * 1024 ** 3))
    inspection = checks[:args.check_count] or training[:args.check_count]
    comparison_records = comparison(model, inspection, cache, args.output / "comparison", args.target, args.size,
                                    "pretrained-base" if base["sha256"] == WEIGHTS_SHA256 else "starting-base", args.seed, device)
    if selected_config is not None and selected_config["schema"] == lora.SCHEMA:
        lora.restore(model, selected_config, selected_tensors)
        parameters = [parameter for name, parameter in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))]
    else:
        parameters = lora.install(model, args.target, args.scope, args.lora_rank, args.lora_alpha)
    if hasattr(model, "ups"):
        model.ups[lora.TARGET_BRANCH[args.target]].material_gradient_checkpointing = True
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    started, stop_requested, completed = time.monotonic(), False, 0
    checkpoint_requested = False
    training_started = False
    previous_signals = {}
    def stop(_number, _frame):
        nonlocal stop_requested
        if not training_started:
            raise TrainingAborted("Training aborted during setup")
        stop_requested = True
        print(json.dumps({"event": "stopping", "message": "Finishing this update and saving the material adapter"}), flush=True)
    def abort(_number, _frame):
        raise TrainingAborted("Training aborted without saving a new checkpoint")
    def request_checkpoint(_number, _frame):
        nonlocal checkpoint_requested
        checkpoint_requested = True
        print(json.dumps({"event": "checkpoint_queued", "message": "Full validation and checkpoint queued after this update"}), flush=True)
    previous_signals[signal.SIGUSR1] = signal.signal(signal.SIGUSR1, request_checkpoint)
    previous_signals[signal.SIGINT] = signal.signal(signal.SIGINT, stop)
    previous_signals[signal.SIGTERM] = signal.signal(signal.SIGTERM, abort)
    report = {"status": "running", "dataset": identity, "training_size": args.size, "target": args.target,
              "training_performed": False, "training_input_dimensions": [args.size, args.size],
              "image_padding": False, "image_resizing": False, "steps": [], "validation_settings": validation,
              "validation_history": [], "checkpoints": []}
    def check(full=False):
        result = validation_check(model, checks, cache, args.target, args.size, args.seed, device,
                                  limit=0 if full else validation["quick_count"], step=step + completed)
        report["validation_history"].append(result)
        if len(report["validation_history"]) > 200:
            del report["validation_history"][:-200]
        with (args.output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(dict({k: v for k, v in result.items() if k != "samples"}, event="validation"), allow_nan=False), flush=True)
        return result
    def save_checkpoint():
        if report.get("last_checkpoint_step") == step + completed:
            return report["last_full_validation"]
        result = check(full=True)
        configuration = dict(configuration_for(model, args, base, step + completed), validation=result)
        values, specs = lora.adapter_state(model)
        path = args.output / f"checkpoint-step-{step + completed:08d}.safetensors"
        lora.save_tensors(path, values, dict(configuration, layers=specs))
        info = checkpoint_information(path.resolve(), configuration, sha256(path))
        report["checkpoints"].append(info)
        report.update(last_checkpoint_step=step + completed, last_full_validation=result)
        if result["status"] == "checked" and result["mae"] < report.get("best_validation_mae", float("inf")):
            report.update(best_validation_mae=result["mae"], best_checkpoint_path=str(path.resolve()))
        write_json(args.output / "run.json", report)
        print(json.dumps(dict(info, event="checkpoint_saved", validation={k: v for k, v in result.items() if k != "samples"})), flush=True)
        return result
    try:
        write_json(args.output / "run.json", report)
        report["baseline_validation"] = check()
        rng = random.Random(args.seed)
        for epoch in range(args.updates_per_map):
            if stop_requested:
                break
            order = list(range(len(training)))
            rng.shuffle(order)
            for index in order:
                pair = select_input_variant(training[index], rng)
                rgb, reference, rectangle = crop_pair(*cache.load(pair), args.size, rng, whole_maps=True)
                rgb, reference = rgb.to(device), reference.to(device)
                optimizer.zero_grad(set_to_none=True)
                if not training_started:
                    training_started = True
                    print(json.dumps({"event": "training_started", "message": "Training updates have started"}), flush=True)
                prediction = predicted(model, rgb, args.target)
                loss, metrics = height_loss(prediction, reference, margin=0)
                if not torch.isfinite(loss):
                    raise ValueError("Training error is nonfinite; stopped before updating material weights")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
                if not torch.isfinite(gradient_norm):
                    raise ValueError("Nonfinite gradient; stopped before updating material weights")
                optimizer.step()
                completed += 1
                report["training_performed"] = True
                event = {"event": "update", "step": step + completed, "epoch": epoch + 1,
                         "sample": training[index]["metadata"]["sample_id"], "source_rectangle": rectangle,
                         "diffuse_variant_id": pair.get("input_variant", {}).get("variant_id"),
                         "diffuse_path": str(pair["paths"]["input"]), "diffuse_sha256": pair["sha256"]["input"],
                         "model_input_dimensions": [args.size, args.size], "metrics": metrics}
                report["steps"].append(event)
                # Extended runs keep a bounded live summary. The compact JSONL
                # journal preserves every real input identity without repeatedly
                # rewriting an ever-growing history on each checkpoint save.
                if len(report["steps"]) > 200:
                    del report["steps"][:-200]
                    report["steps_summary_truncated"] = True
                with (args.output / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                report["updates_log"] = str((args.output / "updates.jsonl").resolve())
                print(json.dumps(event, allow_nan=False), flush=True)
                del rgb, reference, prediction, loss
                if checkpoint_requested or (args.checkpoint_every and completed % args.checkpoint_every == 0):
                    checkpoint_requested = False
                    save_checkpoint()
                elif completed % args.validation_every == 0:
                    check()
                    write_json(args.output / "run.json", report)
                if stop_requested or time.monotonic() - started >= args.max_minutes * 60:
                    break
            if stop_requested or time.monotonic() - started >= args.max_minutes * 60:
                break
        report["final_validation"] = save_checkpoint()
        configuration = dict(configuration_for(model, args, base, step + completed), validation=report["final_validation"])
        result = export_model(model, configuration, args.output / "export", args.developer_mode, source)
        (args.output / "checkpoint.latest.safetensors").unlink(missing_ok=True)
        report.update(result, status="stopped" if stop_requested else "completed", completed_updates=completed,
                      training_performed=completed > 0)
        if not stop_requested:
            comparison_records += comparison(model, inspection, cache, args.output / "comparison", args.target,
                                             args.size, "refined-material", args.seed, device)
        else:
            report["review_deferred"] = True
        review = args.output / "review-manifest.json"
        save_review(comparison_records, report, review)
        report["review_manifest"] = str(review.resolve())
        write_json(args.output / "run.json", report)
        return report
    except TrainingAborted as error:
        report.update(status="aborted", error=str(error), completed_updates=completed,
                      training_performed=completed > 0)
        write_json(args.output / "run.json", report)
        raise
    except BaseException as error:
        if completed:
            tensors, specs = lora.adapter_state(model)
            lora.save_tensors(args.output / "checkpoint.latest.safetensors", tensors,
                              dict(configuration_for(model, args, base, step + completed), layers=specs))
        report.update(status="failed", error=str(error), completed_updates=completed)
        write_json(args.output / "run.json", report)
        raise
    finally:
        for number, handler in previous_signals.items():
            signal.signal(number, handler)
        # Once comparisons are written, no copied training inputs are needed.
        # Only the preparation module's positively identified owned stage may
        # be purged; original source maps are never removed.
        from material_native_size import cleanup_prepared_dataset
        cleanup_prepared_dataset(args.dataset)


def load_diffuse(path: Path, encoding: str):
    from material_dataset import read_png
    if path.suffix.lower() != ".png":
        raise ValueError("The material model expects the prepared diffuse map as a lossless PNG")
    codes, details = read_png(path)
    if encoding == "auto":
        encoding = "linear" if details.get("png_gamma") == 1 else "srgb"
    rgb = input_rgb(codes, encoding)
    if min(rgb.shape[-2:]) < 64 or max(rgb.shape[-2:]) > 8192:
        raise ValueError("Prepared diffuse dimensions must be between 64 and 8192 pixels")
    return torch.from_numpy(rgb).unsqueeze(0), details


@torch.no_grad()
def tiled_prediction(model, rgb, target, device, tile_size=512, halo=64):
    """Bound inference memory while retaining every original output pixel."""
    from torch.nn import functional as F
    height, width = rgb.shape[-2:]
    channels = 3 if target == "normal" else 1
    result = torch.empty((1, channels, height, width), dtype=torch.float32)
    tile_count = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            bottom, right = min(height, y + tile_size), min(width, x + tile_size)
            top, left = max(0, y - halo), max(0, x - halo)
            end_y, end_x = min(height, bottom + halo), min(width, right + halo)
            tile = rgb[..., top:end_y, left:end_x].to(device)
            pad_y, pad_x = (-tile.shape[-2]) % 64, (-tile.shape[-1]) % 64
            if pad_y or pad_x:
                tile = F.pad(tile, (0, pad_x, 0, pad_y), mode="replicate")
            prediction = predicted(model, tile, target).cpu()
            result[..., y:bottom, x:right] = prediction[..., y - top:bottom - top, x - left:right - left]
            tile_count += 1
            del tile, prediction
    return result, {"tiled": True, "tile_size": tile_size, "context_halo": halo, "tile_count": tile_count,
                    "generation_boundary_padding": "replicate to architecture grid only; discarded after prediction",
                    "source_pixels_resized": False, "source_pixels_discarded": False}


def inference_prediction(model, rgb, configuration, device, tile_size=512):
    """Preserve the whole learned training context for matching-grid inference."""
    training_size = configuration.get("training_size")
    whole_limit = max(tile_size, training_size if type(training_size) is int else tile_size)
    if max(rgb.shape[-2:]) <= whole_limit and all(side % 64 == 0 for side in rgb.shape[-2:]):
        output = predicted(model, rgb.to(device), configuration["target"]).cpu()
        return output, {"tiled": False, "model_input_dimensions": list(rgb.shape[-2:][::-1]),
                        "matches_training_grid": list(rgb.shape[-2:]) == [training_size, training_size],
                        "source_pixels_resized": False, "source_pixels_discarded": False}
    output, provenance = tiled_prediction(model, rgb, configuration["target"], device, tile_size)
    return output, dict(provenance, matches_training_grid=False)


@torch.no_grad()
def infer(args) -> dict:
    if args.input_kind != "diffuse":
        raise ValueError("Prepare the photo with Texture Studio's shared diffuse-map process before material inference")
    path, configuration, tensors, checksum = snapshot(args.checkpoint, args.expected_sha256)
    image_hash = sha256(args.image)
    rgb, details = load_diffuse(args.image, args.input_encoding)
    device = torch.device(args.device)
    if (path.parent / "source").is_dir():
        args.code_directory = path.parent / "source"
    model = load_base(args, device, configuration["base"]) if args.baseline else load_selected(args, configuration, tensors, device)
    model.requires_grad_(False).eval()
    prediction, tiling = inference_prediction(model, rgb, configuration, device, args.tile_size)
    output = prediction[0].numpy()
    output = output.transpose(1, 2, 0) if output.shape[0] == 3 else output[0]
    if sha256(args.image) != image_hash:
        raise ValueError("The prepared diffuse changed during inference")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / (configuration["target"] + ".float32.exr")
    write_exr(destination, output)
    result = {"checkpoint_sha256": checksum, "checkpoint_step": 0 if args.baseline else configuration["step"],
              "target": configuration["target"], "input_kind": "diffuse", "diffuse_path": str(args.image.resolve()),
              "image_sha256": image_hash, "native_dimensions": list(rgb.shape[-2:][::-1]),
              "source_bits": details["sample_bits"], "source_bytes_modified": False, "generation": tiling,
              "outputs": {configuration["target"]: {"path": str(destination.resolve()), "sha256": sha256(destination),
                  "encoding": "linear_data", "storage": "FLOAT32", "blender_color_space": "Non-Color"}}}
    write_json(args.output / "inference.json", result)
    return result


def review_source(args) -> dict:
    from material_dataset import read_png, write_png
    if sha256(args.image) != args.expected_sha256:
        raise ValueError("Original review map changed since this training run")
    codes, metadata = read_png(args.image)
    rectangle = getattr(args, "source_rectangle", None)
    if rectangle is not None:
        if (len(rectangle) != 4 or any(type(value) is not int for value in rectangle)
                or min(rectangle[:2]) < 0 or rectangle[2:] != [args.size, args.size]
                or rectangle[0] + rectangle[2] > codes.shape[1] or rectangle[1] + rectangle[3] > codes.shape[0]):
            raise ValueError("Original review crop must be the exact selected training grid within its source")
        x, y, width, height = rectangle
        values = np.ascontiguousarray(codes[y:y + height, x:x + width])
        algorithm = "exact_native_integer_codes"
    else:
        if tuple(codes.shape[:2][::-1]) != (args.size, args.size):
            raise ValueError("Reviewing a larger original requires the recorded exact source crop rectangle")
        values = codes
        algorithm = "exact_native_integer_codes"
    if args.map_type == "normal" and args.normal_convention.lower() == "directx":
        if values.ndim != 3 or values.shape[-1] not in (3, 4):
            raise ValueError("DirectX normal reconstruction requires RGB normal codes")
        values = values.copy()
        values[..., 1] = np.iinfo(values.dtype).max - values[..., 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Choose an unused temporary reconstructed-map path")
    write_png(args.output, values, metadata)
    return {"path": str(args.output.resolve()), "sha256": sha256(args.output), "source_sha256": args.expected_sha256,
            "native_dimensions": [args.size, args.size], "source_bits": metadata["sample_bits"],
            "source_resize_algorithm": algorithm, "source_crop_rectangle": rectangle}


def package(args) -> dict:
    path, configuration, tensors, checksum = snapshot(args.checkpoint, args.expected_sha256)
    if (path.parent / "source").is_dir():
        args.code_directory = path.parent / "source"
    if configuration["schema"] == lora.FULL_SCHEMA:
        full_configuration, full_tensors = configuration, tensors
        adapter_path = path.with_name("adapter.safetensors")
        if not adapter_path.is_file() or configuration.get("fused_adapter_sha256") != sha256(adapter_path):
            raise ValueError("The full checkpoint's separately saved adapter is required for re-export")
        _, configuration, tensors, _ = snapshot(adapter_path)
        if not args.adapter and args.checkpoint_weight == 1:
            # Re-export the verified fused model and its exact saved adapter.
            # Its original base is unnecessary, including after base cleanup.
            args.output.mkdir(parents=True, exist_ok=False)
            try:
                lora.save_tensors(args.output / "adapter.safetensors", tensors, configuration)
                if args.developer_mode:
                    lora.save_tensors(args.output / "model.safetensors", full_tensors,
                                      dict(full_configuration, fused_adapter_sha256=sha256(args.output / "adapter.safetensors")))
                result = finish_package(full_configuration if args.developer_mode else configuration, args.output,
                                        args.developer_mode, source_directory(args))
                return dict(result, source_checkpoint_sha256=checksum)
            except BaseException:
                shutil.rmtree(args.output)
                raise
    if args.adapter or args.checkpoint_weight != 1:
        adapters = [(configuration, tensors, args.checkpoint_weight)]
        for entry in args.adapter:
            name, separator, weight = entry.rpartition("=")
            if not separator:
                raise ValueError("Weighted adapter requires PATH=WEIGHT")
            _, extra_config, extra_tensors, _ = snapshot(Path(name))
            adapters.append((extra_config, extra_tensors, float(weight)))
        configuration, tensors = lora.combine(adapters)
    model = load_base(args, torch.device("cpu"), configuration["base"])
    lora.restore(model, configuration, tensors)
    source = source_directory(args)
    result = export_model(model, configuration, args.output, args.developer_mode, source)
    return dict(result, source_checkpoint_sha256=checksum)


def install_base(args) -> dict:
    source, weights = obtain_pretrained(args.destination, True)
    return {"directory": str(args.destination.resolve()), "model_directory": str(args.destination.resolve()),
            "code_directory": str(source.resolve()), "weights_sha256": sha256(weights), "weights_bytes": weights.stat().st_size}


def remove_base(args) -> dict:
    weights = args.directory / WEIGHTS_NAME
    if weights.is_symlink() or not weights.is_file() or weights.stat().st_size != WEIGHTS_BYTES or sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Remove only the verified material base weights from this model directory")
    weights.unlink()
    return {"directory": str(args.directory.resolve()), "removed": True, "bytes_reclaimed": WEIGHTS_BYTES,
            "architecture_and_licenses_retained": True, "can_redownload": True}


def hub_api():
    from huggingface_hub import HfApi
    return HfApi()


def hub_account(_args) -> dict:
    try:
        identity = hub_api().whoami()
        return {"authenticated": True, "username": identity["name"], "message": "Signed in as " + identity["name"]}
    except Exception:
        return {"authenticated": False, "username": None, "message": "Sign in using hf auth login, then refresh the account"}


def read_catalog() -> list:
    return json.loads(CATALOG.read_text()).get("models", []) if CATALOG.is_file() else []


def register_model(record: dict) -> None:
    models = {item["repository"]: item for item in read_catalog()}
    models[record["repository"]] = record
    write_json(CATALOG, {"models": list(models.values())})


def hub_models(_args) -> dict:
    account = hub_account(_args)
    models = {item["repository"]: item for item in read_catalog()}
    if account["authenticated"]:
        api = hub_api()
        try:
            for model in api.list_models(author=account["username"], filter="texture-studio-material", full=True, limit=100):
                files = {item.rfilename for item in (model.siblings or [])}
                checkpoint = "model.safetensors" if "model.safetensors" in files else "adapter.safetensors" if "adapter.safetensors" in files else None
                if checkpoint:
                    models[model.id] = dict(models.get(model.id, {}), repository=model.id, revision=model.sha,
                                           checkpoint_filename=checkpoint)
        except Exception:
            return {"models": list(models.values()), "authenticated": True, "username": account["username"],
                    "message": "Hub model discovery is unavailable; your saved downloadable models remain listed"}
    return {"models": list(models.values()), "authenticated": account["authenticated"], "username": account["username"]}


def verified_package(path: Path) -> tuple[dict, dict]:
    hashes = json.loads((path / "SHA256SUMS.json").read_text())
    allowed_top = {"adapter.safetensors", "model.safetensors", "config.json", "README.md", "LICENSE"}
    if (not isinstance(hashes, dict) or not {"adapter.safetensors", "config.json", "README.md", "LICENSE"}.issubset(hashes)
            or any(name not in allowed_top and name not in {"source/" + file for file in SOURCE_FILES} for name in hashes)):
        raise ValueError("Upload a complete material model export without datasets or source images")
    for name, checksum in hashes.items():
        file = path / name
        if file.is_symlink() or not file.is_file() or sha256(file) != checksum:
            raise ValueError("Export files changed; re-export the material model")
    configuration = json.loads((path / "config.json").read_text())
    snapshot(path / configuration["checkpoint_filename"])
    return configuration, hashes


def upload(args) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo):
        raise ValueError("Choose a Hugging Face repository as owner/model-name")
    configuration, hashes = verified_package(args.package)
    api = hub_api()
    try:
        api.whoami()
    except Exception:
        raise ValueError("Hugging Face authentication is unavailable; sign in using hf auth login") from None
    try:
        api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
        info = api.repo_info(args.repo, repo_type="model")
        if bool(info.private) != (not args.public):
            raise ValueError("Existing Hub visibility differs from the selected upload visibility")
        commit = api.upload_folder(repo_id=args.repo, folder_path=str(args.package),
                                  allow_patterns=sorted(set(hashes) | {"SHA256SUMS.json"}),
                                  commit_message="Upload Texture Studio material checkpoint and LoRA")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Hugging Face upload failed ({type(error).__name__}); verify repository access and retry") from None
    record = {"repository": args.repo, "revision": commit.oid, "target": configuration["target"],
              "checkpoint_filename": configuration["checkpoint_filename"],
              "sha256": hashes[configuration["checkpoint_filename"]], "url": f"https://huggingface.co/{args.repo}"}
    register_model(record)
    return dict(record, commit_url=commit.commit_url, private=not args.public, source_photos_uploaded=False)


def upload_selected(args) -> dict:
    exported = package(args)
    uploaded = upload(SimpleNamespace(package=args.output, repo=args.repo, public=args.public))
    return dict(uploaded, package_path=exported["package_path"], source_checkpoint_sha256=exported["source_checkpoint_sha256"])


def download_model(args) -> dict:
    from huggingface_hub import snapshot_download
    marker = ".texture-studio-hub-model.json"
    ownership = None
    if args.destination.is_symlink():
        raise ValueError("Model downloads require a real directory")
    if args.destination.exists():
        try:
            ownership = json.loads((args.destination / marker).read_text())
            configuration, _ = verified_package(args.destination)
        except (OSError, ValueError, KeyError):
            raise ValueError("The existing model directory is not a verified owned Hub download; choose another destination") from None
        if ownership.get("schema") != SCHEMA or ownership.get("directory") != str(args.destination.resolve()):
            raise ValueError("Existing model ownership differs from this download destination")
        if ownership.get("repository") == args.repo and ownership.get("revision") == args.revision:
            path, actual, _, checksum = snapshot(args.destination / configuration["checkpoint_filename"], ownership["sha256"])
            return dict(checkpoint_information(path, actual, checksum), package_path=str(args.destination.resolve()), reused=True)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".material-hub-download-", dir=args.destination.parent))
    previous = None
    try:
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(stage),
                          allow_patterns=["*.safetensors", "config.json", "README.md", "LICENSE", "SHA256SUMS.json", "source/*"],
                          max_workers=2)
        configuration, _ = verified_package(stage)
        _, actual, _, checksum = snapshot(stage / configuration["checkpoint_filename"])
        write_json(stage / marker, {"schema": SCHEMA, "directory": str(args.destination.resolve()),
                                   "repository": args.repo, "revision": args.revision, "sha256": checksum})
        shutil.rmtree(stage / ".cache", ignore_errors=True)
        if args.destination.exists():
            previous = args.destination.with_name(".material-hub-previous-" + next(tempfile._get_candidate_names()))
            args.destination.rename(previous)
        stage.rename(args.destination)
        path = args.destination / configuration["checkpoint_filename"]
        if previous:
            shutil.rmtree(previous)
            previous = None
        register_model({"repository": args.repo, "revision": args.revision, "target": actual["target"],
                        "checkpoint_filename": configuration["checkpoint_filename"], "sha256": checksum})
        return dict(checkpoint_information(path, actual, checksum), package_path=str(args.destination.resolve()),
                    model_directory=str(args.destination.resolve()), code_directory=str((args.destination / "source").resolve()), reused=False)
    except BaseException as error:
        if previous and previous.exists() and not args.destination.exists():
            previous.rename(args.destination)
        if isinstance(error, (OSError, ValueError, KeyboardInterrupt, SystemExit)):
            raise
        raise ValueError(f"Hugging Face model download failed ({type(error).__name__}); retry the saved model download") from None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    subcommands = cli.add_subparsers(dest="command", required=True)
    for command in ("capabilities", "checkpoint", "train", "refine", "infer", "package", "install-base", "remove-base",
                    "hub-account", "hub-models", "upload", "upload-selected", "download-model", "review-source"):
        p = subcommands.add_parser(command)
        if command in ("train", "refine", "infer", "package", "upload-selected"):
            p.add_argument("--model-directory", type=Path, default=ROOT / "out/material-training/pbrnxt-base")
            p.add_argument("--code-directory", type=Path)
            p.add_argument("--base-checkpoint", type=Path)
        if command in ("checkpoint", "refine", "infer", "package", "upload-selected"):
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--expected-sha256")
        if command == "train":
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--expected-sha256")
        if command in ("train", "refine", "infer", "package", "upload-selected"):
            p.add_argument("--output", type=Path, required=True)
        if command in ("train", "refine", "package", "upload-selected"):
            p.add_argument("--developer-mode", action="store_true")
        if command in ("capabilities", "train", "refine"):
            p.add_argument("--memory-gib", type=float, default=None, help=argparse.SUPPRESS)
            p.add_argument("--cache-gib", type=float, default=1)
            p.add_argument("--scope", choices=("final-map", "map-decoder"), default="final-map" if command == "capabilities" else None)
        if command in ("train", "refine"):
            p.add_argument("--dataset", type=Path, required=True)
            p.add_argument("--target", choices=tuple(lora.TARGET_BRANCH))
            p.add_argument("--size", type=int, default=1024)
            p.add_argument("--whole-maps", action="store_true")
            p.add_argument("--material", action="append")
            p.add_argument("--updates-per-map", type=int, default=100)
            p.add_argument("--max-minutes", type=float, default=30)
            p.add_argument("--learning-rate", type=float, default=1e-5)
            p.add_argument("--lora-rank", type=int, choices=(1, 2, 4, 8, 16, 32), default=8)
            p.add_argument("--lora-alpha", type=float, default=8)
            p.add_argument("--check-count", type=int, default=3)
            p.add_argument("--validation-every", type=int, default=20, help="Updates between quick validation checks")
            p.add_argument("--checkpoint-every", type=int, default=0, help="Updates between full validation and saved checkpoints; 0 saves on request and at the end")
            p.add_argument("--seed", type=int, default=17)
            p.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
            p.add_argument("--download", action="store_true")
        if command == "infer":
            p.add_argument("--image", type=Path, required=True)
            p.add_argument("--input-kind", choices=("diffuse", "photo"), default="diffuse")
            p.add_argument("--input-encoding", choices=("auto", "srgb", "linear"), default="auto")
            p.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
            p.add_argument("--baseline", action="store_true")
            p.add_argument("--tile-size", type=int, choices=(256, 512, 1024), default=512)
            p.add_argument("--memory-gib", type=float, default=None, help=argparse.SUPPRESS)
        if command == "review-source":
            p.add_argument("--image", type=Path, required=True)
            p.add_argument("--expected-sha256", required=True)
            p.add_argument("--size", type=int, choices=(256, 512, 1024, 2048), required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--source-rectangle", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
            p.add_argument("--map-type", choices=("input", "height", "roughness", "normal"), default="input")
            p.add_argument("--normal-convention", choices=("opengl", "directx", "OpenGL", "DirectX"), default="opengl")
        if command in ("package", "upload-selected"):
            p.add_argument("--adapter", action="append", default=[], help="PATH=WEIGHT; exact same base and material target")
            p.add_argument("--checkpoint-weight", type=float, default=1, help="Weight of the selected checkpoint's adapter in the mixture")
        if command in ("install-base", "download-model"):
            p.add_argument("--destination", type=Path, required=True)
        if command == "remove-base":
            p.add_argument("--directory", type=Path, required=True)
        if command in ("upload", "upload-selected", "download-model"):
            p.add_argument("--repo", required=True)
        if command in ("upload", "upload-selected"):
            p.add_argument("--public", action="store_true")
        if command == "upload":
            p.add_argument("--package", type=Path, required=True)
        if command == "download-model":
            p.add_argument("--revision", required=True, help="Exact Hub revision from the model catalog")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    handlers = {"capabilities": capabilities, "checkpoint": checkpoint_info, "train": train, "refine": train,
                "infer": infer, "package": package, "install-base": install_base, "remove-base": remove_base,
                "hub-account": hub_account, "hub-models": hub_models, "upload": upload,
                "upload-selected": upload_selected, "download-model": download_model, "review-source": review_source}
    early_handlers = {}
    if args.command in ("train", "refine"):
        def stop_during_setup(_number, _frame):
            raise TrainingAborted("Training aborted during setup")
        for number in (signal.SIGINT, signal.SIGTERM):
            early_handlers[number] = signal.signal(number, stop_during_setup)
    try:
        result = handlers[args.command](args)
        print(json.dumps(dict(result, protocol_schema=SCHEMA, command=args.command, ok=True), allow_nan=False), flush=True)
        return 0
    except TrainingAborted as error:
        print(json.dumps({"protocol_schema": SCHEMA, "command": args.command, "event": "aborted", "ok": False,
                          "error": {"code": "TrainingAborted", "message": str(error)}}), flush=True)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, SafetensorError) as error:
        print(json.dumps({"protocol_schema": SCHEMA, "command": args.command, "ok": False,
                          "error": {"code": type(error).__name__, "message": str(error)}}), flush=True)
        return 1
    finally:
        for number, handler in early_handlers.items():
            signal.signal(number, handler)
        if args.command in ("train", "refine"):
            from material_native_size import cleanup_prepared_dataset
            try:
                cleanup_prepared_dataset(args.dataset)
            except (OSError, ValueError) as error:
                print(json.dumps({"event": "cleanup_error", "message": str(error)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
