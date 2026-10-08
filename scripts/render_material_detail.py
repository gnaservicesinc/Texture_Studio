#!/usr/bin/env python3
"""Run inside Blender: render native detail cameras from an existing packed review.

blender --background review.blend --python render_material_detail.py -- \
  --review-directory /absolute/review --sample material_id --variant target
The existing blend, packed maps, geometry, lighting and strengths stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def names(values):
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) for value in values):
        raise ValueError("Sample and variant names must be plain material identifiers")
    return list(dict.fromkeys(values))


def plan_jobs(prepared, samples, variants, preview_size):
    if not 16 <= preview_size <= 2048:
        raise ValueError("Preview size must be between 16 and 2048")
    groups = {group["material_id"]: group for group in prepared["materials"]}
    jobs = []
    for sample in names(samples):
        if sample not in groups:
            raise ValueError(f"Unknown material: {sample}")
        group = groups[sample]
        available = {variant["name"]: variant for variant in group["variants"]}
        for variant in names(variants):
            if variant not in available:
                raise ValueError(f"Unknown variant: {sample}/{variant}")
            shape = available[variant]["height_provenance"]["source_shape"]
            source_width = shape[1]
            repeats = group["tiling_repeats"]
            # Centered source-width footprint, shared between both render modes.
            scale = group["physical_width_m"] * preview_size / source_width / repeats
            for mode in ("material", "clay"):
                scene = f"{sample}__{variant}__{mode}__grazing"
                jobs.append({"scene": scene, "camera": scene + " native detail camera",
                             "path": f"supplemental-detail/{sample}/{variant}__{mode}__grazing__native.png",
                             "material_id": sample, "variant": variant, "render_mode": mode,
                             "source_width_pixels": source_width, "ortho_scale_m": scale,
                             "frame_policy": "Centered orthographic footprint equals preview-width/source-width of one native texture tile; tilted projection changes sampling on the vertical axis"})
    return jobs


def main(argv=None):
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-directory", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--preview-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="CPU")
    args = parser.parse_args(argv)
    root = args.review_directory.resolve()
    blend = root / "material-quality-review.blend"
    if Path(bpy.data.filepath).resolve() != blend or not blend.is_file():
        raise ValueError("Open the existing review's material-quality-review.blend before running this helper")
    if not 1 <= args.samples <= 512:
        raise ValueError("Samples must be between 1 and 512")
    audit_path = root / "supplemental-detail-audit.json"
    if audit_path.exists():
        raise ValueError("Supplemental audit already exists; prior review outputs will not be overwritten")
    prepared_path = root / "prepared-manifest.json"
    prepared = json.loads(prepared_path.read_text())
    jobs = plan_jobs(prepared, args.sample, args.variant, args.preview_size)
    for job in jobs:
        if job["scene"] not in bpy.data.scenes or job["camera"] not in bpy.data.objects:
            raise ValueError(f"Packed scene or detail camera is missing: {job['scene']}")
        if (root / job["path"]).exists():
            raise ValueError(f"Detail render already exists: {job['path']}")
    original_hash = digest(blend)
    if args.device == "METAL":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        if not any(device.type == "METAL" for device in preferences.devices):
            raise ValueError("Metal was requested but no Metal device is available")
        for device in preferences.devices:
            device.use = device.type == "METAL"
    audit = {"schema": "texture-studio-supplemental-detail-v1", "source_blend": str(blend),
             "source_blend_sha256": original_hash, "prepared_manifest_sha256": digest(prepared_path),
             "blender_version": bpy.app.version_string, "device": args.device,
             "preview_size": args.preview_size, "samples": args.samples,
             "geometry_changed": False, "lighting_changed": False, "height_strength_changed": False,
             "native_map_assets_copied": False, "source_blend_modified": False, "rendered": []}
    for job in jobs:
        scene = bpy.data.scenes[job["scene"]]
        camera = bpy.data.objects[job["camera"]]
        scene.camera = camera
        camera.data.ortho_scale = job["ortho_scale_m"]
        scene.render.resolution_x = scene.render.resolution_y = args.preview_size
        scene.render.resolution_percentage = 100
        scene.cycles.samples = args.samples
        scene.cycles.device = "GPU" if args.device == "METAL" else "CPU"
        destination = root / job["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True, scene=scene.name)
        audit["rendered"].append({**job, "png_sha256": digest(destination)})
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    audit["source_blend_modified"] = digest(blend) != original_hash
    if audit["source_blend_modified"]:
        raise ValueError("Existing review blend changed during detail rendering")
    audit["complete"] = True
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print("SUPPLEMENTAL_DETAIL_COMPLETE", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(arguments))
