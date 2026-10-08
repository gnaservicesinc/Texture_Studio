#!/usr/bin/env python3
"""Render material variants in identical Cycles scenes for human quality review.

Run with the repository's Python environment (NumPy, OpenEXR, OpenCV, Pillow).
The manifest supplies shared diffuse/roughness and scale, plus each height map.
Source maps/checkpoints are read-only. PNG integer codes are preserved before
the explicitly declared code/max-code conversion to Float32 shader inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

SCHEMA = "texture-studio-material-quality-review-v1"
BLENDER_DEFAULT = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RATING_PROMPTS = {
    "useful_detail": "Does relief follow useful surface structure, including fine grain, without broad cloudy shapes?",
    "noise": "Are there speckles, ringing, block/grid patterns, spikes, or oversharpened edges?",
    "lighting_leakage": "Do diffuse shadows or highlights incorrectly become raised or recessed geometry? Compare clay views.",
    "tiling": "Are repeated tile boundaries visible, and is repeated structure distracting under grazing light?",
    "overall_usability": "Would you use this material in a production scene at this shared strength?",
}
RATING_LABELS = {"noise": "noise cleanliness", "lighting_leakage": "lighting cleanliness"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("Material/variant names must use letters, digits, underscores or hyphens")
    return value


def source_path(value: str, base: Path) -> Path:
    path = Path(value)
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing source map: {path}")
    return path


def load_numeric(path: Path) -> tuple[np.ndarray, dict]:
    """Decode untouched arrays. No gamma, contrast, interpolation or stretching."""
    information = {"source_path": str(path), "source_sha256": digest(path)}
    if path.suffix.lower() == ".png":
        from material_dataset import read_png
        values, metadata = read_png(path)
        information.update(sample_bits=metadata["sample_bits"], png_gamma=metadata.get("png_gamma"),
                           png_color_tags_ignored_for_numeric_data=True)
    elif path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".exr":
        import OpenEXR
        with OpenEXR.File(str(path), separate_channels=True) as image:
            channels = image.channels()
            if "Y" in channels:
                values = channels["Y"].pixels.copy()
            elif all(name in channels for name in "RGB"):
                values = np.stack([channels[name].pixels for name in "RGB"], axis=-1)
                if "A" in channels:
                    values = np.concatenate([values, channels["A"].pixels[..., None]], axis=-1)
            else:
                raise ValueError(f"EXR requires Y or RGB channels: {path}")
            information["exr_source_compression"] = str(image.header().get("compression"))
    else:
        raise ValueError(f"Supported source map formats are PNG, NPY and EXR: {path}")
    if values.dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"Unsupported source dtype: {path}: {values.dtype}")
    if values.ndim not in (2, 3) or min(values.shape[:2]) < 2:
        raise ValueError(f"Source maps must be H×W or H×W×channels arrays: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite numerical samples: {path}")
    information.update(source_dtype=str(values.dtype), source_shape=list(values.shape),
                       source_code_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
    if np.issubdtype(values.dtype, np.integer):
        maximum = np.iinfo(values.dtype).max
        values = values.astype(np.float32) / np.float32(maximum)
        information["representation_conversion"] = f"Float32(integer code) / {maximum}; no gamma and no min/max stretch"
    elif values.dtype == np.float64 and not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
        raise ValueError(f"Float64 values cannot be preserved exactly in FLOAT32 shader maps: {path}")
    else:
        values = values.astype(np.float32)
        information["representation_conversion"] = "Exact floating source values preserved in Float32"
    information.update(shader_min=float(values.min()), shader_max=float(values.max()),
                       outside_0_1_samples=int(np.count_nonzero((values < 0) | (values > 1))))
    return np.ascontiguousarray(values), information


def alpha_is_opaque(values: np.ndarray, source_dtype: str | None) -> bool:
    threshold = np.float32(65527) / np.float32(65535) if source_dtype == "uint16" else 1
    return bool(np.all(values >= threshold))


def scalar_map(values: np.ndarray, role: str, source_dtype: str | None = None) -> np.ndarray:
    if values.ndim == 2:
        return values
    if values.shape[2] == 1:
        return values[:, :, 0]
    if values.shape[2] == 2:
        if not alpha_is_opaque(values[:, :, 1], source_dtype):
            raise ValueError(f"{role} has meaningful non-opaque alpha")
        return values[:, :, 0]
    if values.shape[2] in (3, 4) and np.array_equal(values[:, :, 0], values[:, :, 1]) and np.array_equal(values[:, :, 0], values[:, :, 2]):
        if values.shape[2] == 4 and not alpha_is_opaque(values[:, :, 3], source_dtype):
            raise ValueError(f"{role} has non-opaque alpha")
        return values[:, :, 0]
    raise ValueError(f"{role} requires scalar samples or identical RGB channels; averaging is not inferred")


def rgb_map(values: np.ndarray, role: str, source_dtype: str | None = None) -> np.ndarray:
    if values.ndim != 3 or values.shape[2] not in (3, 4):
        raise ValueError(f"{role} requires RGB components")
    if values.shape[2] == 4 and not alpha_is_opaque(values[:, :, 3], source_dtype):
        raise ValueError(f"{role} has non-opaque alpha")
    return values[:, :, :3]


def write_exr(path: Path, values: np.ndarray) -> None:
    import OpenEXR
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim == 2:
        channels = {"Y": values}
    elif values.ndim == 3 and values.shape[2] == 3:
        channels = {name: np.ascontiguousarray(values[:, :, index]) for index, name in enumerate("RGB")}
    else:
        raise ValueError("Shader EXR requires scalar or RGB values")
    if path.exists():
        raise ValueError(f"Refusing to overwrite shader map: {path}")
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage,
                  "reviewEncoding": "raw_linear_numeric_FLOAT32"}, dict(channels)).write(str(path))
    with OpenEXR.File(str(path), separate_channels=True) as decoded:
        if any(decoded.channels()[name].pixels.dtype != np.float32 or not np.array_equal(decoded.channels()[name].pixels, values)
               for name, values in channels.items()):
            raise ValueError(f"Numeric shader map round trip changed samples: {path}")


def bounded_number(group: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    value = float(group.get(key, default))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be finite and between {minimum} and {maximum}")
    return value


def prepare_manifest(manifest: dict, base: Path, output: Path, preview_size: int, samples: int,
                     device: str = "CPU", view_modes: tuple[str, ...] = ("whole",)) -> dict:
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("materials"), list) or not manifest["materials"]:
        raise ValueError(f"Manifest requires schema={SCHEMA!r} and a nonempty materials list")
    if not 16 <= preview_size <= 2048 or not 1 <= samples <= 512:
        raise ValueError("Preview size must be 16–2048 and Cycles samples 1–512")
    if len(manifest["materials"]) > 64:
        raise ValueError("One quality review is limited to 64 material groups")
    if output.exists():
        raise ValueError(f"Use a new review output directory to preserve existing ratings/renders: {output}")
    output.mkdir(parents=True)
    assets = output / "assets"
    assets.mkdir()
    if not view_modes or any(mode not in ("whole", "detail") for mode in view_modes):
        raise ValueError("View modes must be whole and/or detail")
    result = {"schema": SCHEMA, "samples": samples, "preview_size": preview_size, "device": device,
              "materials": [], "sources_modified": False, "checkpoints_modified": False,
              "per_image_normalization": False, "automatic_model_promotion": False,
              "alpha_policy": "Raw scalar/RGB values are not multiplied by alpha. UInt16 alpha >=65527 is treated as opaque rounding; UInt8/float alpha must be opaque. Original source bytes are retained.",
              "lighting": {"neutral": {"position_relative_to_width": [.7, -.5, 1.8], "energy_per_width_squared": 60, "size_relative_to_width": 1.2},
                           "grazing": {"position_relative_to_width": [-1.4, -.7, .25], "energy_per_width_squared": 80, "size_relative_to_width": .35}},
              "camera": {"position_relative_to_width": [0, -.65, 1.7], "orthographic_scale_relative_to_width": 1.38},
              "display": {"view_transform": "AgX", "exposure": 0, "gamma": 1},
              "render_modes": ["material", "clay"], "rating_prompts": RATING_PROMPTS,
              "view_modes": list(view_modes), "detail_zoom": 3,
              "limitations": ["Geometry displacement uses explicitly bounded tessellation; details finer than the mesh sample spacing may be absent. Bump and normal-only modes compare shading relief.",
                              "A native map may contain more detail than the bounded preview can show; inspect the packed Blender scene closer.",
                              "PNG metadata alone does not establish numerical transfer; numeric codes are interpreted as declared linear data.",
                              "Diagnostic metrics are displayed without automatic ranking or production approval."]}
    seen_materials = set()
    for group in manifest["materials"]:
        identity = safe_name(group["material_id"])
        if identity in seen_materials:
            raise ValueError(f"Duplicate material identity: {identity}")
        seen_materials.add(identity)
        directory = assets / identity
        directory.mkdir()
        diffuse = source_path(group["diffuse"], base)
        encoding = group.get("diffuse_encoding")
        if encoding not in ("sRGB", "linear"):
            raise ValueError("Every material explicitly declares diffuse_encoding as sRGB or linear")
        # Directly copy PNG diffuse bytes so 16-bit RGB color is never truncated
        # by a display-image decoder. Other formats use checked FLOAT32 EXR.
        diffuse_values, diffuse_info = load_numeric(diffuse)
        diffuse_values = rgb_map(diffuse_values, "Diffuse", diffuse_info["source_dtype"])
        dimensions = list(diffuse_values.shape[:2])
        if diffuse.suffix.lower() == ".png":
            diffuse_target = directory / "diffuse.png"
            shutil.copyfile(diffuse, diffuse_target)
            if digest(diffuse_target) != diffuse_info["source_sha256"]:
                raise ValueError("Copied diffuse bytes do not match original")
        else:
            diffuse_target = directory / "diffuse.exr"
            write_exr(diffuse_target, diffuse_values)
        width = bounded_number(group, "physical_width_m", 1, .000001, 10000)
        comparison = group.get("comparison_target", "normal" if group.get("surface_mode") == "normal_only" else "height")
        if comparison not in ("height", "roughness", "normal"):
            raise ValueError("comparison_target must be height, roughness or normal")
        mode = group.get("surface_mode", "normal_only" if comparison == "normal" else "geometric_displacement")
        if mode not in ("height_bump", "normal_only", "geometric_displacement"):
            raise ValueError("surface_mode must be height_bump, normal_only or geometric_displacement")
        if comparison == "normal" and mode != "normal_only":
            raise ValueError("Normal comparison requires normal_only mode so equivalent height relief is not doubled")
        if comparison != "normal" and mode == "normal_only":
            raise ValueError("Height/roughness comparison requires height relief, not normal_only mode")
        material = {"material_id": identity, "diffuse": diffuse_target.relative_to(output).as_posix(),
                    "comparison_target": comparison,
                    "diffuse_encoding": encoding, "diffuse_provenance": diffuse_info,
                    "physical_width_m": width, "physical_scale_verified": bool(group.get("physical_scale_verified", False)),
                    "displacement_scale_m": bounded_number(group, "displacement_scale_m", .03 * width, 0, 1000),
                    "height_gain": bounded_number(group, "height_gain", 1, 0, 100),
                    "height_offset": bounded_number(group, "height_offset", 0, -100, 100),
                    "height_midlevel": bounded_number(group, "height_midlevel", .5, -100, 100),
                    "tiling_repeats": bounded_number(group, "tiling_repeats", 2, 1, 8),
                    "metallic": bounded_number(group, "metallic", 0, 0, 1), "surface_mode": mode,
                    "default_roughness": bounded_number(group, "default_roughness", .65, 0, 1),
                    "geometry_subdivision_level": int(bounded_number(group, "geometry_subdivision_level",
                                                                     min(11, max(10, math.ceil(math.log2(dimensions[1])))), 1, 11)),
                    "height_convention": "Larger height raises surface along +Z; shared gain/offset; no per-variant normalization",
                    "normal_convention": "OpenGL +Y, tangent RGB encoded 0..1; used only by normal_only mode",
                    "shader_relief": "height-only Bump strength=1 distance=shared scale; optional normal is not combined with equivalent height" if mode == "height_bump" else mode,
                    "variants": []}
        level = material["geometry_subdivision_level"]
        material["geometry_edges_per_width"] = 2 ** level
        material["geometry_samples_per_texture_width"] = (2 ** level) / material["tiling_repeats"]
        material["geometry_source_texels_per_mesh_edge"] = dimensions[1] * material["tiling_repeats"] / (2 ** level)
        if group.get("roughness"):
            roughness_path = source_path(group["roughness"], base)
            roughness, roughness_info = load_numeric(roughness_path)
            roughness = scalar_map(roughness, "Roughness", roughness_info["source_dtype"])
            if list(roughness.shape[:2]) != dimensions:
                raise ValueError("Shared roughness and diffuse dimensions differ; resizing is not inferred")
            roughness_target = directory / "roughness.exr"
            write_exr(roughness_target, roughness)
            material.update(roughness=roughness_target.relative_to(output).as_posix(), roughness_provenance=roughness_info)
        shared_height = None
        if comparison == "roughness":
            if not group.get("height"):
                raise ValueError("Roughness comparison requires one shared group height map")
            height_path = source_path(group["height"], base)
            height, height_info = load_numeric(height_path)
            height = scalar_map(height, "Shared height", height_info["source_dtype"])
            if list(height.shape[:2]) != dimensions:
                raise ValueError("Shared height/diffuse dimensions differ")
            shared_height_path = directory / "shared.height.exr"
            write_exr(shared_height_path, height)
            shared_height = (shared_height_path.relative_to(output).as_posix(), height_info)
        variants = group.get("variants", [])
        if not 1 <= len(variants) <= 8:
            raise ValueError("Each material requires 1–8 variants")
        seen_variants = set()
        for variant in variants:
            name = safe_name(variant["name"])
            if name in seen_variants:
                raise ValueError(f"Duplicate variant: {identity}/{name}")
            seen_variants.add(name)
            if comparison != "roughness" and variant.get("roughness"):
                raise ValueError("Per-variant roughness is allowed only for roughness comparison; height/normal review uses shared roughness")
            if comparison == "roughness":
                if variant.get("height"):
                    raise ValueError("Roughness comparison forbids per-variant height; use the one shared group height map")
                if not variant.get("roughness"):
                    raise ValueError("Roughness comparison requires a roughness map for every variant")
                height_relative, height_info = shared_height
            else:
                height_path = source_path(variant["height"], base)
                height, height_info = load_numeric(height_path)
                height = scalar_map(height, "Height", height_info["source_dtype"])
                if list(height.shape[:2]) != dimensions:
                    raise ValueError(f"Height/diffuse dimensions differ for {identity}/{name}; native maps must be aligned")
                height_target = directory / (name + ".height.exr")
                write_exr(height_target, height)
                height_relative = height_target.relative_to(output).as_posix()
            item = {"name": name, "height": height_relative,
                    "height_provenance": height_info, "diagnostic_metrics": variant.get("diagnostic_metrics", {}),
                    "diagnostic_metrics_do_not_rank": True}
            if comparison == "roughness":
                roughness, roughness_info = load_numeric(source_path(variant["roughness"], base))
                roughness = scalar_map(roughness, "Variant roughness", roughness_info["source_dtype"])
                if list(roughness.shape[:2]) != dimensions:
                    raise ValueError("Variant roughness/diffuse dimensions differ")
                roughness_target = directory / (name + ".roughness.exr")
                write_exr(roughness_target, roughness)
                item.update(roughness=roughness_target.relative_to(output).as_posix(), roughness_provenance=roughness_info)
            if variant.get("normal"):
                if variant.get("normal_convention", "OpenGL +Y") != "OpenGL +Y":
                    raise ValueError("Normals must explicitly use OpenGL +Y; silent channel flipping is not performed")
                normal, normal_info = load_numeric(source_path(variant["normal"], base))
                normal = rgb_map(normal, "Normal", normal_info["source_dtype"])
                if list(normal.shape[:2]) != dimensions:
                    raise ValueError("Normal/diffuse dimensions differ")
                normal_target = directory / (name + ".normal.exr")
                write_exr(normal_target, normal)
                item.update(normal=normal_target.relative_to(output).as_posix(), normal_provenance=normal_info,
                            normal_used=mode == "normal_only")
            elif mode == "normal_only":
                raise ValueError("normal_only comparison requires an independent OpenGL normal for every variant")
            if variant.get("checkpoint"):
                checkpoint = source_path(variant["checkpoint"], base)
                checkpoint_hash = digest(checkpoint)
                if variant.get("checkpoint_sha256") and variant["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError(f"Checkpoint bytes changed from the declared snapshot: {checkpoint}")
                item["checkpoint_provenance"] = {"path": str(checkpoint), "sha256": checkpoint_hash,
                                                 "declared_step": variant.get("checkpoint_step")}
            material["variants"].append(item)
        result["materials"].append(material)
    return result


BLENDER_SCRIPT = r'''# Generated portable Cycles material quality review; source/checkpoint files are never edited.
import bpy, json, math
from pathlib import Path
from mathutils import Vector
root = Path(__file__).resolve().parent
configuration = json.loads((root / "prepared-manifest.json").read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
if configuration["device"] == "METAL":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    metal = [device for device in preferences.devices if device.type == "METAL"]
    if not metal:
        raise RuntimeError("Explicit Metal render requested but no Metal device is available")
    for device in preferences.devices:
        device.use = device.type == "METAL"
images = {}
def texture(nodes, relative, encoding):
    key = (relative, encoding)
    if key not in images:
        image = bpy.data.images.load(str(root / relative), check_existing=False)
        image.colorspace_settings.name = "sRGB" if encoding == "sRGB" else "Non-Color"
        images[key] = image
    node = nodes.new("ShaderNodeTexImage")
    node.image = images[key]
    node.interpolation = "Linear"
    node.extension = "REPEAT"
    return node
def aim(object, target=(0, 0, 0)):
    object.rotation_euler = (Vector(target) - object.location).to_track_quat("-Z", "Y").to_euler()
render_jobs = []
for group in configuration["materials"]:
    width = group["physical_width_m"]
    for variant in group["variants"]:
        for display_mode in configuration["render_modes"]:
            for lighting, light_config in configuration["lighting"].items():
                name = group["material_id"] + "__" + variant["name"] + "__" + display_mode + "__" + lighting
                scene = bpy.data.scenes.new(name)
                scene["quality_material_id"] = group["material_id"]
                scene["quality_variant"] = variant["name"]
                scene["quality_surface_mode"] = group["surface_mode"]
                scene["quality_comparison_target"] = group.get("comparison_target", "height")
                scene["quality_height_gain"] = group["height_gain"]
                scene["quality_height_offset"] = group["height_offset"]
                scene["quality_height_midlevel"] = group["height_midlevel"]
                scene["quality_displacement_scale_m"] = group["displacement_scale_m"]
                scene["quality_geometry_edges_per_width"] = group["geometry_edges_per_width"]
                scene["quality_geometry_source_texels_per_mesh_edge"] = group["geometry_source_texels_per_mesh_edge"]
                scene.render.engine = "CYCLES"
                scene.cycles.device = "GPU" if configuration["device"] == "METAL" else "CPU"
                scene.cycles.samples = configuration["samples"]
                scene.cycles.seed = 0
                scene.cycles.use_denoising = False
                scene.render.resolution_x = scene.render.resolution_y = configuration["preview_size"]
                scene.render.resolution_percentage = 100
                scene.render.image_settings.file_format = "PNG"
                scene.render.image_settings.color_mode = "RGB"
                scene.render.image_settings.color_depth = "8"
                scene.render.film_transparent = False
                scene.view_settings.view_transform = configuration["display"]["view_transform"]
                scene.view_settings.exposure = configuration["display"]["exposure"]
                scene.view_settings.gamma = configuration["display"]["gamma"]
                world = bpy.data.worlds.new(name + " world")
                world.use_nodes = True
                world.node_tree.nodes["Background"].inputs["Color"].default_value = (.18, .18, .18, 1)
                world.node_tree.nodes["Background"].inputs["Strength"].default_value = .08
                scene.world = world
                mesh = bpy.data.meshes.new(name + " mesh")
                # UV orientation is Blender's standard +U/+V with tangent +Y.
                half = width / 2
                mesh.from_pydata([(-half,-half,0),(half,-half,0),(half,half,0),(-half,half,0)], [], [(0,1,2,3)])
                for polygon in mesh.polygons:
                    polygon.use_smooth = True
                mesh.uv_layers.new(name="UVMap")
                uv = [(0,0),(1,0),(1,1),(0,1)]
                for index, loop in enumerate(mesh.uv_layers.active.data):
                    loop.uv = uv[index]
                object = bpy.data.objects.new(name + " surface", mesh)
                scene.collection.objects.link(object)
                material = bpy.data.materials.new(name + " material")
                material.use_nodes = True
                nodes, links = material.node_tree.nodes, material.node_tree.links
                nodes.clear()
                shader = nodes.new("ShaderNodeBsdfPrincipled")
                output = nodes.new("ShaderNodeOutputMaterial")
                links.new(shader.outputs["BSDF"], output.inputs["Surface"])
                shader.inputs["Metallic"].default_value = group["metallic"] if display_mode == "material" else 0
                shader.inputs["Roughness"].default_value = group["default_roughness"] if display_mode == "material" else .65
                coordinates = nodes.new("ShaderNodeTexCoord")
                mapping = nodes.new("ShaderNodeVectorMath")
                mapping.operation = "SCALE"
                mapping.inputs[3].default_value = group["tiling_repeats"]
                links.new(coordinates.outputs["UV"], mapping.inputs[0])
                def mapped_texture(relative, encoding="linear"):
                    node = texture(nodes, relative, encoding)
                    links.new(mapping.outputs["Vector"], node.inputs["Vector"])
                    return node
                if display_mode == "material":
                    diffuse = mapped_texture(group["diffuse"], group["diffuse_encoding"])
                    links.new(diffuse.outputs["Color"], shader.inputs["Base Color"])
                else:
                    shader.inputs["Base Color"].default_value = (.38,.38,.38,1)
                roughness_path = variant.get("roughness") if group.get("comparison_target") == "roughness" else group.get("roughness")
                if roughness_path and (display_mode == "material" or group.get("comparison_target") == "roughness"):
                    roughness = mapped_texture(roughness_path)
                    links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
                height = mapped_texture(variant["height"])
                gain = nodes.new("ShaderNodeMath")
                gain.operation = "MULTIPLY_ADD"
                links.new(height.outputs["Color"], gain.inputs[0])
                gain.inputs[1].default_value = group["height_gain"]
                gain.inputs[2].default_value = group["height_offset"]
                if group["surface_mode"] == "height_bump":
                    bump = nodes.new("ShaderNodeBump")
                    bump.inputs["Strength"].default_value = 1
                    bump.inputs["Distance"].default_value = group["displacement_scale_m"]
                    bump.invert = False
                    links.new(gain.outputs[0], bump.inputs["Height"])
                    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
                elif group["surface_mode"] == "normal_only":
                    normal_image = mapped_texture(variant["normal"])
                    normal = nodes.new("ShaderNodeNormalMap")
                    normal.space = "TANGENT"
                    normal.inputs["Strength"].default_value = 1
                    links.new(normal_image.outputs["Color"], normal.inputs["Color"])
                    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
                else:
                    displacement = nodes.new("ShaderNodeDisplacement")
                    displacement.space = "OBJECT"
                    displacement.inputs["Midlevel"].default_value = group["height_midlevel"]
                    displacement.inputs["Scale"].default_value = group["displacement_scale_m"]
                    links.new(gain.outputs[0], displacement.inputs["Height"])
                    links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
                    material.displacement_method = "DISPLACEMENT"
                    subdivide = object.modifiers.new("Shared displacement tessellation", "SUBSURF")
                    subdivide.subdivision_type = "SIMPLE"
                    subdivide.levels = subdivide.render_levels = group["geometry_subdivision_level"]
                object.data.materials.append(material)
                camera_data = bpy.data.cameras.new(name + " camera")
                camera_data.type = "ORTHO"
                camera_data.ortho_scale = configuration["camera"]["orthographic_scale_relative_to_width"] * width
                camera = bpy.data.objects.new(name + " camera", camera_data)
                scene.collection.objects.link(camera)
                camera.location = Vector(configuration["camera"]["position_relative_to_width"]) * width
                aim(camera)
                scene.camera = camera
                detail_camera_data = camera_data.copy()
                detail_camera_data.name = name + " native detail camera"
                detail_camera_data.ortho_scale = camera_data.ortho_scale / configuration["detail_zoom"]
                detail_camera = camera.copy()
                detail_camera.data = detail_camera_data
                detail_camera.name = name + " native detail camera"
                scene.collection.objects.link(detail_camera)
                light_data = bpy.data.lights.new(name + " key", "AREA")
                light_data.energy = light_config["energy_per_width_squared"] * width ** 2
                light_data.shape = "DISK"
                light_data.size = light_config["size_relative_to_width"] * width
                light = bpy.data.objects.new(name + " key", light_data)
                scene.collection.objects.link(light)
                light.location = Vector(light_config["position_relative_to_width"]) * width
                aim(light)
                for view_mode in configuration["view_modes"]:
                    relative = "renders/" + group["material_id"] + "/" + variant["name"] + "__" + display_mode + "__" + lighting + "__" + view_mode + ".png"
                    (root / relative).parent.mkdir(parents=True, exist_ok=True)
                    scene.render.filepath = str(root / relative)
                    render_jobs.append((scene, relative, camera if view_mode == "whole" else detail_camera, view_mode))
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(root / "material-quality-review.blend"))
audit = {"blender_version": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(),
         "scene_count": len(render_jobs), "cycles_device": configuration["device"],
         "packed_images": [{"name": image.name, "color_space": image.colorspace_settings.name,
                            "is_float": image.is_float, "size": list(image.size)} for image in images.values()],
         "sources_modified": False, "per_image_normalization": False,
         "scene_contracts": [{"scene": scene.name, "settings": {key: scene[key] for key in scene.keys() if key.startswith("quality_")}}
                             for scene in bpy.data.scenes if "quality_material_id" in scene], "rendered": []}
for scene, relative, camera, view_mode in render_jobs:
    scene.camera = camera
    scene.render.filepath = str(root / relative)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    audit["rendered"].append({"scene": scene.name, "path": relative, "view_mode": view_mode})
    (root / "blender-render-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("MATERIAL_QUALITY_REVIEW_COMPLETE", len(render_jobs), flush=True)
'''


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def review_template(prepared: dict) -> dict:
    entries = []
    for group in prepared["materials"]:
        for variant in group["variants"]:
            entries.append({"material_id": group["material_id"], "variant": variant["name"],
                            "comparison_target": group.get("comparison_target", "height"),
                            "decision": "unreviewed", "ratings": {key: None for key in RATING_PROMPTS},
                            "notes": "", "diagnostic_metrics": variant["diagnostic_metrics"],
                            "diagnostic_metrics_do_not_rank": True,
                            "renders": {mode + "_" + light + "_" + view: "renders/" + group["material_id"] + "/" + variant["name"] + "__" + mode + "__" + light + "__" + view + ".png"
                                        for view in prepared["view_modes"] for mode in prepared["render_modes"] for light in prepared["lighting"]}})
    return {"schema": SCHEMA, "purpose": "Manual production usefulness review; no metric-based winner or automatic model promotion",
            "rating_scale": "0 poor, 5 good; noise and lighting leakage scores mean cleanliness (higher is better)",
            "rating_prompts": RATING_PROMPTS, "entries": entries}


def write_html(output: Path, review: dict) -> None:
    rows = []
    for index, entry in enumerate(review["entries"]):
        images = "".join(f'<figure><img src="{html.escape(path)}" alt="{html.escape(name)}"><figcaption>{html.escape(name.replace("_", " "))}</figcaption></figure>' for name, path in entry["renders"].items())
        scores = "".join(f'<label title="{html.escape(prompt)}">{html.escape(RATING_LABELS.get(key, key.replace("_", " ")))}<input type="number" min="0" max="5" step="1" data-entry="{index}" data-rating="{key}"></label>' for key, prompt in RATING_PROMPTS.items())
        metrics = html.escape(json.dumps(entry["diagnostic_metrics"], indent=2))
        rows.append(f'<section><h2>{html.escape(entry["material_id"])} / {html.escape(entry["variant"])}</h2><div class="images">{images}</div><div class="ratings">{scores}<label>Decision<select data-entry="{index}" data-decision><option>unreviewed</option><option>usable</option><option>needs_work</option><option>reject</option></select></label></div><textarea data-entry="{index}" placeholder="Useful detail, artifacts, lighting leakage, seams, and suggested changes"></textarea><details><summary>Diagnostic metrics (do not rank variants)</summary><pre>{metrics}</pre></details></section>')
    embedded = json.dumps(review).replace("<", "\\u003c")
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Material quality review</title><style>
body{font:16px system-ui;background:#18191c;color:#eee;margin:24px}section{border-top:1px solid #555;padding:18px 0}.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{width:256px;max-width:100%;background:#333}figcaption{font-size:13px}.ratings{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}label{display:grid;gap:6px}input,select,textarea,button{font:inherit;padding:8px;background:#292b30;color:#fff;border:1px solid #777}input{width:60px}textarea{width:95%;height:80px}button{cursor:pointer;position:sticky;top:10px}a{color:#9dccff}
</style></head><body><h1>Material quality review</h1><p>Every variant uses the same camera, lighting, height gain, scale and shared color maps within its material. Height/normal comparisons keep roughness shared; roughness comparisons keep height shared and vary only roughness. Clay views isolate relief or gloss from the original diffuse. Review useful detail, noise, lighting leakage and seams; diagnostic metrics do not choose a winner.</p><p><strong>Rating direction: 0 = poor, 5 = good.</strong> Noise and lighting leakage scores measure cleanliness: higher means fewer artifacts or less leakage.</p><p><a href="material-quality-review.blend">Open packed Blender scene</a> · <a href="contact-sheet.png">Contact sheet</a> · <a href="review.json">Saved review JSON</a></p><button id="save">Save edited review JSON</button>''' + "".join(rows) + '''<script>
const review = ''' + embedded + ''';
document.querySelectorAll('[data-rating]').forEach(el=>el.addEventListener('input',()=>{review.entries[Number(el.dataset.entry)].ratings[el.dataset.rating]=el.value===''?null:Math.max(0,Math.min(5,Number(el.value)))}));
document.querySelectorAll('[data-decision]').forEach(el=>el.addEventListener('change',()=>review.entries[Number(el.dataset.entry)].decision=el.value));
document.querySelectorAll('textarea').forEach(el=>el.addEventListener('input',()=>review.entries[Number(el.dataset.entry)].notes=el.value));
document.getElementById('save').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)+'\\n'],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
</script></body></html>'''
    (output / "index.html").write_text(page)


def contact_sheet(output: Path, review: dict, thumbnail_size: int = 256) -> None:
    # Pillow handles rendered 8-bit display previews only, never training maps.
    from PIL import Image, ImageDraw
    padding, label_height = 12, 36
    columns = max(len(entry["renders"]) for entry in review["entries"])
    sheet = Image.new("RGB", (columns * (thumbnail_size + padding) + padding,
                               len(review["entries"]) * (thumbnail_size + label_height + padding) + padding), (24, 25, 28))
    draw = ImageDraw.Draw(sheet)
    for row, entry in enumerate(review["entries"]):
        y = padding + row * (thumbnail_size + label_height + padding)
        draw.text((padding, y), entry["material_id"] + " / " + entry["variant"], fill=(240, 240, 240))
        for column, (label, relative) in enumerate(entry["renders"].items()):
            path = output / relative
            if not path.is_file():
                raise ValueError(f"Blender did not produce requested render: {path}")
            with Image.open(path) as rendered:
                display = rendered.convert("RGB")
                display.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                x = padding + column * (thumbnail_size + padding)
                sheet.paste(display, (x, y + label_height))
                draw.text((x, y + 17), label.replace("_", " "), fill=(180, 180, 180))
    sheet.save(output / "contact-sheet.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-size", type=int, default=512)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="CPU")
    parser.add_argument("--prepare-only", action="store_true", help="Write portable assets/Blender script without starting Cycles")
    parser.add_argument("--views", choices=("whole", "detail", "both"), default="whole", help="Whole material, centered detail camera, or both; packed scene always includes both cameras")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        output = args.output.resolve()
        views = ("whole", "detail") if args.views == "both" else (args.views,)
        prepared = prepare_manifest(manifest, manifest_path.parent, output, args.preview_size, args.samples, args.device, views)
        prepared["manifest_provenance"] = {"path": str(manifest_path), "sha256": digest(manifest_path)}
        prepared["implementation_sha256"] = digest(Path(__file__))
        write_json(output / "prepared-manifest.json", prepared)
        (output / "render_material_quality.py").write_text(BLENDER_SCRIPT)
        review = review_template(prepared)
        write_json(output / "review.json", review)
        write_html(output, review)
        write_json(output / "run.json", {"schema": SCHEMA, "status": "prepared", "sources_modified": False,
                                         "automatic_model_promotion": False, "prepared_manifest_sha256": digest(output / "prepared-manifest.json")})
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "output": str(output), "render_script": str(output / "render_material_quality.py")}, indent=2))
            return 0
        if not args.blender.is_file() or args.timeout_seconds < 1:
            raise ValueError("Blender executable must exist and timeout must be positive")
        command = [str(args.blender.resolve()), "--background", "--factory-startup", "--python", str(output / "render_material_quality.py")]
        with (output / "blender.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds, check=False)
        if process.returncode != 0:
            raise ValueError(f"Blender failed with exit {process.returncode}; inspect {output / 'blender.log'}")
        if not (output / "material-quality-review.blend").is_file() or not (output / "blender-render-audit.json").is_file():
            raise ValueError("Blender did not save the scene and completed render audit")
        contact_sheet(output, review, min(256, args.preview_size))
        write_json(output / "run.json", {"schema": SCHEMA, "status": "rendered", "sources_modified": False,
                                         "automatic_model_promotion": False,
                                         "prepared_manifest_sha256": digest(output / "prepared-manifest.json"),
                                         "render_audit_sha256": digest(output / "blender-render-audit.json")})
        print(json.dumps({"status": "rendered", "output": str(output), "review": str(output / "index.html"), "blend": str(output / "material-quality-review.blend")}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"review_material_quality: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
