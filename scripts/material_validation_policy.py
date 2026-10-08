"""Small, repeatable diagnostic regions without withholding source materials."""
from __future__ import annotations
import hashlib
import math
from material_dataset import heldout_regions, rectangles_overlap

POLICY = "automatic-material-check-5pct-v1"
SEED = "texture-studio-material-check-20261008"

def plan(materials: dict, size: int, material_id: str | None = None) -> dict:
    if material_id is not None and material_id not in materials:
        raise ValueError("Choose a material present in this dataset")
    count = max(1, math.ceil(len(materials) * 0.05))
    ranked = sorted(materials, key=lambda identity: hashlib.sha256((SEED + identity).encode()).hexdigest())
    # Prefer sources with room for both training corners plus a third region.
    def usable(identity):
        material = materials[identity]
        region = next(r for r in heldout_regions(*material["dimensions"], size) if r["split"] == "validation")
        return not any(s["status"] in ("excluded", "rejected") and rectangles_overlap(s["crop_rectangle_top_left_xywh"], region["rectangle"])
                       for s in material.get("samples", []))
    eligible = [identity for identity in ranked if usable(identity)]
    capable = [identity for identity in eligible if len(heldout_regions(*materials[identity]["dimensions"], size)) == 3]
    if material_id and material_id not in eligible:
        raise ValueError("This material's automatic check region was excluded; reapprove its usable crops or choose another material")
    selected = [material_id] if material_id else (capable + [i for i in eligible if i not in capable])[:count]
    if not selected:
        raise ValueError("No usable automatic check region remains; review excluded crops before training")
    for identity, material in materials.items():
        width, height = material["dimensions"]
        if identity in selected:
            material["regions"] = heldout_regions(width, height, size)
        else:
            material["regions"] = [
                {"ordinal":1, "split":"train", "region":"top_left", "rectangle":[0,0,size,size]},
                {"ordinal":2, "split":"train", "region":"opposite_corner", "rectangle":[width-size,height-size,size,size]}]
    return {"policy":POLICY, "fraction":0.05, "seed":SEED,
            "material_ids":sorted(selected), "quick_fit_material_id":material_id,
            "purpose":"diagnostic only; review fresh photographs manually for generalization"}
