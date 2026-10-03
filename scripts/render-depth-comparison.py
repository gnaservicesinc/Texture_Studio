#!/usr/bin/env python3
"""Create a viewing-only contact sheet from the verified teacher exports."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import cv2
import numpy as np
from ipde.extractor import discover_file
from ipde.formats import read_png_exact

output = root / "out/depth-teacher-comparison"
rows = []
for stem in ("IMG_1148", "IMG_1168"):
    discovery = discover_file(Path("/Users/andrewsmith/Downloads") / (stem + ".HEIC"))
    rgb = next(asset.array for asset in discovery.assets if asset.semantic_name == "display")
    panels = [("Display RGB", rgb)]
    for model, label in (("depthpro", "DepthPro: estimated meters"),
                         ("depth-anything-v2", "V2 Large: relative inverse depth"),
                         ("depth-anything-3", "DA3 Giant: relative depth")):
        preview = read_png_exact(output / f"{stem}_display_{model}_depth_preview.png")
        gray = (preview[:, :, 0] / 257).astype(np.uint8)
        panels.append((label, np.repeat(gray[:, :, None], 3, axis=2)))
    images = []
    for label, pixels in panels:
        image = cv2.resize(pixels, (640, 480), interpolation=cv2.INTER_AREA)
        panel = np.full((548, 660, 3), 235, np.uint8)
        panel[55:535, 10:650] = image
        cv2.putText(panel, stem + " - " + label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 20, 20), 1, cv2.LINE_AA)
        images.append(panel)
    rows.append(np.concatenate(images, axis=1))
canvas = np.concatenate(rows, axis=0)
cv2.imwrite(str(output / "teacher-comparison.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(output / "teacher-comparison.png")
