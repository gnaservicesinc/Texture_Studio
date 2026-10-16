#!/usr/bin/env python3
"""Verify SIDD review bytes and prepare a diagnostic split by physical scene.

This does not approve data, train, or modify any source review or Factory project.
All variants of a scene must stay together, even across separate sample receipts.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from sample_sidd import REPO, REVISION, sha


def contained_file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Review file is missing or outside its source review: " + str(path))
    return path


def verify_pair(root, row):
    group = "sidd-scene-" + row["capture"]["scene"]
    if row["group"] != group or row["id"] != row["capture"]["instance"] + "-" + row["frame"]:
        raise ValueError("Capture identity or physical-scene group disagrees with metadata")
    crop = row["crop"]
    if crop["resampled"] or crop["width"] != 1024 or crop["height"] != 1024:
        raise ValueError("Only recorded native 1024 crops belong in this diagnostic")
    verified = {}
    crop_parent = None
    for role in ["input", "reference"]:
        record = row["files"][role]
        path = contained_file(root, record["file"])
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Original size/SHA256 mismatch: " + str(path))
        if crop_parent is not None and path.parent != crop_parent:
            raise ValueError("Pair originals must share their recorded capture directory")
        crop_parent = path.parent
        derived = contained_file(root, str((path.parent / "native-1024" / (role + ".png")).relative_to(root)))
        with Image.open(path) as source, Image.open(derived) as saved:
            x, y = crop["x"], crop["y"]
            if (source.size != (crop["source_width"], crop["source_height"])
                    or source.mode != "RGB" or source.getexif().get(274, 1) != 1
                    or x < 0 or y < 0 or x+1024 > source.width or y+1024 > source.height
                    or saved.mode != "RGB" or saved.size != (1024, 1024)
                    or saved.tobytes() != source.crop((x, y, x+1024, y+1024)).tobytes()
                    or saved.info.get("icc_profile") != source.info.get("icc_profile")):
                raise ValueError("Native crop no longer matches its original: " + str(derived))
        verified[role] = {"original": str(path.resolve()), "sha256": record["sha256"],
                          "native_1024": str(derived.resolve()), "crop_sha256": sha(derived)}
    return {"id": row["id"], "group": group, "files": verified,
            "category": row.get("category"), "reviewed": row.get("reviewed", False)}


def prepare(reviews, validation_groups, test_groups):
    validation, test = set(validation_groups), set(test_groups)
    if not validation or validation & test:
        raise ValueError("Require a validation scene and disjoint validation/test scenes")
    rows = []
    identities = set()
    for review in reviews:
        review = review.expanduser().resolve()
        receipt_path = review / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "photo-capture-review-v1" or receipt.get("dataset") != REPO
                or receipt.get("revision") != REVISION or receipt.get("diffuse_target_claim") is not False):
            raise ValueError("Expected a pinned SIDD photo review, not arbitrary diffuse targets")
        for pair in receipt["pairs"]:
            row = verify_pair(review, pair)
            # Same capture can be sampled into two receipts; it is not extra data.
            if row["id"] in identities:
                raise ValueError("Duplicate capture across reviews: " + row["id"])
            identities.add(row["id"])
            row["split"] = "validation" if row["group"] in validation else "test" if row["group"] in test else "train"
            row["source_receipt"] = {"file": str(receipt_path), "sha256": sha(receipt_path)}
            rows.append(row)
    groups = {row["group"] for row in rows}
    if not (validation | test) <= groups:
        raise ValueError("Requested holdout includes a scene absent from these reviews")
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("No training scenes remain; one surface cannot provide independent train/validation")
    counts = {split: {"pairs": sum(row["split"] == split for row in rows),
                      "physical_scenes": len({row["group"] for row in rows if row["split"] == split})}
              for split in ["train", "validation", "test"]}
    return {"schema": "photo-capture-diagnostic-split-v1", "dataset": REPO, "revision": REVISION,
            "task": "real-phone-denoising", "diagnostic_only": True, "ready_for_training": False,
            "reason": "Small acquisition sample; manual alignment/category review and broader surface coverage required",
            "diffuse_target_claim": False, "counts": counts, "pairs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-groups", nargs="+", required=True)
    parser.add_argument("--test-groups", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        if output.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Split outputs belong in user storage outside the source repository")
        if output.exists():
            raise ValueError("Preserving existing output; choose a new split filename")
        result = prepare(args.reviews, args.validation_groups, args.test_groups)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["counts"]))
        print("Verified native pixels and scene isolation. Diagnostic only; no training or approval.")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
