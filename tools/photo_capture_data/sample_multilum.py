#!/usr/bin/env python3
"""Bounded real repeated-lighting samples from MIT Multi-Illumination.

Downloads two author-example scenes at upstream mip1, preserving JPEG bytes.
These photographs are not neutral albedo targets. No model or training is run.
"""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from sample_sidd import copy_stream, sha

SCENES = {
    "main_experiment120": (14659767, "e828b07db77e4532565cfdb6142219c05a1844bb4faa8b994a54edd2f3a3458a"),
    "elm_revis_kitchen14": (33648169, "ec719b63c5e6ca6f44bf69cc4f6f89d9f36bc7f022fe134fdce4531bea115130"),
}
DIRECTIONS = [0, 6, 12, 18, 24]


def extract_photos(archive, output, scene):
    """Extract only explicit known JPEG entries; no bulk archive extraction."""
    rows = []
    with zipfile.ZipFile(archive) as source:
        for direction in DIRECTIONS:
            basename = f"dir_{direction}_mip1.jpg"
            matches = [entry for entry in source.infolist() if Path(entry.filename).name == basename]
            if len(matches) != 1 or not 0 < matches[0].file_size <= 16*2**20:
                raise ValueError("Missing, duplicate or oversized illumination photo: " + basename)
            data = source.read(matches[0])  # ZipFile verifies CRC; archive SHA is checked by the caller.
            target = output / scene / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("Preserving existing photograph: " + str(target))
            with target.open("xb") as stream:
                stream.write(data)
            with Image.open(target) as photo:
                if photo.size != (3000, 2000) or photo.mode != "RGB":
                    raise ValueError("Unexpected upstream mip1 photograph dimensions/color")
                dimensions = photo.size
            rows.append({"direction": direction, "file": str(target.relative_to(output)),
                         "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "dimensions": dimensions})
    return rows


def verify_existing(root):
    """Read-only check of acquired archives and byte-identical extracted photos."""
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("dataset") != "MIT Multi-Illumination":
        raise ValueError("Not a MIT Multi-Illumination review")
    if sorted(r["scene"] for r in receipt["records"]) != sorted(SCENES):
        raise ValueError("Expected both pinned repeated-light scenes exactly once")
    for record in receipt["records"]:
        scene = record["scene"]
        expected_size, expected_sha = SCENES[scene]
        archive = root / (scene + ".zip")
        if archive.is_symlink() or archive.stat().st_size != expected_size or sha(archive) != expected_sha:
            raise ValueError("Archive identity differs: " + scene)
        if sorted(r["direction"] for r in record["source_photographs"]) != DIRECTIONS:
            raise ValueError("Expected five distinct selected lighting directions")
        with zipfile.ZipFile(archive) as source:
            for row in record["source_photographs"]:
                path = root / row["file"]
                if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                        or row["file"] != f"{scene}/dir_{row['direction']}_mip1.jpg"
                        or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]):
                    raise ValueError("Extracted photograph identity differs")
                names = [n for n in source.namelist() if Path(n).name == path.name]
                if len(names) != 1 or hashlib.sha256(source.read(names[0])).hexdigest() != row["sha256"]:
                    raise ValueError("Extracted bytes differ from the pinned source archive")
    return len(receipt["records"])


def sample(output, max_bytes):
    output = output.expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Downloads belong in user storage outside the source repository")
    if output.exists():
        raise ValueError("Choose a new output directory; preserve existing reviews")
    if sum(size for size, _ in SCENES.values()) > max_bytes:
        raise ValueError("The two pinned archives require 46.1 MiB; increase the explicit budget")
    output.mkdir(parents=True)
    records = []
    downloaded = 0
    for scene, (size, checksum) in SCENES.items():
        url = f"https://data.csail.mit.edu/multilum/{scene}/{scene}_mip1_jpg.zip"
        archive = output / (scene + ".zip")
        print("Downloading " + scene, flush=True)
        with urllib.request.urlopen(url, timeout=45) as response:
            downloaded += copy_stream(response, archive, {"size": size, "lfs": {"sha256": checksum}}, max_bytes-downloaded)
        records.append({"scene": scene, "url": url, "archive_sha256": checksum,
                        "archive_bytes": size, "upstream_mip": 1,
                        "same_camera_different_lights": True, "neutral_reference_available": False,
                        "source_photographs": extract_photos(archive, output, scene)})
    receipt = {"dataset": "MIT Multi-Illumination", "source": "https://projects.csail.mit.edu/illumination/",
               "license": "CC-BY (upstream project declaration)", "download_limit_bytes": max_bytes,
               "records": records, "training_performed": False, "derived_albedo_downloaded": False,
               "original_download_bytes_unchanged": True, "preview_resized_only": True,
               "resolution_note": "Author-provided mip1 derivative; not full-resolution RAW"}
    with (output / "receipt.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return verify_existing(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify-existing", type=Path)
    parser.add_argument("--max-mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 128:
        parser.error("Use a bounded archive budget of 1...128 MiB")
    try:
        count = verify_existing(args.verify_existing.expanduser().resolve()) if args.verify_existing else sample(args.output, args.max_mib*2**20)
        print(f"Verified {count} real repeated-light scenes. No albedo claim or training.")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
