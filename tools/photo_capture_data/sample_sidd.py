#!/usr/bin/env python3
"""Bounded acquisition/review of REAL matched phone captures, not PBR assets.

This development utility never trains or runs a model, changes Factory's source
project, or treats a denoised photograph as shadow-free diffuse ground truth.
"""
import argparse
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = "MhdMohammadi/SIDD-CC"
REVISION = "5eb2ab6375d7f99f36b59bb5aacb46af18409412"
CAPTURE = re.compile(r"(?P<instance>\d{4})_(?P<scene>\d{3})_(?P<camera>\w{2})_"
                     r"(?P<iso>\d{5})_(?P<shutter>\d{5})_(?P<kelvin>\d{4})_(?P<light>[LNH])")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Hub metadata exceeded its 4 MiB limit")
    return json.loads(data)


def capture_pairs(files):
    """Pair the SAME instance AND burst frame; group every camera by scene."""
    by_path = {f["rfilename"]: f for f in files}
    pairs = []
    for name in sorted(by_path):
        parts = name.split("/")
        if len(parts) != 5 or parts[:3] != ["SRGB", "train", "Data"]:
            continue
        match = CAPTURE.fullmatch(parts[3])
        if not match:
            continue
        frame = re.fullmatch(match["instance"] + r"_NOISY_SRGB_(\d+)\.PNG", parts[4])
        if not frame:
            continue
        target = name.replace("_NOISY_SRGB_", "_GT_SRGB_")
        if target not in by_path:
            raise ValueError("Missing reference for " + name)
        pairs.append({"capture": match.groupdict(), "frame": frame[1],
                      "group": "sidd-scene-" + match["scene"],
                      "input": by_path[name], "reference": by_path[target]})
    return pairs


def select_variants(pairs, scenes, variants, scene_ids=None):
    # Prefer one camera per scene for a readable ISO comparison. Cross-camera
    # views still share the physical-scene group, but are never cross-paired.
    groups = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        c = p["capture"]
        groups[p["group"]][c["camera"]].setdefault(c["instance"], p)
    selected = []
    names = sorted(groups)
    if scene_ids:
        names = ["sidd-scene-" + x for x in sorted(set(scene_ids))]
        if any(name not in groups for name in names):
            raise ValueError("Requested scene is not available in this pinned SIDD release")
    for group in names[:scenes]:
        cameras = groups[group]
        camera = sorted(cameras, key=lambda k: (-len({p["capture"]["iso"] for p in cameras[k].values()}), k))[0]
        choices = sorted(cameras[camera].values(), key=lambda p: (int(p["capture"]["iso"]), p["capture"]["instance"]))
        # Select across the available range, retaining the actual capture label.
        count = min(variants, len(choices))
        indices = sorted({round(i * (len(choices)-1) / max(count-1, 1)) for i in range(count)})
        selected.extend(choices[i] for i in indices)
    return selected


def copy_stream(response, destination, expected, remaining):
    size = expected["size"]
    digest = expected.get("lfs", {}).get("sha256")
    if not digest or size > remaining:
        raise ValueError("Missing upstream SHA256 or acquisition byte budget exceeded")
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != size or sha(destination) != digest:
            raise ValueError("Existing file differs; keeping it untouched: " + str(destination))
        return 0
    temporary = destination.with_name(destination.name + ".pending")
    if temporary.exists():
        raise ValueError("Unfinished download exists; choose a new output directory: " + str(temporary))
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with temporary.open("xb") as stream:
            while chunk := response.read(min(1024*1024, size-written+1)):
                written += len(chunk)
                if written > size or written > remaining:
                    raise ValueError("Download exceeded declared size/budget")
                stream.write(chunk)
        if written != size or sha(temporary) != digest:
            raise ValueError("Downloaded source failed its upstream size/SHA256 check")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return written


def native_crop(input_path, reference_path, out, size=1024):
    from PIL import Image
    if size < 32 or size > 8192:
        raise ValueError("Native crop size must be 32...8192")
    for path in [input_path, reference_path]:
        with path.open("rb") as stream:
            header = stream.read(25)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[24:25] != b"\x08":
            raise ValueError("This SIDD sRGB crop loader requires original 8-bit PNGs; it cannot silently reduce precision")
    with Image.open(input_path) as a, Image.open(reference_path) as b:
        if a.size != b.size or min(a.size) < size or a.mode != "RGB" or b.mode != "RGB":
            raise ValueError("Pair must be same-size RGB and large enough for the native crop")
        if a.getexif().get(274, 1) != 1 or b.getexif().get(274, 1) != 1:
            raise ValueError("Oriented captures require an explicit alignment derivative first")
        x, y = (a.width-size)//2, (a.height-size)//2
        box = (x, y, x+size, y+size)
        first, second = a.crop(box), b.crop(box)
        out.mkdir(parents=True, exist_ok=False)
        for name, cropped, original in [("input.png", first, a), ("reference.png", second, b)]:
            destination = out / name
            cropped.save(destination, icc_profile=original.info.get("icc_profile"))
            with Image.open(destination) as saved:
                if saved.tobytes() != cropped.tobytes():
                    raise ValueError("Native crop pixel verification failed")
        # The side-by-side image is a native-pixel REVIEW, not another target.
        review = Image.new("RGB", (size*2, size))
        review.paste(first, (0, 0)); review.paste(second, (size, 0))
        review.save(out / "input-left-reference-right.png")
        return {"source_width": a.width, "source_height": a.height,
                "x": x, "y": y, "width": size, "height": size,
                "resampled": False, "pixel_bytes_verified": True}


def sample(output, scenes, variants, max_bytes, metadata=None, scene_ids=None):
    output = output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo_root):
        raise ValueError("Downloads belong in user storage, outside the source repository")
    if (output / "receipt.json").exists():
        raise ValueError("This review already exists; choose a new directory to preserve it")
    output.mkdir(parents=True, exist_ok=True)
    info = metadata or read_json(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true")
    if info["sha"] != REVISION:
        raise ValueError("The Hub returned a different revision")
    pairs = capture_pairs(info["siblings"])
    chosen = select_variants(pairs, scenes, variants, scene_ids)
    if not chosen:
        raise ValueError("No complete full-resolution real capture pairs were found")
    total = sum(p[k]["size"] for p in chosen for k in ["input", "reference"])
    if total > max_bytes:
        raise ValueError(f"Selected originals require {total/2**20:.1f} MiB; budget is {max_bytes/2**20:.1f} MiB. Use fewer scenes/variants or a larger explicit budget.")
    receipt = {"schema": "photo-capture-review-v1", "dataset": REPO, "revision": REVISION,
               "task": "real-phone-denoising", "downloaded_bytes": 0, "original_bytes": total,
               "physical_scene_groups_in_source_train": len({p['group'] for p in pairs}),
               "source_pairs_available": len(pairs), "pairs": [], "trained": False,
               "category_coverage": "unreviewed", "diffuse_target_claim": False,
               "license_source": "https://abdokamel.github.io/sidd/#license",
               "license_note": "Original SIDD states MIT. This reprocessing inherits SIDD terms; retain both dataset citations."}
    for p in chosen:
        identity = p["capture"]["instance"] + "-" + p["frame"]
        folder = output / p["group"] / identity
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "pair.json").exists():
            raise ValueError("A reviewed pair already exists; choose a new output directory")
        files = {}
        for role in ["input", "reference"]:
            record = p[role]; remote = record["rfilename"]
            destination = folder / (role + ".PNG")
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(remote, safe='/')}"
            if destination.exists():
                with io.BytesIO() as stream:
                    receipt["downloaded_bytes"] += copy_stream(stream, destination, record, max_bytes-receipt["downloaded_bytes"])
            else:
                print("Downloading " + remote, flush=True)
                with urllib.request.urlopen(url, timeout=45) as response:
                    receipt["downloaded_bytes"] += copy_stream(response, destination, record, max_bytes-receipt["downloaded_bytes"])
            files[role] = {"file": str(destination.relative_to(output)), "upstream_file": remote,
                           "bytes": record["size"], "sha256": sha(destination)}
        crop = native_crop(folder/"input.PNG", folder/"reference.PNG", folder/"native-1024")
        row = {"id": identity, "group": p["group"], "capture": p["capture"], "frame": p["frame"],
               "files": files, "crop": crop, "split": "unassigned", "reviewed": False,
               "category": None, "alignment": "dataset-paired; manual native-pixel review still required"}
        (folder/"pair.json").write_text(json.dumps(row, indent=2)+"\n")
        receipt["pairs"].append(row)
        (output/"receipt.in-progress.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output/"receipt.in-progress.json").unlink(missing_ok=True)
    print(f"Saved {len(chosen)} real pairs; {total/2**20:.1f} MiB of originals. No training or category assignment.")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--max-mib", type=int, default=256)
    parser.add_argument("--scene-ids", nargs="+", help="Explicit physical scene numbers, e.g. 008 for the photographed carpet")
    args = parser.parse_args()
    if not (1 <= args.scenes <= 10 and 1 <= args.variants <= 10 and 1 <= args.max_mib <= 2048):
        parser.error("Use 1...10 scenes/variants and an explicit 1...2048 MiB byte budget")
    if args.scene_ids and (len(args.scene_ids)>10 or any(not re.fullmatch(r"\d{3}", x) for x in args.scene_ids)):
        parser.error("Use up to ten three-digit physical scene numbers")
    try:
        sample(args.output, len(args.scene_ids) if args.scene_ids else args.scenes,
               args.variants, args.max_mib*2**20, scene_ids=args.scene_ids)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+"\n")


if __name__ == "__main__":
    main()
