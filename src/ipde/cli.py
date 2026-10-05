"""Command-line interface for IPDE."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .extractor import ExtractOptions, ExtractionError, extract_file, inspect_file
from .spatial import RaftStereoOptions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipde",
        description=(
            "Extract HEIF depth, auxiliary planes, and spatial-photo stereo views without "
            "normalization or display transforms."
        ),
    )
    parser.add_argument("sources", nargs="+", type=Path, help="HEIC/HEIF input file(s)")
    parser.add_argument("--select", action="append", metavar="PRODUCT_ID", help="export only this product (repeatable; IDs are in --inspect --json)")
    parser.add_argument("--inspect", action="store_true", help="inventory assets without writing output files")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per source")
    parser.add_argument("--output-dir", type=Path, help="output directory (default: each source directory)")
    parser.add_argument("--overwrite", action="store_true", help="replace colliding outputs after verification")
    parser.add_argument("--no-npy", action="store_true", help="omit exact NumPy array companions")
    parser.add_argument("--manifest", action="store_true", help="also write a provenance JSON manifest (off by default)")
    parser.add_argument("--learned-depth", action="store_true",
                        help="include AI depth in inspection; without --select export full display and native model predictions as float32 EXR")
    parser.add_argument("--learned-model", choices=("depthpro", "depth-anything-v2", "depth-anything-3"), default="depthpro")
    parser.add_argument("--learned-model-path", type=Path)
    parser.add_argument("--learned-source-dir", type=Path)
    parser.add_argument("--learned-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--learned-input-size", type=int, default=1036,
                        help="V2 shortest side / DA3 longest side (default 1036); 0 requests native input, subject to DA3's patch limit")
    parser.add_argument(
        "--no-metric-depth",
        action="store_true",
        help="do not calibrate eligible uint8 uniform-disparity planes as float32 meter EXRs",
    )
    parser.add_argument(
        "--no-physical-disparity",
        action="store_true",
        help="do not calibrate eligible uint8 uniform-disparity planes as float32 inverse-meter EXRs",
    )
    parser.add_argument(
        "--displacement-maps",
        action="store_true",
        help=(
            "also export explicit float32 0..1 displacement derivatives using the "
            "full linear-depth range per map"
        ),
    )
    parser.add_argument(
        "--color-matching",
        action="store_true",
        help="histogram-match the non-Hero stereo view's RGB channels before inference",
    )
    parser.add_argument(
        "--color-hero",
        choices=("left", "right"),
        default="left",
        help="Hero view retained unchanged by --color-matching (default: left)",
    )
    parser.add_argument(
        "--raft-stereo",
        action="store_true",
        help="infer one full-resolution float32 RAFT-Stereo height map for an Apple Spatial Photo",
    )
    parser.add_argument(
        "--raft-diagnostics",
        action="store_true",
        help="also export RAFT signed flow and calibrated metric distance diagnostic EXRs",
    )
    parser.add_argument(
        "--raft-root",
        type=Path,
        help="RAFT-Stereo checkout (otherwise auto-detected or read from IPDE_RAFT_STEREO_DIR)",
    )
    parser.add_argument(
        "--raft-model",
        type=Path,
        help="selected RAFT-Stereo or trained depth checkpoint, or ZIP (otherwise auto-detected or read from IPDE_RAFT_MODEL)",
    )
    parser.add_argument(
        "--raft-model-member",
        help="checkpoint member inside --raft-model ZIP (default: raftstereo-middlebury.pth)",
    )
    parser.add_argument(
        "--raft-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="inference device (auto prefers CUDA, then Apple Metal, then CPU)",
    )
    parser.add_argument(
        "--raft-iterations",
        type=int,
        default=32,
        metavar="N",
        help="full-resolution RAFT update iterations in [1, 256] (default: 32)",
    )
    parser.add_argument("--version", action="version", version=f"IPDE {__version__}")
    return parser


def _human_report(report: dict, inspected: bool) -> str:
    source = report["source"]
    lines = [
        f"{'Inspected' if inspected else 'Extracted'}: {source['path']}",
        f"Container: {source['mimetype']} | top-level images: {len(source['top_level_images'])}",
        f"Auxiliary planes: {report['asset_count']}",
    ]
    spatial = source.get("spatial_photo")
    if spatial:
        lines.append(
            "Spatial Photo: left image {left}, right image {right}, baseline {baseline:.6f} mm, "
            "disparity adjustment {adjustment:.4%}".format(
                left=spatial["left_image_index"],
                right=spatial["right_image_index"],
                baseline=spatial["baseline_millimeters"],
                adjustment=spatial["disparity_adjustment_fraction_of_width"],
            )
        )
    for asset in report["assets"]:
        shape = f"{asset['width']}x{asset['height']}"
        if asset["channels"] > 1:
            shape += f"x{asset['channels']}ch"
        lines.append(
            f"  {asset['semantic_name']}: {shape} {asset['dtype_name']} "
            f"(source {asset['source_bit_depth']}-bit, min={asset['minimum']}, max={asset['maximum']})"
        )
        for output in asset.get("outputs", []):
            lines.append(f"    {output['role']}: {output['path']}")
    for warning in report.get("warnings", []):
        lines.append(f"Warning: {warning}")
    if not inspected and report.get("manifest_path"):
        lines.append(f"Manifest: {report['manifest_path']}")
    return "\n".join(lines)


def _backend_response(operation, *args, **kwargs):
    # Third-party model constructors may print to stdout. Reserve stdout for
    # the one JSON result consumed by the GUI and command-line clients.
    with redirect_stdout(sys.stderr):
        return operation(*args, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selections = tuple(args.select) if args.select is not None else (
        ("learned-display-depth", "learned-display-native") if args.learned_depth else None)
    include_learned = args.learned_depth or bool(selections and any(key.startswith("learned-") for key in selections))
    failed = False
    for source in args.sources:
        try:
            if args.inspect:
                # Direct AI inventory does not depend on a stale RAFT selection
                # or IPDE_RAFT_MODEL environment setting.
                raft_options = None if include_learned else RaftStereoOptions(
                    root=args.raft_root, model=args.raft_model, model_member=args.raft_model_member)
                report = _backend_response(inspect_file, source, include_learned=include_learned, raft_options=raft_options)
            else:
                report = _backend_response(extract_file,
                    source,
                    ExtractOptions(
                        selected_products=selections,
                        output_dir=args.output_dir,
                        write_npy=not args.no_npy,
                        write_manifest=args.manifest,
                        write_metric_depth=not args.no_metric_depth,
                        write_physical_disparity=not args.no_physical_disparity,
                        write_raft_stereo=(
                            args.raft_stereo
                            or args.raft_diagnostics
                        ),
                        write_raft_diagnostics=args.raft_diagnostics,
                        histogram_color_matching=args.color_matching,
                        color_matching_hero=args.color_hero,
                        write_displacement_maps=args.displacement_maps,
                        raft_root=args.raft_root,
                        raft_model=args.raft_model,
                        raft_model_member=args.raft_model_member,
                        raft_device=args.raft_device,
                        raft_iterations=args.raft_iterations,
                        learned_model=args.learned_model,
                        learned_model_path=args.learned_model_path,
                        learned_source_dir=args.learned_source_dir,
                        learned_device=args.learned_device,
                        learned_input_size=args.learned_input_size,
                        write_learned_depth=args.learned_depth,
                        overwrite=args.overwrite,
                    ),
                )
            if args.json:
                print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            else:
                print(_human_report(report, args.inspect))
        except (ExtractionError, OSError, ValueError) as exc:
            failed = True
            error = {"source": str(source.expanduser().resolve()), "error": str(exc)}
            if args.json:
                print(json.dumps(error, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            else:
                print(f"Error: {error['source']}: {error['error']}", file=sys.stderr)
    return 1 if failed else 0
