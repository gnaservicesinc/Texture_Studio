"""JSON process interface shared by Photo Studio, Raw Studio and CLI users."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .extractor import _jsonable
from .learned_depth import LearnedDepthConfig
from .photo_workflow import MediaWorkflowError, export_photo, inspect_photo, rewrite_photo
from .raw_workflow import export_raw, inspect_raw
from .spatial import RaftStereoOptions


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("depthpro", "depth-anything-v2", "depth-anything-v2-small", "depth-anything-3"), default="depthpro")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-source-dir", "--source-dir", dest="model_source_dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=1036)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Precision-preserving portrait and RAW workflows")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("photo-inspect", "raw-inspect"):
        p = commands.add_parser(command)
        p.add_argument("source", type=Path)
        p.add_argument("--preview-dir", type=Path)
    for command in ("photo-export", "raw-export"):
        p = commands.add_parser(command)
        p.add_argument("source", type=Path)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--operation", required=True, choices=("original", "depth-upscale", "cutout", "isolate", "learned-depth", "raft-depth")
                       if command == "photo-export" else ("original", "sensor", "rendered", "auxiliary", "learned-depth", "person-mask"))
        p.add_argument("--format", choices=("npy", "png", "tiff", "exr"), default="npy")
        p.add_argument("--asset", type=int)
        _model_options(p)
        if command == "photo-export":
            p.add_argument("--matte", action="append", type=int, default=[])
            p.add_argument("--raft-model", type=Path)
            p.add_argument("--raft-root", type=Path)
            p.add_argument("--raft-model-member")
            p.add_argument("--raft-iterations", type=int, default=32)
        else:
            p.add_argument("--crop", help="x,y,width,height in the selected product's encoded grid")
    p = commands.add_parser("photo-rewrite")
    p.add_argument("source", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--privacy", action="store_true")
    p.add_argument("--replacement", action="append", default=[], metavar="INDEX=PATH",
                   help="Replace an Apple native depth plane with explicitly prepared float values")
    p.add_argument("--replacement-kind", choices=("native", "depth", "disparity"), default="native",
                   help="native preserves representation; depth is meters; disparity is inverse meters")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        # Models and native dependencies can print diagnostics; reserve stdout
        # for exactly one machine-readable JSON result.
        with redirect_stdout(sys.stderr):
            if args.command == "photo-inspect":
                report = inspect_photo(args.source, args.preview_dir)
            elif args.command == "raw-inspect":
                report = inspect_raw(args.source, args.preview_dir)
            elif args.command == "photo-rewrite":
                replacements = {}
                for specification in args.replacement:
                    index, separator, path = specification.partition("=")
                    if not separator or not path:
                        raise MediaWorkflowError("Replacement must use INDEX=PATH")
                    if int(index) in replacements:
                        raise MediaWorkflowError("Only one replacement per asset may be supplied")
                    replacements[int(index)] = Path(path)
                report = rewrite_photo(args.source, args.output_dir, privacy=args.privacy,
                                       replacements=replacements, replacement_kind=args.replacement_kind)
            else:
                learned = LearnedDepthConfig(model=args.model, model_path=args.model_path,
                    source_dir=args.model_source_dir, device=args.device, input_size=args.input_size)
                if args.command == "photo-export":
                    report = export_photo(args.source, args.output_dir, args.operation, args.format,
                        asset_index=args.asset, matte_indices=args.matte, learned=learned,
                        raft=RaftStereoOptions(root=args.raft_root, model=args.raft_model,
                            model_member=args.raft_model_member, device=args.device, iterations=args.raft_iterations))
                else:
                    crop = tuple(int(value.strip()) for value in args.crop.split(",")) if args.crop else None
                    if crop is not None and len(crop) != 4:
                        raise MediaWorkflowError("Crop must contain x,y,width,height")
                    report = export_raw(args.source, args.output_dir, args.operation, args.format,
                                        crop=crop, learned=learned, asset_index=args.asset)
        print(json.dumps(_jsonable(report), allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "operation": args.command}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
