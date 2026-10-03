"""Separate RAFT Studio process API; IPDE's extraction interface stays focused."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from typing import Sequence


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("depthpro", "depth-anything-v2", "depth-anything-3"), default="depthpro")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=1036)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raft-studio", description="Manage local teacher datasets and distill RAFT models for IPDE.")
    parser.add_argument("--json", action="store_true", help="one structured JSON response; logs go to stderr")
    commands = parser.add_subparsers(dest="command", required=True)
    workspace = commands.add_parser("workspace", help="list datasets and training runs")
    workspace.add_argument("workspace", type=Path)
    inspect = commands.add_parser("inspect-dataset", help="verify a dataset's arrays and provenance")
    inspect.add_argument("dataset", type=Path)
    dataset = commands.add_parser("dataset", help="generate a lossless teacher-target dataset")
    dataset.add_argument("sources", type=Path, nargs="+")
    dataset.add_argument("--output-dir", required=True, type=Path)
    dataset.add_argument("--groups", type=Path, help="JSON mapping absolute source paths to scene IDs")
    dataset.add_argument("--grouping", choices=("capture", "scene"), default="capture", help="use scene only when groups are verified independent scenes")
    dataset.add_argument("--metric-anchor", choices=("depthpro",), help="explicitly anchor a relative teacher's scale to a separate DepthPro estimate")
    dataset.add_argument("--anchor-model-path", type=Path)
    dataset.add_argument("--anchor-source-dir", type=Path)
    _model_arguments(dataset)
    teacher = commands.add_parser("teacher", help="export teacher estimates for visual and numerical comparison")
    teacher.add_argument("sources", type=Path, nargs="+")
    teacher.add_argument("--output-dir", required=True, type=Path)
    teacher.add_argument("--select", action="append", help="learned-depth/native/preview or learned-display-depth/native/preview")
    teacher.add_argument("--overwrite", action="store_true")
    _model_arguments(teacher)
    train = commands.add_parser("train", help="fine-tune a real RAFT checkpoint on teacher-generated flow targets")
    train.add_argument("dataset", type=Path)
    train.add_argument("--checkpoint", required=True, type=Path)
    train.add_argument("--raft-root", type=Path)
    train.add_argument("--raft-model", type=Path)
    train.add_argument("--raft-model-member")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--steps", type=int, default=16)
    train.add_argument("--patch-size", type=int, default=256)
    train.add_argument("--iterations", type=int, default=4)
    train.add_argument("--scope", choices=("update", "full"), default="update")
    train.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    train.add_argument("--photometric-support", action="store_true")
    export = commands.add_parser("export", help="verify and export a RAFT checkpoint ready to select in IPDE")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--output", required=True, type=Path, help="new export directory for raft-model.pth and provenance")
    return parser


def workspace_report(workspace: Path) -> dict:
    root = workspace.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("workspace path is not a directory")
    result = {"workspace": str(root), "datasets": [], "runs": [], "warnings": []}
    # Read only small manifests. Full array verification belongs to inspection
    # and the training preflight, rather than every GUI refresh.
    for path in sorted((root / "datasets").glob("*/dataset.json")):
        try:
            data = json.loads(path.read_text())
            if data.get("schema") != "ipde-depth-dataset-v1":
                raise ValueError("unsupported dataset schema")
            samples = data.get("samples", [])
            result["datasets"].append({"path": str(path.parent), "name": path.parent.name,
                "sample_count": len(samples), "teacher": samples[0].get("teacher", {}).get("metadata", {}).get("model_id", "") if samples else "",
                "train_count": sum(s.get("split") == "train" for s in samples),
                "validation_count": sum(s.get("split") == "validation" for s in samples)})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            result["warnings"].append(f"{path.name}: {exc}")
    for path in sorted((root / "runs").rglob("*.pth.json")):
        try:
            data = json.loads(path.read_text())
            checkpoint = path.with_suffix("")
            if checkpoint.is_file():
                result["runs"].append({"path": str(checkpoint), "name": checkpoint.name,
                    "best_epoch": data.get("best_epoch"), "validation": data.get("validation", {}),
                    "training_steps": data.get("total_steps", data.get("training_steps")),
                    "checkpoint_sha256": data.get("checkpoint_sha256")})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            result["warnings"].append(f"{path.name}: {exc}")
    return result


def _config(args):
    from .learned_depth import LearnedDepthConfig
    return LearnedDepthConfig(model=args.model, model_path=args.model_path, source_dir=args.source_dir,
                              device=args.device, input_size=args.input_size)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Third-party model imports/loggers must not corrupt the GUI JSON protocol.
        with redirect_stdout(sys.stderr):
            if args.command == "workspace":
                report = workspace_report(args.workspace)
            elif args.command == "inspect-dataset":
                from .dataset import load_dataset
                report = load_dataset(args.dataset)
            elif args.command == "dataset":
                from .dataset import DatasetOptions, build_dataset
                from .learned_depth import LearnedDepthConfig
                groups = json.loads(args.groups.read_text()) if args.groups else None
                if groups is not None and (not isinstance(groups, dict) or any(not isinstance(v, str) for v in groups.values())):
                    raise ValueError("scene groups must be a JSON object mapping source paths to scene ID strings")
                report = build_dataset(args.sources, args.output_dir,
                    DatasetOptions(teacher=_config(args), group_ids=groups, include_display_teacher=True, grouping_semantics=args.grouping,
                        metric_anchor=LearnedDepthConfig(model="depthpro", model_path=args.anchor_model_path,
                            source_dir=args.anchor_source_dir, device=args.device) if args.metric_anchor else None))
                report = {"dataset_path": str(args.output_dir.expanduser().resolve()), **report}
            elif args.command == "teacher":
                from .extractor import ExtractOptions, extract_file
                selections = tuple(args.select or ("learned-depth", "learned-native", "learned-preview",
                    "learned-display-depth", "learned-display-native", "learned-display-preview"))
                if any(not product.startswith("learned-") for product in selections):
                    raise ValueError("teacher exports accept learned-* products; use IPDE for embedded/stereo products")
                report = {"sources": [extract_file(source, ExtractOptions(
                    selected_products=selections, output_dir=args.output_dir, write_manifest=True,
                    learned_model=args.model, learned_model_path=args.model_path,
                    learned_source_dir=args.source_dir, learned_device=args.device,
                    learned_input_size=args.input_size, overwrite=args.overwrite,
                )) for source in args.sources]}
            elif args.command == "train":
                from .training import TrainingOptions, train_dataset
                report = train_dataset(args.dataset, args.checkpoint, TrainingOptions(
                    epochs=args.epochs, steps_per_epoch=args.steps, patch_size=args.patch_size,
                    iterations=args.iterations, train_scope=args.scope, device=args.device,
                    raft_root=args.raft_root, raft_model=args.raft_model, raft_model_member=args.raft_model_member,
                    require_photometric_support=args.photometric_support,
                ))
            else:
                from .training import export_raft_checkpoint
                report = export_raft_checkpoint(args.checkpoint, args.output)
        print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=None if args.json else 2))
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}) if args.json else f"Error: {exc}", file=sys.stdout if args.json else sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
