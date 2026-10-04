"""Separate RAFT Studio process API; IPDE's extraction interface stays focused."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import json
import sys
from pathlib import Path
from typing import Sequence


@contextmanager
def _edit_cancellation():
    """Let QProcess.terminate() unwind atomic edit staging before exit."""
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    def cancelled(signum, frame):
        raise RuntimeError("Dataset editing cancelled; source datasets remain unchanged")
    signal.signal(signal.SIGTERM, cancelled)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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
    workspace.add_argument("--linked-dataset", type=Path, action="append", default=[],
                           help="include an existing dataset directory by reference without copying files")
    inspect = commands.add_parser("inspect-dataset", help="verify a dataset's arrays and provenance")
    inspect.add_argument("dataset", type=Path)
    review = commands.add_parser("review-dataset", help="list samples and depth labels for visual review")
    review.add_argument("dataset", type=Path)
    preview = commands.add_parser("preview-sample", help="render disposable RGB/depth previews without changing dataset arrays")
    preview.add_argument("dataset", type=Path)
    preview.add_argument("--sample", required=True)
    preview.add_argument("--label", choices=("training", "teacher", "anchored_teacher", "metric_anchor",
        "display_teacher", "registered_display_teacher", "anchored_display_teacher", "display_metric_anchor", "reference"), default="training")
    preview.add_argument("--output-dir", required=True, type=Path)
    preview.add_argument("--max-dimension", type=int, default=1600, help="0 preserves full pixel size in display copies")
    comparison = commands.add_parser("compare-samples", help="compare two or three teacher variants of one photo")
    comparison.add_argument("dataset", type=Path)
    comparison.add_argument("--sample", action="append", required=True)
    comparison.add_argument("--output-dir", required=True, type=Path)
    comparison.add_argument("--max-dimension", type=int, default=0)
    models = commands.add_parser("compare-models", help="visual A/B of baseline and project RAFT on one spatial photo")
    models.add_argument("source", type=Path)
    models.add_argument("--baseline-model", type=Path, required=True)
    models.add_argument("--candidate-model", type=Path, required=True)
    models.add_argument("--raft-root", type=Path)
    models.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    models.add_argument("--iterations", type=int, default=32)
    models.add_argument("--output-dir", type=Path, required=True)
    scan = commands.add_parser("scan-spatial", help="recursively find valid calibrated spatial photos without following links")
    scan.add_argument("directory", type=Path)
    scan.add_argument("--no-recursive", action="store_true")
    compose = commands.add_parser("compose-datasets", help="assemble reviewed datasets into a new training set")
    compose.add_argument("datasets", nargs="+", type=Path)
    compose.add_argument("--output-dir", required=True, type=Path)
    compose.add_argument("--validation-dataset", action="append", type=Path, default=[])
    compose.add_argument("--split-mode", choices=("global-random", "equal-per-dataset", "explicit"), default="global-random")
    compose.add_argument("--validation-fraction", type=float, default=.2)
    compose.add_argument("--seed", type=int, default=0)
    compose.add_argument("--grouping", choices=("preserve", "ignore"), default="preserve")
    compose.add_argument("--validation-count-per-dataset", type=int)
    compose.add_argument("--storage-mode", choices=("shared", "copy"), default="shared",
                         help="reuse array storage without a full copy (default), or make a portable compressed copy")
    compact = commands.add_parser("compact-dataset", help="write a verified lossless compressed, deduplicated dataset copy")
    compact.add_argument("dataset", type=Path)
    compact.add_argument("--output-dir", required=True, type=Path)
    hf = commands.add_parser("import-hf", help="import mapped lossless stereo arrays through the optional datasets backend")
    hf.add_argument("dataset")
    hf.add_argument("--output-dir", type=Path, required=True)
    hf.add_argument("--mapping-json", type=Path, required=True)
    hf.add_argument("--config")
    hf.add_argument("--split", default="train")
    hf.add_argument("--revision")
    hf.add_argument("--asset-dir", type=Path)
    hf.add_argument("--data-files", help="local/Hub JSON or Parquet files for a datasets builder")
    curate = commands.add_parser("curate-dataset", help="copy kept samples into a new lossless dataset, preserving original splits")
    curate.add_argument("dataset", type=Path)
    curate.add_argument("--keep", action="append", required=True, help="sample ID to retain; repeat for each kept sample")
    curate.add_argument("--output-dir", required=True, type=Path)
    edit = commands.add_parser("edit-dataset", help="save image additions, removals and linked-group split choices into a fresh lossless dataset")
    edit.add_argument("dataset", type=Path)
    edit.add_argument("--edits-json", required=True, type=Path, help="object with keep base sample IDs and splits mapping IDs to train/validation")
    edit.add_argument("--add-dataset", action="append", type=Path, default=[], help="add every image/teacher entry from another prepared dataset")
    edit.add_argument("--output-dir", required=True, type=Path)
    dataset = commands.add_parser("dataset", help="generate a lossless teacher-target dataset")
    dataset.add_argument("sources", type=Path, nargs="+")
    dataset.add_argument("--output-dir", required=True, type=Path)
    dataset.add_argument("--groups", type=Path, help="JSON mapping absolute source paths to scene IDs")
    dataset.add_argument("--grouping", choices=("none", "capture", "scene"), default="capture", help="use scene only when groups are verified independent scenes")
    dataset.add_argument("--name", default="")
    dataset.add_argument("--category", default="")
    dataset.add_argument("--teachers-json", type=Path, help="array of 1-3 model configs; each produces a separately reviewed entry")
    dataset.add_argument("--include-display-teacher", action="store_true", help="also infer the separate display-camera grid (large storage cost)")
    dataset.add_argument("--uncompressed", action="store_true", help="store NPY instead of lossless NPZ")
    dataset.add_argument("--validation-fraction", type=float, default=.2)
    dataset.add_argument("--seed", type=int, default=0)
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
    train.add_argument("--mode", choices=("auto", "distillation", "supervised", "mixed"), default="distillation")
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
    archive = commands.add_parser("archive-dataset", help="move an owned dataset out of the library without deleting files")
    archive.add_argument("dataset", type=Path)
    archive.add_argument("--workspace", required=True, type=Path)
    for command, label in (("cleanup-dataset", "remove an owned generated dataset, preserving source photos and trained models"),
                           ("cleanup-run", "remove run intermediates while retaining checkpoints and provenance")):
        cleanup = commands.add_parser(command, help=label)
        cleanup.add_argument("dataset" if command == "cleanup-dataset" else "checkpoint", type=Path)
        cleanup.add_argument("--workspace", required=True, type=Path)
        cleanup.add_argument("--confirm", action="store_true", help="explicitly confirm permanent removal")
    for command in (inspect, scan, compose, compact, hf, curate, edit, dataset, train):
        command.add_argument("--workers", type=int, default=0,
                             help="CPU file/preparation workers; 0 uses available cores (default)")
    return parser


def workspace_report(workspace: Path, linked_datasets: Sequence[Path] = ()) -> dict:
    from .training import training_target_eligibility
    root = workspace.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("workspace path is not a directory")
    result = {"workspace": str(root), "datasets": [], "runs": [], "warnings": []}
    # Read only small manifests. Full array verification belongs to inspection
    # and the training preflight, rather than every GUI refresh.
    local_manifests = sorted((root / "datasets").glob("*/dataset.json"))
    candidates = [(path, False) for path in local_manifests]
    candidates.extend((path.expanduser() / "dataset.json", True) for path in linked_datasets)
    seen: set[Path] = set()
    for path, linked in candidates:
        try:
            directory = path.parent.resolve()
            if directory in seen:
                continue
            seen.add(directory)
            if not directory.is_dir():
                raise ValueError("linked dataset directory is unavailable")
            data = json.loads(path.read_text())
            if data.get("schema") != "ipde-depth-dataset-v1":
                raise ValueError("unsupported dataset schema")
            samples = data.get("samples", [])
            collection = data.get("collection") or {}
            storage = {key: collection[key] for key in ("storage_mode", "storage_format", "storage_methods",
                       "array_storage_bytes", "reused_array_storage_bytes", "added_array_storage_bytes", "added_storage_bytes") if key in collection}
            # Reviewed and compacted copies own newly copied payloads; inherited collection
            # provenance must not describe them as shared storage.
            if data.get("curation") or data.get("storage_compaction"):
                storage = {}
            result["datasets"].append({"path": str(directory), "name": data.get("name") or directory.name, "linked": linked,
                "sample_count": len(samples), "source_count": len({s.get("source_sha256", s.get("id")) for s in samples}),
                "training_mode": "supervised" if samples and all(s.get("training_target_choice") == "reference" for s in samples) else "mixed" if any(s.get("training_target_choice") == "reference" for s in samples) else "distillation",
                "category": data.get("category", ""), "generation_state": data.get("generation_state", "complete"),
                "storage_bytes": sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file() and not p.is_symlink()),
                "storage": storage, "training_eligibility": training_target_eligibility(data),
                "teacher": ", ".join(sorted({s.get("teacher", {}).get("metadata", {}).get("model_id", "") for s in samples} - {""})),
                "train_count": sum(s.get("split") == "train" for s in samples),
                "validation_count": sum(s.get("split") == "validation" for s in samples)})
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            result["warnings"].append(f"{path}: {exc}")
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


def _progress(event):
    print("IPDE_EVENT " + json.dumps(event, allow_nan=False), file=sys.stderr, flush=True)


def _teacher_configs(args):
    if not args.teachers_json:
        return [_config(args)], []
    from .learned_depth import LearnedDepthConfig
    entries = json.loads(args.teachers_json.read_text())
    if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
        raise ValueError("teachers JSON must be an array of one to three model configurations")
    configs, ids = [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("model") not in {"depthpro", "depth-anything-v2", "depth-anything-3"}:
            raise ValueError("Each teacher must specify a supported model")
        if set(entry) - {"id", "model", "model_path", "source_dir", "device", "input_size"}:
            raise ValueError("Unknown teacher configuration field")
        configs.append(LearnedDepthConfig(model=entry["model"], model_path=Path(entry["model_path"]) if entry.get("model_path") else None,
            source_dir=Path(entry["source_dir"]) if entry.get("source_dir") else None, device=entry.get("device", args.device),
            input_size=int(entry.get("input_size", args.input_size))))
        ids.append(str(entry.get("id", f"teacher-{index + 1}")))
    return configs, ids


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Third-party model imports/loggers must not corrupt the GUI JSON protocol.
        with redirect_stdout(sys.stderr), ExitStack() as locks:
            from .resource_lock import resource_lock
            if args.command == "edit-dataset":
                locks.enter_context(_edit_cancellation())
            if args.command in {"inspect-dataset", "review-dataset", "preview-sample", "compare-samples", "compose-datasets", "compact-dataset", "curate-dataset", "edit-dataset", "train"}:
                inputs = [*args.datasets, *args.validation_dataset] if args.command == "compose-datasets" else [args.dataset, *args.add_dataset] if args.command == "edit-dataset" else [args.dataset]
                for path in sorted({path.expanduser().resolve() for path in inputs}):
                    locks.enter_context(resource_lock(path, shared=True))
            if args.command == "train":
                locks.enter_context(resource_lock(args.checkpoint.expanduser().resolve().parent))
            if args.command in {"dataset", "compose-datasets", "compact-dataset", "curate-dataset", "edit-dataset", "import-hf"}:
                locks.enter_context(resource_lock(args.output_dir))
            if args.command == "export":
                locks.enter_context(resource_lock(args.checkpoint.expanduser().resolve().parent, shared=True))
            if args.command == "workspace":
                report = workspace_report(args.workspace, args.linked_dataset)
            elif args.command == "inspect-dataset":
                from .dataset import load_dataset
                report = load_dataset(args.dataset, workers=args.workers)
            elif args.command == "review-dataset":
                from .dataset_review import review_dataset
                report = review_dataset(args.dataset)
            elif args.command == "preview-sample":
                from .dataset_review import preview_sample
                report = preview_sample(args.dataset, args.sample, args.label, args.output_dir, max_dimension=args.max_dimension)
            elif args.command == "compare-samples":
                from .dataset_review import compare_samples
                report = compare_samples(args.dataset, args.sample, args.output_dir, max_dimension=args.max_dimension)
            elif args.command == "compare-models":
                from .model_comparison import compare_models
                report = compare_models(args.source, args.output_dir, baseline_model=args.baseline_model,
                    candidate_model=args.candidate_model, raft_root=args.raft_root, device=args.device, iterations=args.iterations)
            elif args.command == "scan-spatial":
                from .spatial_scan import scan_spatial_directory
                report = scan_spatial_directory(args.directory, recursive=not args.no_recursive, progress_callback=_progress, workers=args.workers)
            elif args.command == "compose-datasets":
                from .dataset_collection import CollectionOptions, compose_datasets
                report = compose_datasets(args.datasets, args.output_dir, CollectionOptions(split_mode=args.split_mode,
                    validation_fraction=args.validation_fraction, split_seed=args.seed, grouping=args.grouping,
                    validation_count_per_dataset=args.validation_count_per_dataset, storage_mode=args.storage_mode), validation_datasets=args.validation_dataset, progress_callback=_progress, workers=args.workers)
            elif args.command == "compact-dataset":
                from .dataset_collection import compress_dataset
                report = compress_dataset(args.dataset, args.output_dir, progress_callback=_progress, workers=args.workers)
            elif args.command == "import-hf":
                from .huggingface_datasets import import_huggingface_dataset
                report = import_huggingface_dataset(args.dataset, args.output_dir, json.loads(args.mapping_json.read_text()),
                    config=args.config, split=args.split, revision=args.revision, asset_dir=args.asset_dir, data_files=args.data_files, workers=args.workers)
            elif args.command == "curate-dataset":
                from .dataset_review import curate_dataset
                report = curate_dataset(args.dataset, args.keep, args.output_dir, workers=args.workers)
            elif args.command == "edit-dataset":
                from .dataset_edit import edit_dataset
                report = edit_dataset(args.dataset, json.loads(args.edits_json.read_text(encoding="utf-8")), args.output_dir,
                                      add_datasets=args.add_dataset, progress_callback=_progress, workers=args.workers)
            elif args.command == "dataset":
                from .dataset import DatasetOptions, build_dataset
                from .learned_depth import LearnedDepthConfig
                groups = json.loads(args.groups.read_text()) if args.groups else None
                if groups is not None and (not isinstance(groups, dict) or any(not isinstance(v, str) for v in groups.values())):
                    raise ValueError("scene groups must be a JSON object mapping source paths to scene ID strings")
                teacher_configs, teacher_ids = _teacher_configs(args)
                report = build_dataset(args.sources, args.output_dir,
                    DatasetOptions(teacher=teacher_configs[0], additional_teachers=tuple(teacher_configs[1:]), teacher_ids=tuple(teacher_ids),
                        group_ids=groups, include_display_teacher=args.include_display_teacher, grouping_semantics=args.grouping,
                        name=args.name, category=args.category, compress_arrays=not args.uncompressed, require_apple_camera=True,
                        validation_fraction=args.validation_fraction, split_seed=args.seed, workers=args.workers,
                        metric_anchor=LearnedDepthConfig(model="depthpro", model_path=args.anchor_model_path,
                            source_dir=args.anchor_source_dir, device=args.device) if args.metric_anchor else None), progress_callback=_progress)
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
                    epochs=args.epochs, steps_per_epoch=args.steps, patch_size=args.patch_size, mode=args.mode,
                    iterations=args.iterations, train_scope=args.scope, device=args.device,
                    raft_root=args.raft_root, raft_model=args.raft_model, raft_model_member=args.raft_model_member,
                    require_photometric_support=args.photometric_support, workers=args.workers,
                ), progress_callback=_progress)
            elif args.command == "cleanup-dataset":
                from .workspace_cleanup import cleanup_dataset
                report = cleanup_dataset(args.dataset, args.workspace, confirm=args.confirm)
            elif args.command == "archive-dataset":
                from .workspace_cleanup import archive_dataset
                report = archive_dataset(args.dataset, args.workspace)
            elif args.command == "cleanup-run":
                from .workspace_cleanup import cleanup_run
                report = cleanup_run(args.checkpoint, args.workspace, confirm=args.confirm)
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
