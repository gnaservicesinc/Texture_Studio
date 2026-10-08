#!/usr/bin/env python3
"""Make a display-only contact sheet of held-out native material height detail."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ADAPTATION_SCHEMA = 'texture-studio-four-material-adaptation-diagnostic-v1'


def adaptation_summary(run: Path) -> dict:
    """Require the actual matched comparison before making display artifacts."""
    summary = json.loads((run / 'summary.json').read_text())
    if summary.get('schema') != ADAPTATION_SCHEMA or summary.get('matched_complete_schedules') is not True:
        raise ValueError('Expected a completed, matched material-adaptation comparison')
    variants = summary.get('variants', {})
    if set(variants) != {'frozen', 'lora'}:
        raise ValueError('Adaptation comparison needs exactly frozen and lora variants')
    materials = summary.get('materials', [])
    updates = summary.get('requested_updates_per_crop')
    if len(materials) != 4 or len(set(materials)) != 4 or not isinstance(updates, int) or updates < 2:
        raise ValueError('Expected four materials with declared equal crop updates')
    hashes = set()
    for name, variant in variants.items():
        if variant.get('complete') is not True or variant.get('status') != 'complete':
            raise ValueError('Cannot compare incomplete adaptation schedules or exports')
        if variant.get('completed_steps') != 4 * updates or variant.get('per_crop_update_counts') != [updates] * 4:
            raise ValueError('Adaptation crop update counts are not matched')
        identity = variant.get('initial_head_state_sha256')
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError('Expected checked shared initial head hashes')
        hashes.add(identity)
        phases = variant.get('phase_history', [])
        if not phases or phases[0].get('step') != 0 or phases[-1].get('step') != 4 * updates:
            raise ValueError('Expected initial and final complete adaptation evaluations')
        selected = variant.get('selected_step')
        if selected not in [phase['step'] for phase in phases]:
            raise ValueError('Selected checkpoint must have a complete recorded evaluation')
        if name == 'lora' and variant.get('zero_adapter_initial_features_and_predictions_identical') is not True:
            raise ValueError('Missing zero-adapter shared initialization proof')
    if len(hashes) != 1 or summary.get('initial_shared_head_state_sha256') not in hashes:
        raise ValueError('Adaptation variants do not share the checked initial head')
    return summary


def adaptation_sheet(run: Path) -> Image.Image:
    summary = adaptation_summary(run)
    variants = ('frozen', 'lora')
    sheet = Image.new('RGB', (1536, 717), (24, 26, 29))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), 'Stucco | matched four-material comparison; frozen DINOv2 versus rank-8 LoRA', fill='white')
    draw.text((12, 32), 'FINAL weights shown. Selected known-region checkpoints: ' +
              '; '.join(f"{name} step {summary['variants'][name]['selected_step']}" for name in variants), fill=(225, 210, 180))
    draw.text((12, 52), 'Display only: mean centered, fixed shared gains. Native 256px center details at right; exports unchanged.', fill=(190, 195, 200))
    titles = ['Reference', 'Frozen final', 'LoRA final']
    for row, label in enumerate(('final-training-fit', 'final-known-material-disjoint-region')):
        directories = [run / name / label for name in variants]
        references = [np.load(directory / 'target.height.float32.npy', allow_pickle=False) for directory in directories]
        predictions = [np.load(directory / f'{label}.height.float32.npy', allow_pickle=False) for directory in directories]
        reference = references[0]
        arrays = [*references, *predictions]
        if reference.ndim != 2 or min(reference.shape) < 256 or any(array.dtype != np.float32 or array.shape != reference.shape or not np.isfinite(array).all() for array in arrays):
            raise ValueError('Expected matching finite native Float32 height arrays, at least 256 pixels per side')
        if not np.array_equal(reference, references[1]):
            raise ValueError('Frozen and LoRA previews must use identical native reference targets')
        identities = []
        for name, directory in zip(variants, directories):
            metadata = json.loads((directory / f'{label}.prediction.json').read_text())
            if metadata.get('schema') != ADAPTATION_SCHEMA or metadata.get('variant') != name or not metadata.get('snapshot', '').startswith('final weights'):
                raise ValueError('Expected explicitly labeled final adaptation exports')
            identities.append(metadata.get('sample_id'))
        expected_id = summary['materials'][0] + ('_auto_001' if row == 0 else '_auto_003')
        if identities != [expected_id, expected_id]:
            raise ValueError('Preview exports must refer to the same declared stucco crop')
        top = 80 + row * 313
        draw.text((12, top), 'Repeated training crop' if row == 0 else 'Disjoint region of the same known material; no fresh-material claim', fill='white')
        for column, (array, detail, title) in enumerate([(array, detail, title) for detail in (False, True)
                for array, title in zip([reference, *predictions], titles)]):
            draw.text((column * 256 + 8, top + 20), title + (' detail ×4' if detail else ' ×2'), fill='white')
            sheet.paste(visual_height(array, 4 if detail else 2, detail), (column * 256, top + 43))
    return sheet


def visual_height(values: np.ndarray, gain: float, detail: bool) -> Image.Image:
    # The same display gain is used for reference and prediction. Centering here
    # is confined to this labeled preview and never changes exported arrays.
    centered = values - values.mean()
    if detail:
        y, x = (np.array(values.shape) - 256) // 2
        centered = centered[y:y + 256, x:x + 256]
    display = np.rint(np.clip(.5 + gain * centered, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(display).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)


def repeated_fit_sheet(run: Path) -> Image.Image:
    """Compare objectives with one fixed display scale, without changing outputs."""
    summary = json.loads((run / 'summary.json').read_text())
    if summary.get('schema') != 'texture-studio-repeated-crop-fit-v1':
        raise ValueError('Expected repeated-crop diagnostic summary')
    variants = ['current_l1', 'relative_squared']
    reference = np.load(run / variants[0] / 'target.height.float32.npy', allow_pickle=False)
    predictions = [np.load(run / name / 'final-training-fit.height.float32.npy', allow_pickle=False)
                   for name in variants]
    if reference.ndim != 2 or min(reference.shape) < 256 or any(p.shape != reference.shape for p in predictions):
        raise ValueError('Expected matching native height arrays, at least 256 pixels per side')
    labels = ['Reference height ×2', 'L1 fit ×2', 'Squared fit ×2',
              'Reference detail ×4', 'L1 detail ×4', 'Squared detail ×4']
    sheet = Image.new('RGB', (6 * 256, 350), (24, 26, 29))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{summary['material_id']} | repeated training crop, {summary['requested_steps_per_variant']} steps per objective", fill='white')
    draw.text((12, 32), 'Display only: mean centered, fixed shared gains. Native 256px center details at right.', fill=(190, 195, 200))
    for column, label in enumerate(labels):
        draw.text((column * 256 + 8, 60), label, fill='white')
    for column, (array, detail) in enumerate([(a, d) for d in (False, True) for a in [reference, *predictions]]):
        sheet.paste(visual_height(array, 4 if detail else 2, detail), (column * 256, 85))
    return sheet


def frozen_feature_sheet(run: Path) -> Image.Image:
    summary = json.loads((run / 'summary.json').read_text())
    if not summary.get('matched_completed_steps') or set(summary.get('variants', {})) != {'zero_features', 'frozen_features'}:
        raise ValueError('Expected completed, matched frozen-feature comparison')
    sheet = Image.new('RGB', (1536, 685), (24, 26, 29))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{summary['material_id']} | identical heads/updates; zero features versus frozen DINOv2", fill='white')
    draw.text((12, 32), 'Display only: mean centered, shared fixed gains. Native 256px center details at right.', fill=(190, 195, 200))
    labels = ['Reference ×2', 'Zero features ×2', 'Frozen features ×2',
              'Reference detail ×4', 'Zero features detail ×4', 'Frozen features detail ×4']
    for row, label in enumerate(['final-training-fit', 'final-known-material-disjoint-region']):
        directories = [run / name / label for name in ['zero_features', 'frozen_features']]
        reference = np.load(directories[0] / 'target.height.float32.npy', allow_pickle=False)
        predictions = [np.load(d / f'{label}.height.float32.npy', allow_pickle=False) for d in directories]
        if reference.ndim != 2 or min(reference.shape) < 256 or any(p.shape != reference.shape for p in predictions):
            raise ValueError('Expected matching native height arrays, at least 256 pixels per side')
        top = 58 + row * 313
        draw.text((12, top), 'Repeated training crop' if row == 0 else 'Separate region of same known material', fill='white')
        for column, title in enumerate(labels):
            draw.text((column * 256 + 8, top + 20), title, fill='white')
        for column, (array, detail) in enumerate([(a, d) for d in (False, True) for a in [reference, *predictions]]):
            sheet.paste(visual_height(array, 4 if detail else 2, detail), (column * 256, top + 43))
    return sheet


def four_material_sheet(run: Path) -> Image.Image:
    summary = json.loads((run / 'summary.json').read_text())
    if not summary.get('matched_complete_12_channel_schedules'):
        raise ValueError('Expected completed, matched four-material comparison')
    variants = list(summary['variants'])
    if any(not summary['variants'][name]['complete'] for name in variants):
        raise ValueError('Incomplete fitting schedule')
    columns = 2 * (len(variants) + 1)
    sheet = Image.new('RGB', (columns * 256, 685), (24, 26, 29))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), 'Stucco | four-material fitting; equal final per-crop update counts', fill='white')
    draw.text((12, 32), 'Display only: mean centered, fixed shared gains. Native 256px center details at right.', fill=(190, 195, 200))
    for row, label in enumerate(['final-training-fit', 'final-known-material-disjoint-region']):
        directories = [run / name / label for name in variants]
        reference = np.load(directories[0] / 'target.height.float32.npy', allow_pickle=False)
        predictions = [np.load(d / f'{label}.height.float32.npy', allow_pickle=False) for d in directories]
        if reference.ndim != 2 or min(reference.shape) < 256 or any(p.shape != reference.shape for p in predictions):
            raise ValueError('Expected matching native height arrays')
        top = 58 + row * 313
        draw.text((12, top), 'Repeated training crop' if row == 0 else 'Separate region of same known material', fill='white')
        titles = ['Reference', *variants]
        for column, (array, detail, title) in enumerate([(a, d, t) for d in (False, True)
                                                        for a, t in zip([reference, *predictions], titles)]):
            draw.text((column * 256 + 8, top + 20), title + (' detail ×4' if detail else ' ×2'), fill='white')
            sheet.paste(visual_height(array, 4 if detail else 2, detail), (column * 256, top + 43))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='best')
    parser.add_argument('--max-rows', type=int, default=8)
    parser.add_argument('--repeated-fit', action='store_true', help='Compare the two repeated-crop objective diagnostics')
    parser.add_argument('--frozen-dino', action='store_true', help='Compare zero/frozen features on training and disjoint regions')
    parser.add_argument('--four-material', action='store_true', help='Compare completed mixed/replay native fitting schedules')
    parser.add_argument('--adaptation', action='store_true', help='Compare completed frozen/LoRA final weights with shared display gains')
    args = parser.parse_args()
    if sum((args.repeated_fit, args.frozen_dino, args.four_material, args.adaptation)) > 1:
        parser.error('Choose one diagnostic format')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeated_fit or args.frozen_dino or args.four_material or args.adaptation:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        renderer = repeated_fit_sheet if args.repeated_fit else frozen_feature_sheet if args.frozen_dino else four_material_sheet if args.four_material else adaptation_sheet
        renderer(args.run).save(args.output)
        print(args.output)
        return
    folders = sorted(folder for folder in (args.run / 'predictions').iterdir()
                     if folder.is_dir() and (folder / f'{args.label}.height.float32.npy').is_file())[:args.max_rows]
    if not folders:
        raise ValueError('No held-out predictions found')
    labels = ['Source diffuse', 'Reference height ×2', 'Model height ×2', 'Reference native detail ×4', 'Model native detail ×4']
    sheet = Image.new('RGB', (5 * 256, 85 + len(folders) * 294), (24, 26, 29))
    draw = ImageDraw.Draw(sheet)
    evaluation = args.run / f'evaluation.{args.label}.json'
    scope = json.loads(evaluation.read_text()).get('validation_scope', 'Unseen materials') if evaluation.exists() else 'Predictions'
    draw.text((12, 12), f'{scope} | full maps at left; center 256px crops at right', fill='white')
    draw.text((12, 31), 'Height previews only: mean centered, fixed shared gain. Float32 exports remain unchanged.', fill=(190, 195, 200))
    for i, label in enumerate(labels):
        draw.text((i * 256 + 10, 60), label, fill='white')
    for row, folder in enumerate(folders):
        reference = np.load(folder / 'target.height.float32.npy', allow_pickle=False)
        prediction = np.load(folder / f'{args.label}.height.float32.npy', allow_pickle=False)
        if reference.shape != prediction.shape or reference.ndim != 2 or min(reference.shape) < 256:
            raise ValueError('Expected matching native reference and prediction, at least 256 pixels per side')
        images = [Image.open(folder / 'input.preview.png').convert('RGB').resize((256, 256), Image.Resampling.LANCZOS),
                  visual_height(reference, 2, False), visual_height(prediction, 2, False),
                  visual_height(reference, 4, True), visual_height(prediction, 4, True)]
        top = 85 + row * 294
        for column, image in enumerate(images):
            sheet.paste(image, (column * 256, top))
        draw.text((10, top + 265), folder.name, fill='white')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    sheet.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
