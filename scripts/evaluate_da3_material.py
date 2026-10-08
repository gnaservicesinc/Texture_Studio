#!/usr/bin/env python3
"""Evaluate pinned DA3 scene depth against held-out material height crops.

Raw Float32 model depth is retained unchanged. The affine-aligned comparison
uses held-out ground truth to estimate scale/offset and is explicitly an
optimistic diagnostic, never an exportable material-height prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    result = {'height_mae': float(np.abs(prediction - target).mean()),
              'height_rmse': float(np.sqrt(np.square(prediction - target).mean()))}
    for stride in (1, 4, 16):
        errors = []
        magnitudes = []
        for axis in (0, 1):
            p = np.diff(prediction[::stride, ::stride], axis=axis)
            t = np.diff(target[::stride, ::stride], axis=axis)
            errors.append(float(np.abs(p - t).mean()))
            magnitudes.append(float(np.abs(t).mean()))
        result[f'gradient_mae_stride_{stride}'] = sum(errors) / 2
        result[f'target_gradient_magnitude_stride_{stride}'] = sum(magnitudes) / 2
    return result


def optimistic_affine(raw_depth: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    # DA3 camera-Z increases away from the camera; material height increases out.
    feature = -raw_depth.astype(np.float64)
    centered = feature - feature.mean()
    target64 = target.astype(np.float64)
    variance = float(np.square(centered).mean())
    slope = max(0.0, float((centered * (target64 - target64.mean())).mean()) / variance) if variance > 1e-20 else 0.0
    intercept = float(target64.mean() - slope * feature.mean())
    return (slope * feature + intercept).astype(np.float32), {
        'inverse_camera_z_slope': slope, 'offset': intercept,
        'uses_heldout_ground_truth_for_alignment': True,
        'interpretation': 'Optimistic scale/offset alignment for diagnosis only; not usable for new-photo export.'}


def read_numeric(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None or array.dtype not in (np.uint8, np.uint16):
        raise ValueError(f'Unsupported raw PNG samples: {path}')
    return array


def opaque_components(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 4:
        if not np.all(array[:, :, 3] == np.iinfo(array.dtype).max):
            raise ValueError('Non-opaque alpha needs an explicit training mask')
        return array[:, :, :3]
    return array


def scalar_codes(array: np.ndarray) -> np.ndarray:
    array = opaque_components(array)
    if array.ndim == 3 and array.shape[2] == 3:
        if not np.array_equal(array[:, :, 0], array[:, :, 1]) or not np.array_equal(array[:, :, 0], array[:, :, 2]):
            raise ValueError('RGB height components disagree; no luminance conversion is permitted')
        return array[:, :, 0]
    if array.ndim != 2:
        raise ValueError('Expected scalar numeric height')
    return array


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Metal-enabled DA3 Python interpreter')
    parser.add_argument('--model', type=Path, required=True, help='Exact pinned GIANT 1.1 checkpoint directory')
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates every held-out crop')
    parser.add_argument('--resolution', type=int, choices=(1036, 1540, 2044), default=1036)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest_path = root / 'dataset.json'
    manifest_sha256 = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    selected = [sample for sample in manifest['samples'] if sample['split'] in ('validation', 'val')]
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        parser.error('No held-out material crops in dataset.json')
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error('Evaluation output must be outside the input dataset')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'dataset.snapshot.json').write_bytes(manifest_path.read_bytes())
    worker = Path(__file__).resolve().parents[1] / 'native/TextureStudio/Resources/DA3Backend/worker.py'
    records = []
    for index, record in enumerate(selected, 1):
        location = (root / record['path']).resolve()
        sample_path = location if location.is_file() else location / 'sample.json'
        if root not in sample_path.parents:
            raise ValueError('Sample escapes dataset directory')
        sample = json.loads(sample_path.read_text())
        folder = sample_path.parent
        image_path = (folder / sample['maps']['input']).resolve()
        height_path = (folder / sample['maps']['height']).resolve()
        if folder not in image_path.parents or folder not in height_path.parents:
            raise ValueError('Map escapes sample directory')
        source = opaque_components(read_numeric(image_path))
        target_codes = scalar_codes(read_numeric(height_path))
        if source.ndim != 3 or source.shape[2] != 3 or target_codes.ndim != 2:
            raise ValueError('Expected RGB diffuse and scalar height')
        target = target_codes.astype(np.float32) / np.iinfo(target_codes.dtype).max
        case = output / record['sample_id']
        case.mkdir()
        # Only the model-input derivative is 8-bit. Original training PNGs and
        # targets are neither changed nor interpreted through a display gamma.
        input8 = np.rint(source.astype(np.float64) * (255 / np.iinfo(source.dtype).max)).astype(np.uint8)
        model_input = case / 'model-input-8bit.png'
        if not cv2.imwrite(str(model_input), input8):
            raise OSError('Could not write derived model input')
        started = time.monotonic()
        print(json.dumps({'sample': record['sample_id'], 'index': index, 'total': len(selected)}), flush=True)
        subprocess.run([str(args.python), '-I', '-B', str(worker), '--model', str(args.model),
                        '--image', str(model_input), '--output', str(case / 'raw'),
                        '--resolution', str(args.resolution)], check=True, timeout=1800)
        metadata = json.loads((case / 'raw/metadata.json').read_text())
        raw = np.fromfile(case / 'raw/depth.f32', dtype='<f4').reshape(metadata['height'], metadata['width'])
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite DA3 output')
        resized = cv2.resize(raw, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        aligned, alignment = optimistic_affine(resized, target)
        np.save(case / 'optimistic-aligned-height.npy', aligned, allow_pickle=False)
        comparison = {'sample_id': record['sample_id'], 'material_id': record['material_id'],
                      'elapsed_seconds': time.monotonic() - started,
                      'source_input_sha256': digest(image_path), 'source_height_sha256': digest(height_path),
                      'raw_model_metadata': metadata, 'alignment': alignment,
                      'aligned_da3': metrics(aligned, target),
                      'oracle_constant': metrics(np.full_like(target, target.mean()), target),
                      'alignment_grid': 'bilinear derived copy at native training crop size'}
        (case / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        records.append(comparison)
        (output / 'report.json').write_text(json.dumps({
            'schema': 'texture-material-da3-diagnostic-v1', 'dataset_manifest_sha256': manifest_sha256,
            'warning': 'Scale/offset uses validation targets; these aligned predictions are diagnostics, not production outputs.',
            'completed_samples': len(records), 'total_samples': len(selected), 'samples': records}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
