#!/usr/bin/env python3
"""Check published-resolution crop footprints without changing images or splits."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def footprint(rectangle: list[int], dimensions: list[int]) -> tuple[Fraction, ...]:
    if (len(rectangle) != 4 or len(dimensions) != 2
            or any(type(value) is not int for value in rectangle + dimensions)):
        raise ValueError('Expected integer pixel rectangle and parent dimensions')
    x, y, width, height = rectangle
    parent_width, parent_height = dimensions
    if min(width, height, parent_width, parent_height) <= 0 or min(x, y) < 0 or x + width > parent_width or y + height > parent_height:
        raise ValueError('Crop leaves parent image')
    return (Fraction(x, parent_width), Fraction(y, parent_height),
            Fraction(x + width, parent_width), Fraction(y + height, parent_height))


def overlaps(first: tuple[Fraction, ...], second: tuple[Fraction, ...]) -> bool:
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def candidate_rectangles(dimensions: list[int], crop_size: int) -> list[tuple[str, list[int]]]:
    width, height = dimensions
    if type(crop_size) is not int or crop_size <= 0 or min(width, height) < crop_size:
        raise ValueError('Native parent cannot supply requested crop')
    candidates = [('top_left', [0, 0, crop_size, crop_size]),
                  ('top_right', [width - crop_size, 0, crop_size, crop_size]),
                  ('bottom_left', [0, height - crop_size, crop_size, crop_size]),
                  ('bottom_right', [width - crop_size, height - crop_size, crop_size, crop_size]),
                  ('center', [(width - crop_size) // 2, (height - crop_size) // 2, crop_size, crop_size])]
    seen = set()
    result = []
    for label, rectangle in candidates:
        footprint(rectangle, dimensions)
        if tuple(rectangle) not in seen:
            seen.add(tuple(rectangle))
            result.append((label, rectangle))
    return result


def check_candidates(dimensions: list[int], crop_size: int, heldout: list[dict]) -> list[dict]:
    result = []
    for name, rectangle in candidate_rectangles(dimensions, crop_size):
        region = footprint(rectangle, dimensions)
        blocked = [record['sample_id'] for record in heldout
                   if overlaps(region, footprint(record['rectangle'], record['dimensions']))]
        result.append({'region': name, 'native_rectangle_xywh': rectangle,
                       'normalized_footprint_exact': [str(value) for value in region],
                       'overlapping_validation_sample_ids': blocked,
                       'geometry_disjoint_from_validation': not blocked,
                       'training_assigned': False})
    return result


def contained(root: Path, value: str) -> Path:
    relative = Path(value)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError('Dataset sample path leaves root')
    return resolved


def plan(dataset: Path, parents: Path, crop_size: int) -> dict:
    raw_index = (dataset / 'dataset.json').read_bytes()
    index = json.loads(raw_index)
    if index.get('schema_version') != 2 or index.get('split_strategy') != 'heldout-region-v1':
        raise ValueError('Requires the prepared disjoint-region dataset')
    by_material: dict[str, list[dict]] = {}
    hashes = {}
    for item in index['samples']:
        path = contained(dataset, item['path']) / 'sample.json'
        raw = path.read_bytes()
        record = json.loads(raw)
        if any(record.get(key) != item.get(key) for key in ('sample_id', 'material_id', 'split', 'status')):
            raise ValueError('Index/sample metadata disagree')
        if record.get('status') in ('excluded', 'rejected'):
            continue
        source = record['map_metadata']['height']['source']
        dimensions = [source['width'], source['height']]
        rectangle = record['crop_rectangle_top_left_xywh']
        footprint(rectangle, dimensions)
        by_material.setdefault(record['material_id'], []).append({
            'sample_id': record['sample_id'], 'split': record['split'],
            'rectangle': rectangle, 'dimensions': dimensions})
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    materials = []
    parent_metadata_hashes = {}
    for path in sorted(parents.glob('*/material-source.json')):
        raw_source = path.read_bytes()
        source = json.loads(raw_source)
        identity = source['material_id']
        if identity not in by_material:
            continue
        parent_metadata_hashes[path] = hashlib.sha256(raw_source).hexdigest()
        if (source.get('status') != 'downloaded_or_reused_verified'
                or not source.get('actual_headers_verified')
                or source.get('resized_from_existing_4k') is not False):
            raise ValueError(f'Resolution source is not verified published original: {path}')
        dimensions = source['actual_native_pixel_dimensions']
        for record in by_material[identity]:
            width, height = record['dimensions']
            if width * dimensions[1] != height * dimensions[0]:
                raise ValueError(f'Cross-resolution aspect differs: {identity}')
        heldout = [record for record in by_material[identity] if record['split'] == 'validation']
        candidates = check_candidates(dimensions, crop_size, heldout)
        materials.append({'material_id': identity, 'published_resolution': source['resolution'],
                          'parent_dimensions': dimensions, 'heldout_source_regions': heldout,
                          'resolution_source_metadata_sha256': parent_metadata_hashes[path],
                          'registration_assumption': 'Same asset and proportional full-frame surface coverage; actual registration remains unverified',
                          'registration_verified': source.get('cross_resolution_registration_verified') is True,
                          'geometrically_eligible_candidate_count': sum(item['geometry_disjoint_from_validation'] for item in candidates),
                          'candidates': candidates})
    if hashlib.sha256((dataset / 'dataset.json').read_bytes()).hexdigest() != hashlib.sha256(raw_index).hexdigest():
        raise ValueError('Dataset changed during planning; retry after preparation finishes')
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Sample metadata changed during planning')
    for path, expected in parent_metadata_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Published-resolution metadata changed during planning')
    return {'schema': 'ipde-material-scale-footprint-plan-v1',
            'dataset_index_sha256': hashlib.sha256(raw_index).hexdigest(), 'crop_size': crop_size,
            'source_arrays_read': False, 'source_images_modified': False,
            'active_dataset_modified': False, 'training_assigned': False,
            'scope': 'Exact proportional surface-footprint geometry only; no resampling and no registration proof',
            'materials': materials}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = plan(args.dataset.resolve(), args.parents.resolve(), args.crop_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'materials_checked': len(report['materials']),
                      'eligible_geometry_candidates': sum(item['geometrically_eligible_candidate_count'] for item in report['materials']),
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
