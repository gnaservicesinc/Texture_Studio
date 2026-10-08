"""Real native 1K/2K preservation, curation and separate cache lineage."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import material_native_size as native
import material_workbench as workbench
from material_dataset import GENERATOR, MAP_NAMES, REGION_SPLIT_STRATEGY, heldout_regions, read_png, write_json, write_png
from train_material_height import digest, find_samples

@pytest.fixture(scope='module')
def original_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-original')
    source = root / 'sources' / 'surface'
    source.mkdir(parents=True)
    width = height = 4096
    arrays = {'height': np.arange(width*height, dtype=np.uint16).reshape(height, width, 1),
        'input': np.zeros((height, width, 3), dtype=np.uint8),
        'normal': np.zeros((height, width, 3), dtype=np.uint8),
        'roughness': np.zeros((height, width, 1), dtype=np.uint8)}
    arrays['input'][..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    arrays['input'][..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    arrays['normal'][..., 0] = 128
    arrays['normal'][..., 1] = np.arange(width, dtype=np.uint8)[None, :]
    arrays['normal'][..., 2] = 255
    sources = {}
    for role, array in arrays.items():
        path = source / MAP_NAMES[role]
        write_png(path, array)
        _, header = read_png(path)
        sources[role] = dict(header, path=str(path), filename=path.name,
            suffix='nor_dx' if role == 'normal' else 'disp' if role == 'height' else role)
    for region in heldout_regions(width, height, 1024):
        identity = f"surface_auto_{region['ordinal']:03d}"
        folder = root / 'samples' / identity
        folder.mkdir(parents=True)
        x, y, w, h = region['rectangle']
        sample = {'schema_version': 2, 'generator': GENERATOR, 'sample_id': identity,
            'material_id': 'surface', 'sample_pixel_dimensions': [w, h],
            'source_pixel_dimensions': [width, height], 'source_directory': str(source),
            'crop_rectangle_top_left_xywh': region['rectangle'], 'source_region_name': region['region'],
            'source_region_role': region['split'], 'split': region['split'], 'status': 'approved',
            'split_strategy': REGION_SPLIT_STRATEGY, 'review_status': 'user_approved',
            'normal_convention': 'OpenGL +Y', 'maps': {}, 'map_metadata': {},
            'source_precision_verified': True, 'crop_values_verified': True, 'source_notes': []}
        for role, array in arrays.items():
            crop = array[y:y+h, x:x+w].copy()
            if role == 'normal':
                crop[..., 1] = 255 - crop[..., 1]
            path = folder / MAP_NAMES[role]
            write_png(path, crop)
            sample['maps'][role] = path.name
            sample['map_metadata'][role] = {'source': sources[role], 'filename': path.name,
                'sample_sha256': digest(path), 'sample_bits': int(crop.dtype.itemsize*8),
                'encoding': 'source_srgb_assumed' if role == 'input' else 'linear_data',
                'transforms': [{'type': 'directx_to_opengl', 'component': 'G', 'operation': 'max_integer_code - G', 'reversible': True}] if role == 'normal' else []}
        write_json(folder / 'sample.json', sample)
    entries = []
    for path in sorted((root / 'samples').glob('*/sample.json')):
        sample = json.loads(path.read_text())
        entries.append({key: sample[key] for key in ('sample_id', 'material_id', 'status', 'split')} | {'path': str(path.parent.relative_to(root))})
    write_json(root / 'dataset.json', {'schema_version': 2, 'generator': GENERATOR,
        'crop_size': 1024, 'split_strategy': REGION_SPLIT_STRATEGY,
        'validation_scope': 'unseen_regions_of_known_materials', 'samples': entries})
    return root

@pytest.fixture
def dataset(tmp_path, original_fixture):
    root = tmp_path / 'dataset'
    shutil.copytree(original_fixture, root)
    for path in root.glob('samples/*/sample.json'):
        sample = json.loads(path.read_text())
        for item in sample['map_metadata'].values():
            item['source']['path'] = item['source']['path'].replace(str(original_fixture), str(root))
        sample['source_directory'] = sample['source_directory'].replace(str(original_fixture), str(root))
        write_json(path, sample)
    return root

def prepare(dataset, size=2048, expected=None):
    return workbench.prepare_size(SimpleNamespace(dataset=dataset, size=size, expected_index_sha256=expected))

def test_real_2k_crops_preserve_raw_16bit_codes_and_reversible_normal(dataset):
    original = {str(p.relative_to(dataset)): digest(p) for p in dataset.rglob('*') if p.is_file()}
    result = prepare(dataset / 'dataset.json', expected=digest(dataset / 'dataset.json'))
    derived = Path(result['dataset_path'])
    assert derived != dataset and result['preparation']['split_lineage_changed']
    assert result['preparation']['target_resized'] is False
    assert len(result['materials'][0]['samples']) == 3
    for sample in find_samples(derived, allow_unreviewed=True):
        record = sample['metadata']
        x, y, w, h = record['crop_rectangle_top_left_xywh']
        assert [w, h] == [2048, 2048]
        for role, filename in record['maps'].items():
            parent, _ = read_png(record['map_metadata'][role]['source']['path'])
            crop, _ = read_png(sample['metadata_path'].parent / filename)
            expected = parent[y:y+h, x:x+w].copy()
            if role == 'normal':
                expected[..., 1] = 255 - expected[..., 1]
            np.testing.assert_array_equal(crop, expected)
            assert crop.dtype == expected.dtype
        codes, _ = read_png(sample['metadata_path'].parent / record['maps']['height'])
        assert int(codes[0, 1, 0]) - int(codes[0, 0, 0]) == 1
    assert original == {name: digest(dataset / name) for name in original}
    assert not result['preparation']['original_dataset_modified']
    assert 'Earlier training' in result['preparation']['cross_size_validation_notice']

def test_cached_reuse_keeps_own_review_and_switch_back_uses_original(dataset, monkeypatch):
    first = prepare(dataset)
    derived = Path(first['dataset_path'])
    sample_id = first['materials'][0]['samples'][0]['sample_id']
    monkeypatch.setattr(workbench, 'training_active', lambda _p: False)
    workbench.curate(SimpleNamespace(dataset=derived, sample=sample_id, status='excluded', split=None,
        note='Keep this derived crop excluded', expected_index_sha256=first['index_sha256']))
    reused = prepare(dataset)
    assert reused['preparation']['reused'] and reused['dataset_path'] == str(derived)
    chosen = next(s for s in reused['materials'][0]['samples'] if s['sample_id'] == sample_id)
    assert chosen['status'] == 'excluded' and chosen['note'] == 'Keep this derived crop excluded'
    back = prepare(derived / 'dataset.json', 1024, reused['index_sha256'])
    assert back['dataset_path'] == str(dataset) and back['preparation']['reused']
    assert back['preparation']['source_dataset_path'] == str(dataset)

def test_excluded_region_and_notes_inherit_without_reapproving_unreviewed(dataset):
    sample_path = dataset / 'samples/surface_auto_003/sample.json'
    sample = json.loads(sample_path.read_text())
    sample.update(status='excluded', curation_note='Original center has bad source data')
    write_json(sample_path, sample)
    index = json.loads((dataset / 'dataset.json').read_text())
    next(s for s in index['samples'] if s['sample_id'] == sample['sample_id'])['status'] = 'excluded'
    write_json(dataset / 'dataset.json', index)
    result = prepare(dataset)
    assert all(s['status'] == 'excluded' for s in result['materials'][0]['samples'])
    for entry in result['materials'][0]['samples']:
        record = json.loads(Path(entry['metadata_path']).read_text())
        assert 'bad source data' in record['curation_note']
        assert any(p['status'] == 'excluded' and p['split'] == 'validation' for p in record['prior_crop_decisions'])

@pytest.mark.parametrize('problem', ['missing', 'changed'])
def test_unavailable_or_changed_parent_is_structured_and_original_untouched(dataset, problem):
    path = dataset / 'sources/surface/displacement.png'
    if problem == 'missing':
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b'changed')
    before = digest(dataset / 'dataset.json')
    with pytest.raises(native.NativePreparationError) as raised:
        prepare(dataset)
    expected = 'source_missing' if problem == 'missing' else 'source_changed'
    assert any(p['code'] == expected and p['material_id'] == 'surface' and p['role'] == 'height' and p['action'] for p in raised.value.details['problems'])
    assert digest(dataset / 'dataset.json') == before and not (dataset / '.native-sizes').exists()

def test_stale_selection_and_corrupt_cache_repair_never_overwrites(dataset):
    with pytest.raises(ValueError, match='changed since'):
        prepare(dataset, expected='0' * 64)
    result = prepare(dataset)
    path = Path(result['materials'][0]['samples'][0]['maps']['height']['path'])
    path.write_bytes(path.read_bytes() + b'tampered')
    original_tampered = path.read_bytes()
    repaired = prepare(dataset)
    assert repaired['dataset_path'] != result['dataset_path']
    assert repaired['preparation']['preserved_invalid_caches'][0]['path'] == result['dataset_path']
    assert prepare(dataset)['dataset_path'] == repaired['dataset_path']
    assert path.read_bytes() == original_tampered

def test_source_review_change_creates_new_cache_preserves_existing(dataset):
    first = prepare(dataset)
    path = dataset / 'samples/surface_auto_001/sample.json'
    sample = json.loads(path.read_text())
    sample['curation_note'] = 'New review, version the derived cache'
    write_json(path, sample)
    second = prepare(dataset)
    assert second['dataset_path'] != first['dataset_path'] and not second['preparation']['reused']
    assert Path(first['dataset_path']).is_dir()
    assert all('New review' in s['note'] for s in second['materials'][0]['samples'])

def test_same_size_without_parents_is_valid_and_no_cache_allocated(dataset, monkeypatch):
    monkeypatch.setattr(native, 'problems_for_sources', lambda *_args: pytest.fail('No crop preparation is needed at the current size'))
    result = prepare(dataset, 1024)
    assert result['dataset_path'] == str(dataset) and result['preparation']['reused']
    assert not (dataset / '.native-sizes').exists()

def test_valid_cache_reopens_without_original_parents(dataset):
    first = prepare(dataset)
    shutil.rmtree(dataset / 'sources')
    second = prepare(dataset)
    assert second['dataset_path'] == first['dataset_path']
    assert second['preparation']['reused'] and not second['preparation']['original_parents_required']

def test_new_larger_pixels_do_not_inherit_unrelated_crop_approval(dataset):
    prepared = prepare(dataset)
    assert all(s['status'] == 'unreviewed' for s in prepared['materials'][0]['samples'])
    assert all(p['status'] == 'approved' for p in json.loads(Path(prepared['materials'][0]['samples'][0]['metadata_path']).read_text())['prior_crop_decisions'])

def test_cancelled_preparation_cleans_stage_and_preserves_original(dataset, monkeypatch):
    before = digest(dataset / 'dataset.json')
    actual_write = native.write_png
    def interrupted(*arguments, **keywords):
        actual_write(*arguments, **keywords)
        raise KeyboardInterrupt('Stop preparation')
    monkeypatch.setattr(native, 'write_png', interrupted)
    with pytest.raises(KeyboardInterrupt, match='Stop preparation'):
        prepare(dataset)
    assert digest(dataset / 'dataset.json') == before
    assert list((dataset / '.native-sizes').iterdir()) == []

def test_external_metadata_change_prevents_cache_publication(dataset, monkeypatch):
    actual_write = native.write_png
    changed = []
    def change_review(*arguments, **keywords):
        actual_write(*arguments, **keywords)
        if not changed:
            metadata = dataset / 'samples/surface_auto_001/sample.json'
            metadata.write_bytes(metadata.read_bytes() + b' ')
            changed.append(True)
    monkeypatch.setattr(native, 'write_png', change_review)
    with pytest.raises(ValueError, match='metadata changed'):
        prepare(dataset)
    assert list((dataset / '.native-sizes').iterdir()) == []

def test_2_to_1_native_ratio_keeps_disjoint_train_and_validation_geometry():
    regions = heldout_regions(4096, 2048, 2048)
    assert len(regions) == 2 and {r['split'] for r in regions} == {'train', 'validation'}
    assert not native.rectangles_overlap(regions[0]['rectangle'], regions[1]['rectangle'])

def test_cli_error_returns_actionable_source_details(dataset):
    (dataset / 'sources/surface/displacement.png').unlink()
    process = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/material_workbench.py'), 'prepare-size', '--dataset', str(dataset / 'dataset.json'), '--size', '2048'], text=True, capture_output=True)
    result = json.loads(process.stdout)
    assert process.returncode == 1 and not result['ok']
    assert any(p['code'] == 'source_missing' for p in result['error']['details']['problems'])

@pytest.mark.parametrize('invalid', ['../outside', '/absolute/output', 'nested/name'])
def test_noncanonical_material_identity_cannot_escape_staging(dataset, invalid):
    index = json.loads((dataset / 'dataset.json').read_text())
    for entry in index['samples']:
        path = dataset / entry['path'] / 'sample.json'
        record = json.loads(path.read_text())
        record['material_id'] = invalid
        entry['material_id'] = invalid
        write_json(path, record)
    write_json(dataset / 'dataset.json', index)
    with pytest.raises(ValueError, match='canonical'):
        prepare(dataset)
    assert not (dataset / '.native-sizes').exists()

def test_repair_preserves_exact_derived_exclusion_and_note(dataset, monkeypatch):
    result = prepare(dataset)
    chosen = result['materials'][0]['samples'][0]
    monkeypatch.setattr(workbench, 'training_active', lambda _path: False)
    workbench.curate(SimpleNamespace(dataset=Path(result['dataset_path']), sample=chosen['sample_id'], status='excluded', split=None,
        note='Keep excluded even if cache image needs repair', expected_index_sha256=result['index_sha256']))
    path = Path(chosen['maps']['height']['path'])
    path.write_bytes(path.read_bytes() + b'corrupt')
    repaired = prepare(dataset)
    record = next(s for s in repaired['materials'][0]['samples'] if s['sample_id'] == chosen['sample_id'])
    assert record['status'] == 'excluded' and 'even if' in record['note']
    assert path.read_bytes().endswith(b'corrupt')

def test_dangling_native_cache_symlink_is_preserved_and_refused(dataset):
    index, records = workbench.read_dataset(dataset)
    proof = native.snapshot(dataset, records)
    import hashlib
    key = hashlib.sha256(json.dumps({'schema': native.PREPARATION_SCHEMA, 'size': 2048, 'snapshot': proof}, sort_keys=True).encode()).hexdigest()
    cache = dataset / '.native-sizes'
    cache.mkdir()
    link = cache / ('2048-' + key[:20])
    link.symlink_to(dataset / 'missing-destination', target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic link'):
        prepare(dataset)
    assert link.is_symlink()

def test_concurrent_review_during_repair_is_not_lost(dataset, monkeypatch):
    result = prepare(dataset)
    chosen = result['materials'][0]['samples'][0]
    old_cache = Path(result['dataset_path'])
    path = Path(chosen['maps']['height']['path'])
    path.write_bytes(path.read_bytes() + b'corrupt')
    actual_write = native.write_png
    changed = []
    monkeypatch.setattr(workbench, 'training_active', lambda _path: False)
    def review_during_crop(*arguments, **keywords):
        actual_write(*arguments, **keywords)
        if not changed:
            workbench.curate(SimpleNamespace(dataset=old_cache, sample=chosen['sample_id'], status='excluded', split=None,
                note='Concurrent exclusion must survive', expected_index_sha256=result['index_sha256']))
            changed.append(True)
    monkeypatch.setattr(native, 'write_png', review_during_crop)
    with pytest.raises(ValueError, match='metadata changed'):
        prepare(dataset)
    assert list((dataset / '.native-sizes').glob('*repair*')) == []
    assert workbench.dataset_info(SimpleNamespace(dataset=old_cache))['materials'][0]['samples'][0]['status'] == 'excluded'
