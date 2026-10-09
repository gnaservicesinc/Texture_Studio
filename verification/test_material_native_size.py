"""Complete training grids use original paths and purge only owned staging."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import material_native_size as native
import material_workbench as workbench
from material_dataset import file_sha256, read_png, resolve_map_path, write_json, write_png
from material_pbrnxt_data import PairCache, select_pairs


def sources(root: Path, size=1024, names=('surface',)):
    for name in names:
        source = root / 'sources' / name
        source.mkdir(parents=True)
        yy, xx = np.mgrid[:size, :size]
        codes = ((xx + yy * 3) % 65535).astype(np.uint16)
        arrays = {'diff': np.stack((codes, codes, codes), -1), 'disp': codes[..., None],
                  'nor_gl': np.repeat(np.full((size,size,1), 32768, np.uint16), 3, -1),
                  'rough': (codes // 2)[..., None]}
        for suffix, array in arrays.items():
            write_png(source / f'{name}_{suffix}_2k.png', array)
    write_json(root / 'dataset.json', {'schema_version':2, 'samples':[]})
    workbench.dataset_info(SimpleNamespace(dataset=root))
    return root


def prepare(root: Path, size=1024, **extras):
    return workbench.prepare_size(SimpleNamespace(dataset=root, size=size,
        expected_index_sha256=None, **extras))


def test_source_index_repair_uses_only_original_paths_and_no_image_copies(tmp_path):
    root = sources(tmp_path)
    index, records = workbench.read_dataset(root)
    assert index['storage_policy'] == native.SOURCE_STORAGE
    assert len(records) == 1
    sample = records[0][2]
    assert sample['sample_id'] == 'surface_1k_full'
    assert all(resolve_map_path(records[0][1].parent, sample, role).is_relative_to(root/'sources') for role in sample['maps'])
    assert not list((root/'samples').rglob('*.png'))
    assert not (root/'.native-sizes').exists()


def test_exact_training_size_references_original_images_without_decode_or_copy(tmp_path, monkeypatch):
    root = sources(tmp_path)
    before = {str(p):file_sha256(p) for p in (root/'sources').rglob('*.png')}
    monkeypatch.setattr(native, 'read_png', lambda *_a, **_k: pytest.fail('Exact originals do not need decoding'))
    first = prepare(root)
    stage = Path(first['dataset_path'])
    assert not list(stage.rglob('*.png')) and not list(stage.rglob('*.tif'))
    assert all(Path(m['path']).is_relative_to(root/'sources') for m in first['materials'][0]['samples'][0]['maps'].values())
    assert first['preparation']['target_resized'] is False
    assert prepare(root)['preparation']['reused']
    assert {str(p):file_sha256(p) for p in (root/'sources').rglob('*.png')} == before
    training, checks, _ = select_pairs(stage, expected_size=1024)
    assert not checks
    _, target = PairCache(0).load(training[0])
    original, _ = read_png(root/'sources/surface/surface_disp_2k.png')
    np.testing.assert_array_equal(target.numpy()[0,0], original[...,0].astype(np.float32)/np.float32(65535))


def test_selected_resolution_crops_center_native_codes_once(tmp_path):
    root = sources(tmp_path, size=2048)
    result = prepare(root, size=1024)
    stage = Path(result['dataset_path'])
    assert result['preparation']['target_resized'] is False
    assert result['preparation']['target_cropped']
    assert len(list(stage.rglob('*.png'))) == 4
    samples = result['materials'][0]['samples']
    assert len(samples) == 1
    assert all(m['width'] == m['height'] == 1024 for m in samples[0]['maps'].values())
    training, _, _ = select_pairs(stage, expected_size=1024)
    sample = training[0]['metadata']
    assert sample['crop_rectangle_top_left_xywh'] == [512,512,1024,1024]
    details = sample['map_metadata']['height']
    assert details['transforms'][0]['algorithm'] == 'exact_native_integer_codes'
    original, _ = read_png(root/'sources/surface/surface_disp_2k.png')
    crop, _ = read_png(training[0]['paths']['height'])
    np.testing.assert_array_equal(crop, original[512:1536,512:1536])
    assert details['transforms'][0]['gamma_applied'] is False
    assert (root/'sources/surface/surface_disp_2k.png').is_file()


def test_mismatched_staging_image_is_recreated_from_original_without_new_duplicate_cache(tmp_path):
    root = sources(tmp_path, size=2048)
    initial = prepare(root)
    output = Path(initial['materials'][0]['samples'][0]['maps']['normal']['path'])
    output.unlink()
    write_png(output, np.zeros((512,512,3),np.uint16))
    repaired = prepare(root)
    assert repaired['dataset_path'] == initial['dataset_path']
    assert all(m['width']==m['height']==1024 for m in repaired['materials'][0]['samples'][0]['maps'].values())
    assert len(list((root/native.STAGING_ROOT).iterdir())) == 1


def test_cleanup_preserves_size_reviews_and_never_removes_originals(tmp_path):
    root = sources(tmp_path)
    result = prepare(root)
    stage = Path(result['dataset_path'])
    workbench.curate(SimpleNamespace(dataset=stage, sample=result['materials'][0]['samples'][0]['sample_id'], status='excluded', split=None,
        note='Bad target detail', expected_index_sha256=result['index_sha256']))
    cleanup = native.cleanup_prepared_dataset(stage)
    assert cleanup['removed'] and cleanup['reviews_preserved']
    assert not stage.exists() and len(list((root/'sources').rglob('*.png'))) == 4
    recovered = prepare(root)
    sample = recovered['materials'][0]['samples'][0]
    assert sample['status']=='excluded' and sample['note']=='Bad target detail'
    assert native.cleanup_prepared_dataset(root)['removed'] is False


def test_automatic_validation_holds_out_complete_materials_at_same_size(tmp_path):
    root = sources(tmp_path, names=('first','second','third'))
    result = prepare(root, automatic_validation=True)
    training, checks, identity = select_pairs(Path(result['dataset_path']), expected_size=1024)
    assert len(training)==2 and len(checks)==1
    assert not ({p['metadata']['material_id'] for p in training} & {p['metadata']['material_id'] for p in checks})
    assert all(p['dimensions']==(1024,1024) for p in training+checks)
    assert 'held-out' in identity['check_scope'].lower()


def test_restore_moved_parent_uses_matching_source_identity(tmp_path):
    root = sources(tmp_path)
    index, records = workbench.read_dataset(root)
    sample = records[0][2]
    for details in sample['map_metadata'].values():
        details['source']['path'] = str(tmp_path/'old-location'/details['source']['filename'])
    write_json(records[0][1], sample)
    result = prepare(root)
    assert all(Path(m['path']).is_relative_to(root/'sources') for m in result['materials'][0]['samples'][0]['maps'].values())


def test_missing_source_only_is_removed_from_list(tmp_path):
    root = sources(tmp_path)
    info = workbench.dataset_info(SimpleNamespace(dataset=root))
    path = Path(info['materials'][0]['samples'][0]['maps']['height']['path'])
    args = SimpleNamespace(dataset=root,sample=info['materials'][0]['samples'][0]['sample_id'],path=path,expected_index_sha256=info['index_sha256'])
    assert workbench.remove_missing(args)['removed'] is False
    path.unlink()
    assert workbench.remove_missing(args)['removed'] is True
    assert workbench.read_dataset(root)[1] == []


def test_missing_prepared_map_with_intact_source_is_not_removed(tmp_path):
    root = sources(tmp_path,size=2048)
    info = prepare(root)
    stage = Path(info['dataset_path'])
    path = Path(info['materials'][0]['samples'][0]['maps']['height']['path'])
    path.unlink()
    result = workbench.remove_missing(SimpleNamespace(dataset=stage,sample=info['materials'][0]['samples'][0]['sample_id'],path=path,
        expected_index_sha256=info['index_sha256']))
    assert result['removed'] is False and result['recoverable'] is True


def test_incomplete_selected_grid_never_upscales_source_detail(tmp_path):
    root = sources(tmp_path)
    with pytest.raises(ValueError,match='original training detail'):
        prepare(root,size=2048)
    assert not list((root/native.STAGING_ROOT).glob('.preparing-*'))


def test_source_reference_cannot_point_elsewhere_without_identity_binding(tmp_path):
    root = sources(tmp_path)
    _index, records = workbench.read_dataset(root)
    sample = records[0][2]
    sample['maps']['height'] = '/tmp/unrelated.png'
    with pytest.raises(ValueError,match='source identity'):
        resolve_map_path(records[0][1].parent,sample,'height')


def test_switching_training_sizes_purges_previous_stage_and_keeps_size_reviews(tmp_path):
    root = sources(tmp_path,size=2048)
    first = prepare(root,size=1024)
    previous = Path(first['dataset_path'])
    workbench.curate(SimpleNamespace(dataset=previous,sample=first['materials'][0]['samples'][0]['sample_id'],status='approved',split=None,
        note='Only reviewed the 1K representation',expected_index_sha256=first['index_sha256']))
    second = prepare(previous,size=2048)
    assert not previous.exists()
    assert len(list((root/native.STAGING_ROOT).iterdir()))==1
    assert not list(Path(second['dataset_path']).rglob('*.png'))
    assert second['materials'][0]['samples'][0]['status']=='unreviewed'
    third = prepare(Path(second['dataset_path']),size=1024)
    assert third['materials'][0]['samples'][0]['status']=='approved'
    assert third['materials'][0]['samples'][0]['note']=='Only reviewed the 1K representation'
    assert not Path(second['dataset_path']).exists()


def test_prepared_maps_expose_original_identity_after_training_stage_purge(tmp_path):
    root = sources(tmp_path,size=2048)
    prepared = prepare(root,size=1024)
    stage = Path(prepared['dataset_path'])
    sample = prepared['materials'][0]['samples'][0]
    original_paths = []
    for role, details in sample['maps'].items():
        original = Path(details['original_source_path'])
        assert original.is_relative_to(root/'sources')
        assert details['original_source_sha256']==file_sha256(original)
        assert details['original_source_width']==details['original_source_height']==2048
        assert details['path']!=str(original)
        assert details['original_normal_convention']==('opengl' if role=='normal' else None)
        original_paths.append(original)
    native.cleanup_prepared_dataset(stage)
    assert all(path.is_file() for path in original_paths)


def test_dataset_reports_original_recovered_after_source_directory_moves(tmp_path):
    root = sources(tmp_path)
    old = root/'sources/surface'
    new = root/'sources/relocated'
    old.rename(new)
    reported = workbench.dataset_info(SimpleNamespace(dataset=root))['materials'][0]['samples'][0]
    for details in reported['maps'].values():
        assert Path(details['original_source_path']).parent==new
        assert details['path']==details['original_source_path']
        assert details['dimensions_verified']


def test_native_crop_policy_uses_one_center_for_4k_and_three_corners_for_8k():
    assert native.crop_layout(2048, 2048, 2048) == [('full', [0, 0, 2048, 2048])]
    assert native.crop_layout(4096, 4096, 2048) == [('center', [1024, 1024, 2048, 2048])]
    assert native.crop_layout(4096, 4096, 1024) == [('center', [1536, 1536, 1024, 1024])]
    corners = native.crop_layout(8192, 8192, 2048)
    assert [rectangle for _region, rectangle in corners] == [
        [0, 0, 2048, 2048], [6144, 0, 2048, 2048], [6144, 6144, 2048, 2048]]
    from material_pbrnxt_data import rectangles_overlap
    assert not any(rectangles_overlap(first, second) for i, (_name, first) in enumerate(corners)
                   for _other, second in corners[i + 1:])


def test_selected_grid_keeps_small_sources_manageable_and_prepares_only_eligible_sources(tmp_path):
    root = sources(tmp_path, size=1024, names=('small',))
    sources(root, size=2048, names=('large',))
    reported = workbench.dataset_info(SimpleNamespace(dataset=root, review_size=2048))
    assert reported['supported_training_sizes'] == [256, 512, 1024, 2048]
    assert [m['material_id'] for m in reported['materials']] == ['large_2k', 'small_1k']
    prepared = prepare(root, size=2048)
    assert [m['material_id'] for m in prepared['materials']] == ['large_2k']
    assert not list(Path(prepared['dataset_path']).rglob('*.png'))


def test_after_stage_cleanup_review_recreates_exact_center_and_own_status(tmp_path):
    root = sources(tmp_path, size=2048)
    prepared = prepare(root, size=1024)
    sample = prepared['materials'][0]['samples'][0]
    workbench.curate(SimpleNamespace(dataset=Path(prepared['dataset_path']), sample=sample['sample_id'],
        status='approved', split=None, note='Review native center pixels', expected_index_sha256=None))
    native.cleanup_prepared_dataset(Path(prepared['dataset_path']))
    reviewed = workbench.dataset_info(SimpleNamespace(dataset=root, review_size=1024))['materials'][0]['samples'][0]
    assert reviewed['sample_id'] == sample['sample_id']
    assert reviewed['status'] == 'approved'
    assert reviewed['crop_rectangle'] == [512, 512, 1024, 1024]
    assert reviewed['width'] == reviewed['height'] == 1024
    assert all(item['crop_rectangle'] == reviewed['crop_rectangle'] for item in reviewed['maps'].values())
    assert all(Path(item['path']).is_relative_to(root/'sources') for item in reviewed['maps'].values())
    assert not (root/'.training-data').exists()


def test_quick_fit_center_cannot_hold_out_its_only_training_region(tmp_path):
    root = sources(tmp_path, size=2048)
    result = prepare(root, size=1024, automatic_validation=True, material='surface_2k')
    training, checks, _identity = select_pairs(Path(result['dataset_path']), expected_size=1024)
    assert len(training) == 1 and checks == []


def test_native_corner_training_and_validation_are_independent_and_preserve_codes(tmp_path, monkeypatch):
    # Exercise preparation with small arrays using the exact same disjoint
    # corner geometry, while the literal 8K policy is checked above.
    root = sources(tmp_path, size=2048)
    real_layout = native.crop_layout
    def three_regions(width, height, size):
        if (width, height, size) == (2048, 2048, 512):
            return [('top_left', [0, 0, 512, 512]), ('top_right', [1536, 0, 512, 512]),
                    ('bottom_right', [1536, 1536, 512, 512])]
        return real_layout(width, height, size)
    monkeypatch.setattr(native, 'crop_layout', three_regions)
    result = prepare(root, size=512, automatic_validation=True)
    stage = Path(result['dataset_path'])
    training, checks, _identity = select_pairs(stage, expected_size=512)
    assert len(training) == 2 and len(checks) == 1
    assert checks[0]['metadata']['source_region_id'] == 'bottom_right'
    assert checks[0]['metadata']['validation_scope'] == 'known_disjoint_regions'
    assert len(list(stage.rglob('*.png'))) == 12
    original, _ = read_png(root/'sources/surface/surface_disp_2k.png')
    for pair in training + checks:
        x, y, width, height = pair['metadata']['crop_rectangle_top_left_xywh']
        actual, _ = read_png(pair['paths']['height'])
        np.testing.assert_array_equal(actual, original[y:y + height, x:x + width])
    selected = training[0]['metadata']['sample_id']
    workbench.curate(SimpleNamespace(dataset=stage, sample=selected, status='excluded', split=None,
        note='One corner artifact', expected_index_sha256=None))
    native.cleanup_prepared_dataset(stage)
    repeated = prepare(root, size=512, automatic_validation=True)
    statuses = {s['sample_id']: s['status'] for s in repeated['materials'][0]['samples']}
    assert statuses[selected] == 'excluded'
    assert all(status == 'unreviewed' for identity, status in statuses.items() if identity != selected)


def test_color_variants_crop_once_share_target_and_inspection_exposes_selected_grid(tmp_path):
    root = sources(tmp_path, size=2048)
    variant = root/'sources/surface/surface_col2_2k.png'
    color = np.full((2048,2048,3), 12000, np.uint16)
    color[...,1] = 38000
    write_png(variant, color)
    prepared = prepare(root, size=1024)
    stage = Path(prepared['dataset_path'])
    sample = prepared['materials'][0]['samples'][0]
    assert len(sample['input_variants']) == 2
    assert {item['variant_id'] for item in sample['input_variants']} == {'color_default', 'color_2'}
    assert all(item['width'] == item['height'] == 1024 for item in sample['input_variants'])
    assert all(item['crop_rectangle'] == [512,512,1024,1024] for item in sample['input_variants'])
    assert len(list(stage.rglob('*.png'))) == 5  # two colors, one of each target
    secondary = next(item for item in sample['input_variants'] if item['variant_id'] == 'color_2')
    values, _ = read_png(secondary['path'])
    np.testing.assert_array_equal(values, color[512:1536,512:1536])
    native.cleanup_prepared_dataset(stage)
    review = workbench.dataset_info(SimpleNamespace(dataset=root, review_size=1024))['materials'][0]['samples'][0]
    assert len(review['input_variants']) == 2
    assert all(Path(item['path']).is_relative_to(root/'sources') for item in review['input_variants'])
    assert all(item['crop_rectangle'] == [512,512,1024,1024] for item in review['input_variants'])
    assert not (root/'.training-data').exists()


def test_interrupted_preparation_purges_only_proven_owned_orphan(tmp_path):
    root = sources(tmp_path)
    staging = root/native.STAGING_ROOT
    owned = staging/'.preparing-owned'
    unrelated = staging/'.preparing-user-folder'
    owned.mkdir(parents=True)
    unrelated.mkdir()
    write_json(owned/native.PREPARING_MARKER, {'schema': native.PREPARATION_SCHEMA,
        'source_dataset_path': str(root.resolve()), 'pid': 2147483647})
    (owned/'partial-map.png').write_bytes(b'partial')
    (unrelated/'keep.txt').write_text('User file')
    prepare(root)
    assert not owned.exists()
    assert (unrelated/'keep.txt').read_text() == 'User file'


def test_resolution_siblings_share_split_even_when_high_source_has_disjoint_corners(tmp_path, monkeypatch):
    root = sources(tmp_path, size=2048, names=('surface', 'other'))
    folder = root/'sources/surface'
    for path in list(folder.glob('*_2k.png')):
        values, _header = read_png(path)
        write_png(folder/path.name.replace('_2k.png', '_1k.png'), np.ascontiguousarray(values[::2,::2]))
    real_layout = native.crop_layout
    def high_source_corners(width, height, size):
        if (width, height, size) == (2048, 2048, 512):
            return [('top_left', [0,0,512,512]), ('top_right', [1536,0,512,512]),
                    ('bottom_right', [1536,1536,512,512])]
        return real_layout(width, height, size)
    monkeypatch.setattr(native, 'crop_layout', high_source_corners)
    result = prepare(root, size=512, automatic_validation=True)
    index = json.loads((Path(result['dataset_path'])/'dataset.json').read_text())
    surface = [item for item in index['samples'] if item['asset_family_id'] == 'surface']
    assert {item['material_id'] for item in surface} == {'surface_1k', 'surface_2k'}
    assert len({item['split'] for item in surface}) == 1
    assert 'surface' not in index['automatic_validation']['regional_family_ids']
    select_pairs(Path(result['dataset_path']), expected_size=512)


def test_missing_alternate_color_removes_only_that_variant_and_blocks_rediscovery(tmp_path):
    root = sources(tmp_path, size=2048)
    variant = root/'sources/surface/surface_col2_2k.png'
    write_png(variant, np.full((2048,2048,3), 45000, np.uint16))
    information = workbench.dataset_info(SimpleNamespace(dataset=root, review_size=1024))
    sample = information['materials'][0]['samples'][0]
    args = SimpleNamespace(dataset=root, review_size=1024, sample=sample['sample_id'], path=variant,
                          expected_index_sha256=information['index_sha256'])
    assert not workbench.remove_missing(args)['removed']
    variant.unlink()
    removed = workbench.remove_missing(args)
    assert removed['removed'] and removed['removed_variant_id'] == 'color_2'
    remaining = workbench.dataset_info(SimpleNamespace(dataset=root, review_size=1024))
    sample = remaining['materials'][0]['samples'][0]
    assert len(sample['input_variants']) == 1 and 'height' in sample['maps']
    prepared = prepare(root, size=1024)
    assert len(prepared['materials'][0]['samples'][0]['input_variants']) == 1
    assert len(list(Path(prepared['dataset_path']).rglob('*.png'))) == 4


def test_validation_selects_only_families_with_the_requested_target(tmp_path):
    root = sources(tmp_path, names=('height_a', 'height_b', 'normal_only'))
    (root/'sources/normal_only/normal_only_disp_2k.png').unlink()
    # Build these as fresh provider sets; the normal-only material never
    # promised a displacement source that must later be restored.
    write_json(root/'dataset.json', {'schema_version':2, 'samples':[]})
    result = prepare(root, size=1024, automatic_validation=True, target='height')
    index = json.loads((Path(result['dataset_path'])/'dataset.json').read_text())
    assert index['automatic_validation']['target'] == 'height'
    assert 'normal_only' not in index['automatic_validation']['source_family_ids']
    training, checks, _ = select_pairs(Path(result['dataset_path']), expected_size=1024, target='height')
    assert training and checks
    assert all('height' in pair['paths'] for pair in training + checks)


def test_quick_fit_stages_only_the_selected_material_and_no_other_changed_maps(tmp_path):
    root = sources(tmp_path, size=2048, names=('first', 'second', 'third'))
    result = prepare(root, size=1024, automatic_validation=True, material='second_2k')
    stage = Path(result['dataset_path'])
    assert [material['material_id'] for material in result['materials']] == ['second_2k']
    assert len(list(stage.rglob('*.png'))) == 4
    training, checks, _identity = select_pairs(stage, expected_size=1024)
    assert len(training) == 1 and not checks
    assert training[0]['metadata']['material_id'] == 'second_2k'
    assert len(list((root/'sources').rglob('*.png'))) == 12


def test_insufficient_disk_and_failed_crop_leave_no_owned_training_junk(tmp_path, monkeypatch):
    root = sources(tmp_path, size=2048)
    monkeypatch.setattr(native.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=1))
    with pytest.raises(native.NativePreparationError) as error:
        prepare(root, size=1024)
    assert error.value.details['problems'][0]['required_bytes'] >= 1024 * 1024 * 16
    assert not (root/native.STAGING_ROOT).exists()
    monkeypatch.setattr(native.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=2**40))
    monkeypatch.setattr(native, 'write_png', lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('No space left on device')))
    with pytest.raises(OSError, match='No space'):
        prepare(root, size=1024)
    assert not (root/native.STAGING_ROOT).exists()
    assert len(list((root/'sources').rglob('*.png'))) == 4
