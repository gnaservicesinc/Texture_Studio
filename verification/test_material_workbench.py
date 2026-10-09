"""Current dataset bridge: exact source references and durable curation JSON."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'verification'))
import material_workbench as workbench
from material_dataset import file_sha256, write_png
from test_material_native_size import sources


def args(**values):
    return SimpleNamespace(**values)


def test_dataset_lists_complete_original_maps_and_supports_source_folder(tmp_path):
    root = sources(tmp_path)
    result = workbench.dataset_info(args(dataset=root/'sources'))
    assert result['dataset_path']==str(root.resolve())
    assert result['supported_training_sizes']==[256,512,1024]
    material = result['materials'][0]
    assert len(material['samples'])==1
    sample = material['samples'][0]
    assert sample['width']==sample['height']==1024
    assert all(Path(item['path']).is_relative_to(root/'sources') for item in sample['maps'].values())
    assert all(item['sha256']==file_sha256(Path(item['path'])) for item in sample['maps'].values())


def test_dataset_reports_actual_bad_dimensions_without_losing_source_binding(tmp_path):
    root=sources(tmp_path)
    chosen=workbench.dataset_info(args(dataset=root))['materials'][0]['samples'][0]
    height=Path(chosen['maps']['height']['path'])
    height.unlink()
    write_png(height,np.zeros((512,512,1),np.uint16))
    reported=workbench.dataset_info(args(dataset=root))['materials'][0]['samples'][0]
    assert reported['width']==reported['height']==1024
    assert reported['maps']['height']['width']==512
    height.unlink()
    reported=workbench.dataset_info(args(dataset=root))['materials'][0]['samples'][0]
    assert not reported['maps']['height']['dimensions_verified']
    assert reported['maps']['height']['dimension_issue']


def test_curation_changes_metadata_only_and_stale_selection_does_not_write(tmp_path,monkeypatch):
    root=sources(tmp_path)
    monkeypatch.setattr(workbench,'training_active',lambda _p:False)
    images={str(p):file_sha256(p) for p in (root/'sources').rglob('*.png')}
    before=workbench.dataset_info(args(dataset=root))
    result=workbench.curate(args(dataset=root,sample='surface_1k_full',status='excluded',split=None,
        note='Keep every source pixel untouched',expected_index_sha256=before['index_sha256']))
    after=workbench.dataset_info(args(dataset=root))
    assert after['materials'][0]['samples'][0]['status']=='excluded'
    assert result['source_bytes_modified'] is False
    assert images=={str(p):file_sha256(p) for p in (root/'sources').rglob('*.png')}
    with pytest.raises(ValueError,match='changed since'):
        workbench.curate(args(dataset=root,sample='surface_1k_full',status='approved',split=None,
            note=None,expected_index_sha256=before['index_sha256']))


def test_reviewing_selected_grid_writes_only_size_specific_metadata(tmp_path, monkeypatch):
    root = sources(tmp_path)
    monkeypatch.setattr(workbench, 'training_active', lambda _p: False)
    before = workbench.dataset_info(args(dataset=root))
    native_status = before['materials'][0]['samples'][0]['status']
    image_hashes = {str(p): file_sha256(p) for p in (root/'sources').rglob('*.png')}
    original_metadata = (root/'dataset.json').read_bytes()
    workbench.curate(args(dataset=root, sample='surface_1k_center', status='excluded', split=None,
        note='Artifact on the 512 grid', review_size=512, expected_index_sha256=before['index_sha256']))
    grid512 = workbench.dataset_info(args(dataset=root, review_size=512))
    grid1024 = workbench.dataset_info(args(dataset=root, review_size=1024))
    assert grid512['materials'][0]['samples'][0]['status'] == 'excluded'
    assert grid512['materials'][0]['samples'][0]['note'] == 'Artifact on the 512 grid'
    assert grid1024['materials'][0]['samples'][0]['status'] == native_status
    assert (root/'dataset.json').read_bytes() == original_metadata
    assert not (root/'.training-data').exists()
    assert image_hashes == {str(p): file_sha256(p) for p in (root/'sources').rglob('*.png')}
    prepared = workbench.prepare_size(args(dataset=root, size=512, automatic_validation=True,
        material=None, expected_index_sha256=before['index_sha256']))
    assert prepared['materials'][0]['samples'][0]['status'] == 'excluded'
    workbench.cleanup_size(args(dataset=Path(prepared['dataset_path'])))
    assert workbench.dataset_info(args(dataset=root, review_size=512))['materials'][0]['samples'][0]['status'] == 'excluded'


def test_active_training_prevents_curation_and_resolution_change(tmp_path,monkeypatch):
    root=sources(tmp_path)
    monkeypatch.setattr(workbench,'training_active',lambda _p:True)
    with pytest.raises(ValueError,match='Stop active'):
        workbench.curate(args(dataset=root,sample='surface_1k_full',status='approved',split=None,note=None,
            expected_index_sha256=None))
    with pytest.raises(ValueError,match='Stop active'):
        workbench.prepare_size(args(dataset=root,size=1024,expected_index_sha256=None))


def test_interrupted_metadata_transaction_recovers_note_and_index(tmp_path,monkeypatch):
    root=sources(tmp_path)
    monkeypatch.setattr(workbench,'training_active',lambda _p:False)
    actual=workbench.atomic_bytes
    calls=[]
    def interrupted(path,value):
        calls.append(path)
        if len(calls)==3:
            raise OSError('Interrupted before index publication')
        return actual(path,value)
    monkeypatch.setattr(workbench,'atomic_bytes',interrupted)
    with pytest.raises(OSError,match='Interrupted'):
        workbench.curate(args(dataset=root,sample='surface_1k_full',status='approved',split=None,
            note='Retain this review',expected_index_sha256=None))
    assert (root/workbench.JOURNAL).is_file()
    monkeypatch.setattr(workbench,'atomic_bytes',actual)
    recovered=workbench.dataset_info(args(dataset=root))['materials'][0]['samples'][0]
    assert recovered['status']=='approved' and recovered['note']=='Retain this review'
    assert not (root/workbench.JOURNAL).exists()


def test_curation_journal_refuses_map_byte_mutation(tmp_path):
    root=sources(tmp_path)
    target=root/'sources/surface/surface_disp_2k.png'
    original=target.read_bytes()
    (root/workbench.JOURNAL).write_text(json.dumps({'schema':workbench.SCHEMA,'changes':[
        {'path':str(target.relative_to(root)),'old_sha256':file_sha256(target),'new_sha256':'0'*64,'new_base64':''}]}))
    with pytest.raises(ValueError,match='JSON'):
        workbench.recover_journal(root)
    assert target.read_bytes()==original


def test_bridge_cli_returns_one_json_object_for_success_and_failure(tmp_path):
    root=sources(tmp_path)
    script=str(ROOT/'scripts/material_workbench.py')
    result=subprocess.run([sys.executable,'-B',script,'dataset','--dataset',str(root)],text=True,capture_output=True)
    assert result.returncode==0
    payload=json.loads(result.stdout)
    assert payload['ok'] and payload['command']=='dataset'
    result=subprocess.run([sys.executable,'-B',script,'dataset','--dataset',str(root/'missing')],text=True,capture_output=True)
    assert result.returncode==1 and not json.loads(result.stdout)['ok']


def test_cleanup_cli_never_removes_original_dataset(tmp_path):
    root=sources(tmp_path)
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/material_workbench.py'),'cleanup-size',
        '--dataset',str(root)],text=True,capture_output=True)
    assert result.returncode==0 and json.loads(result.stdout)['removed'] is False
    assert len(list((root/'sources').rglob('*.png')))==4
