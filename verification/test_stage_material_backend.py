"""The app ships only its current material source dependency closure."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from stage_material_backend import MATERIAL_SOURCES, stage


def write(root, name, content=b'fixture source\n'):
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(content)
    return file


def test_stage_contains_current_sources_and_removes_unused_development_payload(tmp_path):
    root, destination = tmp_path / 'repo', tmp_path / 'Resources/MaterialBackend'
    for name in MATERIAL_SOURCES:
        write(root, 'scripts/' + name)
    write(root, 'src/ipde/__init__.py')
    write(root, 'src/ipde/learned_depth.py')
    write(root, 'src/ipde/__pycache__/ignored.py')
    write(root, 'LICENSE')
    write(root, 'scripts/material_training_cycle.py')
    write(root, 'scripts/export_material_candidate.py')
    write(destination, 'frozen_dino_height.py')
    write(destination, 'model.safetensors')
    write(destination, '__pycache__/stale.pyc')
    stage(root, destination)
    files = {str(file.relative_to(destination)) for file in destination.rglob('*') if file.is_file()}
    assert files == set(MATERIAL_SOURCES) | {'LICENSE'}
    assert not list(destination.rglob('__pycache__'))
    before = {name: (destination / name).read_bytes() for name in files}
    stage(root, destination)
    assert before == {name: (destination / name).read_bytes() for name in files}


def test_shipped_dependency_closure_has_no_discarded_material_experiment():
    assert 'material_model_workbench.py' in MATERIAL_SOURCES
    assert 'material_lora.py' in MATERIAL_SOURCES
    assert all('dino' not in name and 'da3' not in name for name in MATERIAL_SOURCES)
    assert 'material_training_cycle.py' not in MATERIAL_SOURCES
    assert 'export_material_candidate.py' not in MATERIAL_SOURCES
    for name in MATERIAL_SOURCES:
        assert (ROOT / 'scripts' / name).is_file()


def test_isolated_shipped_bundle_discovers_ambient_source_with_verified_package_sidecar(tmp_path):
    import json
    import subprocess
    import numpy as np
    from material_dataset import PACKAGE_AUDIT_SCHEMA, file_sha256, write_json, write_png

    bundle = tmp_path/'MaterialBackend'
    stage(ROOT, bundle)
    folder = tmp_path/'sources/Test'
    folder.mkdir(parents=True)
    downloaded = {}
    audited = []
    for role, suffix, values in (
        ('input', 'Color', np.full((2048,2048,3), 96, np.uint8)),
        ('height', 'Displacement', np.full((2048,2048,1), 32000, np.uint16)),
    ):
        path = folder/f'Test_2K-PNG_{suffix}.png'
        write_png(path, values)
        checksum, count = file_sha256(path), path.stat().st_size
        downloaded[role] = {'path': str(path), 'sha256': checksum, 'bytes': count}
        audited.append({'source_filename': path.name, 'source_sha256': checksum,
            'member_sha256': checksum, 'source_bytes': count, 'member_bytes': count,
            'exact_full_file_match': True, 'source_stable': True, 'archive_member': path.name})
    audit_path = tmp_path/'ambient-package-audit.json'
    write_json(audit_path, {'schema': PACKAGE_AUDIT_SCHEMA, 'provider': 'ambientCG',
        'material_id': 'test', 'asset_id': 'Test', 'asset_url': 'https://ambientcg.com/view?id=Test',
        'package': {'url': 'https://ambientcg.com/get?file=Test_2K-PNG.zip',
                    'filename': 'Test_2K-PNG.zip', 'file_bytes': 123456, 'sha256': 'a' * 64},
        'license': {'spdx': 'CC0-1.0', 'url': 'https://docs.ambientcg.com/license/',
                    'snapshot_sha256': 'b' * 64}, 'files': audited})
    write_json(folder/'material-source-2k.json', {'provider': 'ambientCG', 'material_id': 'test',
        'resolution': '2k', 'downloaded_maps': downloaded, 'package_audit_path': str(audit_path)})
    child = subprocess.run([sys.executable, '-I', '-c', '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import material_dataset
materials = material_dataset.discover_source_sets(Path(sys.argv[2]))
assert Path(material_dataset.__file__).parent == Path(sys.argv[1])
assert "material_recreation" not in sys.modules
assert len(materials) == 1
assert materials[0]["material_id"] == "test_2k"
for role in ("input", "height"):
    source = materials[0]["maps"][role]
    assert source["provider"] == "ambientCG"
    assert source["license"] == "CC0-1.0"
    assert source["published_member_sha256"] == source["file_sha256"]
    assert source["published_member_bytes"] == source["file_bytes"]
print(json.dumps({"sources": len(materials), "provider": "ambientCG"}))
''', str(bundle), str(tmp_path/'sources')], cwd=tmp_path, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {'sources': 1, 'provider': 'ambientCG'}
    assert not (bundle/'material_recreation.py').exists()
