"""Memory diagnostics never gate complete training grids or inference."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_material_pbrnxt as trainer


def test_bundled_training_cli_runs_in_isolated_mode(tmp_path):
    from scripts.stage_material_backend import stage
    root = Path(__file__).resolve().parents[1]
    backend = tmp_path / 'MaterialBackend'
    stage(root, backend)
    result = subprocess.run([sys.executable, '-I', '-B', str(backend / 'train_material_pbrnxt.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert 'train' in result.stdout


def memory64(monkeypatch):
    monkeypatch.setattr(trainer.os, 'sysconf', lambda name: 16 * 1024 * 1024 if name == 'SC_PHYS_PAGES' else 4096)


def test_training_memory_diagnostics_do_not_reject_any_scope_or_small_legacy_budget(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.resource_plan(1024, 56, 4)
    assert plan['combined_estimate_gib'] == 27 and plan['refinement_scope'] == 'final-map'
    wider = trainer.resource_plan(1024, 56, 4, 'map-decoder')
    assert wider['combined_estimate_gib'] == 42
    qualified = trainer.resource_plan(2048, 48, .5)
    assert qualified['required_memory_gib'] == 47.5 and qualified['recommended_memory_gib'] == 48
    assert qualified['qualified_native_2048'] and qualified['training_activation_checkpointing'] == 'rrdb-block-v1'
    wider = trainer.resource_plan(2048, 56, 4, 'map-decoder')
    assert wider['combined_estimate_gib'] > 56
    smaller = trainer.resource_plan(2048, 1, .5)
    assert smaller['required_memory_gib'] == 47.5
    assert smaller['memory_admission_enabled'] is False


def test_automatic_plan_reports_memory_needed_without_hiding_size_due_to_previous_small_budget(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.training_memory_plan(2048, .5)
    assert plan['hardware_supported'] and plan['recommended_memory_gib'] == 48
    impossible = trainer.training_memory_plan(4096, .5)
    assert not impossible['hardware_supported']
    assert impossible['recommended_memory_gib'] <= impossible['maximum_memory_gib']


def test_inference_has_no_memory_admission_or_custom_driver_cap(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.resource_plan(2048, 56, 4, 'map-decoder', inference=True)
    assert plan['operation'] == 'inference' and plan['memory_admission_enabled'] is False
    assert 'driver_budget_gib' not in plan
    assert not plan['peak_measured'] and plan['driver_estimate_gib'] is None
    assert 'unmeasured' in plan['estimate_basis']
    assert trainer.resource_plan(2048, 1, 54, inference=True)['memory_admission_enabled'] is False


@pytest.mark.parametrize('size', [0, 64, 128, 255, 257, 1000, 2047])
def test_invalid_training_grid_rejected(size, monkeypatch):
    memory64(monkeypatch)
    with pytest.raises(ValueError, match='no padding or resizing'):
        trainer.resource_plan(size, 56, 1)


@pytest.mark.parametrize('cache', [-1, float('nan'), float('inf')])
def test_invalid_optional_cache_sizes_rejected(cache, monkeypatch):
    memory64(monkeypatch)
    with pytest.raises(ValueError):
        trainer.resource_plan(256, None, cache)


@pytest.mark.parametrize('memory', [None, 0, -1, 61, float('nan'), float('inf')])
def test_retired_memory_argument_cannot_disable_a_run(memory, monkeypatch):
    memory64(monkeypatch)
    assert trainer.resource_plan(256, memory, 1)['memory_admission_enabled'] is False


def test_standalone_trainer_uses_the_same_supported_json_bridge(monkeypatch):
    import material_model_workbench
    received = []
    monkeypatch.setattr(material_model_workbench, 'main', lambda argv: received.append(argv) or 0)
    arguments = ['train', '--dataset', 'source-dataset', '--output', 'new-run', '--size', '512']
    assert trainer.main(arguments) == 0
    assert received == [arguments]
