"""Resource budgets constrain complete training grids without limiting generation."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_material_pbrnxt as trainer


def memory64(monkeypatch):
    monkeypatch.setattr(trainer.os, 'sysconf', lambda name: 16 * 1024 * 1024 if name == 'SC_PHYS_PAGES' else 4096)


def test_training_budget_qualifies_checkpointed_2048_final_map_and_rejects_unmeasured_decoder(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.resource_plan(1024, 56, 4)
    assert plan['combined_estimate_gib'] == 27 and plan['refinement_scope'] == 'final-map'
    wider = trainer.resource_plan(1024, 56, 4, 'map-decoder')
    assert wider['combined_estimate_gib'] == 42
    qualified = trainer.resource_plan(2048, 48, .5)
    assert qualified['required_memory_gib'] == 47.5 and qualified['recommended_memory_gib'] == 48
    assert qualified['qualified_native_2048'] and qualified['training_activation_checkpointing'] == 'rrdb-block-v1'
    with pytest.raises(ValueError, match='smaller complete training grid'):
        trainer.resource_plan(2048, 56, 4, 'map-decoder')
    with pytest.raises(ValueError, match='smaller complete training grid'):
        trainer.resource_plan(2048, 46, .5)


def test_automatic_plan_reports_memory_needed_without_hiding_size_due_to_previous_small_budget(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.training_memory_plan(2048, .5)
    assert plan['hardware_supported'] and plan['recommended_memory_gib'] == 48
    impossible = trainer.training_memory_plan(4096, .5)
    assert not impossible['hardware_supported']
    assert impossible['recommended_memory_gib'] <= impossible['maximum_memory_gib']


def test_inference_budget_does_not_apply_backward_pass_memory_rejection(monkeypatch):
    memory64(monkeypatch)
    plan = trainer.resource_plan(2048, 56, 4, 'map-decoder', inference=True)
    assert plan['driver_budget_gib'] == 50 and plan['operation'] == 'inference'
    assert not plan['peak_measured'] and plan['driver_estimate_gib'] is None
    assert 'unmeasured' in plan['estimate_basis']
    with pytest.raises(ValueError, match='at least 1 GiB'):
        trainer.resource_plan(2048, 56, 54, inference=True)


@pytest.mark.parametrize('size', [0, 64, 128, 255, 257, 1000, 2047])
def test_invalid_training_grid_rejected(size, monkeypatch):
    memory64(monkeypatch)
    with pytest.raises(ValueError, match='no padding or resizing'):
        trainer.resource_plan(size, 56, 1)


@pytest.mark.parametrize('memory,cache', [(0, 1), (-1, 1), (61, 1), (float('nan'), 1), (56, -1), (56, float('inf'))])
def test_invalid_resource_budgets_rejected(memory, cache, monkeypatch):
    memory64(monkeypatch)
    with pytest.raises(ValueError):
        trainer.resource_plan(256, memory, cache)


def test_standalone_trainer_uses_the_same_supported_json_bridge(monkeypatch):
    import material_model_workbench
    received = []
    monkeypatch.setattr(material_model_workbench, 'main', lambda argv: received.append(argv) or 0)
    arguments = ['train', '--dataset', 'source-dataset', '--output', 'new-run', '--size', '512']
    assert trainer.main(arguments) == 0
    assert received == [arguments]
