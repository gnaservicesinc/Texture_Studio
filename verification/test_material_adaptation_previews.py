"""Display-only adaptation comparisons preserve raw arrays and honest scopes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_material_curriculum import phase_figure
from render_material_training import ADAPTATION_SCHEMA, adaptation_sheet, adaptation_summary, visual_height
import matplotlib.pyplot as plt


MATERIALS = ['white_stucco_02', 'farm_soil', 'cotton_jersey', 'broken_brick_wall']


def fixture(run: Path):
    run.mkdir()
    summary = {'schema': ADAPTATION_SCHEMA, 'matched_complete_schedules': True,
        'materials': MATERIALS, 'requested_updates_per_crop': 2,
        'initial_shared_head_state_sha256': 'a' * 64, 'variants': {}}
    yy, xx = np.mgrid[:256, :256]
    target = (0.5 + 0.1 * np.sin(xx / 7) + 0.08 * np.cos(yy / 11)).astype(np.float32)
    for name in ('frozen', 'lora'):
        phases = []
        for step in (0, 4, 8):
            record = {'step': step}
            for key in ('training_fit', 'known_material_disjoint_regions'):
                record[key] = {'samples': [{'material_id': material, 'metrics': {
                    'detail_highpass_correlation_radius_4': None if step == 0 else 0.3 + step / 20}}
                    for material in MATERIALS]}
            phases.append(record)
        summary['variants'][name] = {'complete': True, 'status': 'complete',
            'completed_steps': 8, 'per_crop_update_counts': [2] * 4,
            'initial_head_state_sha256': 'a' * 64,
            'zero_adapter_initial_features_and_predictions_identical': name == 'lora',
            'selected_step': 0 if name == 'frozen' else 4, 'phase_history': phases}
        for label, suffix in (('final-training-fit', '001'), ('final-known-material-disjoint-region', '003')):
            directory = run / name / label
            directory.mkdir(parents=True)
            np.save(directory / 'target.height.float32.npy', target, allow_pickle=False)
            np.save(directory / (label + '.height.float32.npy'), target * (0.8 if name == 'frozen' else 0.9), allow_pickle=False)
            (directory / (label + '.prediction.json')).write_text(json.dumps({
                'schema': ADAPTATION_SCHEMA, 'variant': name,
                'sample_id': MATERIALS[0] + '_auto_' + suffix,
                'snapshot': 'final weights; selected validation weights may be initial or earlier'}))
    (run / 'summary.json').write_text(json.dumps(summary))
    return summary


def file_hashes(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def test_completed_display_and_phase_plot_do_not_change_numeric_exports(tmp_path):
    run = tmp_path / 'run'
    summary = fixture(run)
    original = file_hashes(run)
    image = adaptation_sheet(run)
    assert image.size == (1536, 717)
    reference = np.load(run / 'frozen/final-training-fit/target.height.float32.npy')
    assert np.array_equal(np.asarray(image)[123:379, :256], np.asarray(visual_height(reference, 2, False)))
    figure = phase_figure(run)
    assert len(figure.axes) == 8
    assert len(figure.axes[0].lines) == 3  # two model series and the zero reference
    series = figure.axes[0].lines[0]
    assert np.array_equal(series.get_xdata(), [0, 4, 8])
    assert np.isnan(series.get_ydata()[0])  # undefined flat correlation remains missing
    assert 'selected step 0' in figure._suptitle.get_text()
    plt.close(figure)
    assert file_hashes(run) == original
    assert adaptation_summary(run) == summary


@pytest.mark.parametrize('defect', ('incomplete', 'unequal_counts', 'different_head', 'missing_identity', 'missing_initial', 'unrecorded_selected'))
def test_invalid_or_unmatched_comparison_rejected_before_render(tmp_path, defect):
    run = tmp_path / 'run'
    summary = fixture(run)
    details = summary['variants']['lora']
    if defect == 'incomplete':
        details['complete'] = False
    elif defect == 'unequal_counts':
        details['per_crop_update_counts'][0] = 1
    elif defect == 'different_head':
        details['initial_head_state_sha256'] = 'b' * 64
    elif defect == 'missing_identity':
        details['zero_adapter_initial_features_and_predictions_identical'] = False
    elif defect == 'missing_initial':
        details['phase_history'] = details['phase_history'][1:]
    else:
        details['selected_step'] = 5
    (run / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        adaptation_sheet(run)
    with pytest.raises(ValueError):
        phase_figure(run)


def test_preview_refuses_different_reference_or_mislabeled_selected_arrays(tmp_path):
    run = tmp_path / 'run'
    fixture(run)
    directory = run / 'lora/final-training-fit'
    path = directory / 'target.height.float32.npy'
    original = np.load(path, allow_pickle=False)
    altered = original.copy()
    altered[0, 0] += 0.001
    np.save(path, altered, allow_pickle=False)
    with pytest.raises(ValueError, match='identical native reference'):
        adaptation_sheet(run)
    np.save(path, original, allow_pickle=False)
    metadata = directory / 'final-training-fit.prediction.json'
    payload = json.loads(metadata.read_text())
    payload['snapshot'] = 'selected weights'
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='final adaptation exports'):
        adaptation_sheet(run)


def test_native_detail_crop_has_shared_gain_without_resampling(tmp_path):
    yy, xx = np.mgrid[:512, :512]
    values = (0.5 + 0.001 * xx + 0.05 * np.sin(yy)).astype(np.float32)
    original = values.copy()
    detail = np.asarray(visual_height(values, 4, True))[:, :, 0]
    expected = np.rint(np.clip(0.5 + 4 * (values - values.mean())[128:384, 128:384], 0, 1) * 255).astype(np.uint8)
    assert np.array_equal(detail, expected)
    assert np.array_equal(values, original)
