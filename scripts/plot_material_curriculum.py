#!/usr/bin/env python3
"""Plot actual per-material phase metrics from the bounded fitting diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from render_material_training import ADAPTATION_SCHEMA, adaptation_summary


def phase_figure(run: Path):
    summary = json.loads((run / 'summary.json').read_text())
    adaptation = summary.get('schema') == ADAPTATION_SCHEMA
    if adaptation:
        summary = adaptation_summary(run)
    elif summary.get('schema') != 'texture-studio-four-material-fit-diagnostic-v1' or not summary.get('matched_complete_12_channel_schedules'):
        raise ValueError('Expected completed, matched four-material comparison')
    materials = summary['materials'] if adaptation else summary['materials_in_introduction_order']
    figure, axes = plt.subplots(2, len(materials), figsize=(15, 6.7), sharex=True, sharey=True, constrained_layout=True)
    styles = {'mixed12': ('#2863b8', '-', 'Mixed, 12 channels'),
              'curriculum12': ('#a85217', '--', 'Gradual replay, 12 channels'),
              'mixed24': ('#3b873d', ':', 'Mixed, 24 channels'),
              'frozen': ('#2863b8', '-', 'Frozen encoder'),
              'lora': ('#a85217', '--', 'Rank-8 LoRA')}
    metric = 'detail_highpass_correlation_radius_4'
    for variant, details in summary['variants'].items():
        if not details['complete']:
            raise ValueError('Cannot compare incomplete update schedules')
        report = details if adaptation else json.loads((run / variant / 'summary.json').read_text())
        color, linestyle, label = styles[variant]
        for row, key in enumerate(('training_fit', 'known_material_disjoint_regions')):
            phases = report['phase_history']
            for column, material in enumerate(materials):
                values = [next(sample['metrics'][metric] for sample in phase[key]['samples']
                               if sample['material_id'] == material) for phase in phases]
                # A flat prediction has undefined correlation, not a zero score.
                values = [np.nan if value is None else value for value in values]
                axes[row, column].plot([phase['step'] for phase in phases], values,
                                       color=color, linestyle=linestyle, marker='o', linewidth=1.6, label=label)
    for row in range(2):
        for column, material in enumerate(materials):
            axis = axes[row, column]
            axis.set_ylim(-1.02, 1.02)
            axis.axhline(0, color='gray', linewidth=.7, alpha=.35)
            axis.grid(alpha=0.2)
            for boundary in summary.get('phase_end_steps', [])[:-1]:
                axis.axvline(boundary, color='gray', linewidth=.65, alpha=.25)
            if row == 0:
                axis.set_title(material.replace('_', ' '), fontsize=10)
            if column == 0:
                axis.set_ylabel(('Repeated training crop' if row == 0 else 'Disjoint known-material region') + '\nNative detail correlation')
            if row == 1:
                axis.set_xlabel('Total optimizer updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), frameon=False)
    if adaptation:
        selection = '; '.join(f"{name} selected step {details['selected_step']}" for name, details in summary['variants'].items())
        title = 'Matched frozen/LoRA fitting: exact same initial head and per-crop update schedule\n' + selection + '; curves show actual evaluated snapshots\nKnown-material regions only; warm-start head already fitted to stucco'
    else:
        title = 'Four-material fitting: equal final updates per crop; phase order differs\nSeparate regions belong to the same source materials; no fresh-material generalization claim'
    figure.suptitle(title, fontsize=12)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    figure = phase_figure(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)
    print(args.output)


if __name__ == '__main__':
    main()
