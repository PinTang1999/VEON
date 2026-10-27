#!/usr/bin/env python3
"""Plot measured AP only, requiring identical evaluation instances by default."""
import argparse
import json
from pathlib import Path

from common import write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+')
    p.add_argument('--labels', nargs='+')
    p.add_argument('--groups', nargs='+', choices=['seen', 'heldout', 'novel', 'overall'], default=['seen', 'heldout', 'overall'])
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in a.results]
    labels = a.labels or [Path(path).parent.name for path in a.results]
    if len(labels) != len(reports):
        p.error('--labels must have one label per result')
    for report in reports:
        if not report['metadata']['complete']:
            raise ValueError('Incomplete smoke-test outputs cannot be used for a performance comparison')
    for scope in ('all', 'visible'):
        for group in a.groups:
            keys = [r['aggregate'][group][scope]['instance_keys'] for r in reports]
            if any(k != keys[0] for k in keys[1:]):
                raise ValueError('Different query/instance sets in {}/{}; evaluate identical prompts and held-out analysis groups'.format(group, scope))
    for report in reports[1:]:
        for key in ('benchmark_sha256', 'ann_file_sha256', 'annotations_sha256'):
            if report['metadata'][key] != reports[0]['metadata'][key]:
                raise ValueError('Evaluation data provenance differs: {}'.format(key))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    rows = []
    width = .75 / len(reports)
    for ax, scope in zip(axes, ('all', 'visible')):
        for i, (report, label) in enumerate(zip(reports, labels)):
            values = [report['aggregate'][group][scope]['map'] for group in a.groups]
            positions = np.arange(len(a.groups)) + (i - (len(reports) - 1) / 2) * width
            for x, value, group in zip(positions, values, a.groups):
                rows.append(dict(model=label, scope=scope, group=group, map_percent=None if value is None else 100 * value))
                if value is None:
                    ax.text(x, 1, 'N/A', rotation=90, ha='center', fontsize=8)
            ax.bar(positions, [np.nan if v is None else 100 * v for v in values], width, label=label)
        ax.set_xticks(range(len(a.groups)), [{'heldout': 'Held-out'}.get(g, g.title()) for g in a.groups])
        ax.set_title('All points' if scope == 'all' else 'Visible points')
        ax.set_ylim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Scene-query mAP (%)')
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .90))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'vocabulary_comparison.pdf', bbox_inches='tight')
    fig.savefig(out / 'vocabulary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    write_csv(out / 'vocabulary_comparison.csv', rows)


if __name__ == '__main__':
    main()
