"""Create a reproducible per-body surface-error table and plot."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/orca_collision_refit'
LABELS=['Palm','Thumb root','Thumb middle','Thumb proximal','Thumb tip',
        'Index base','Index proximal','Index tip','Middle base','Middle proximal','Middle tip',
        'Ring base','Ring proximal','Ring tip','Pinky base','Pinky proximal','Pinky tip']


def main():
    report=json.loads((BASE/'G07_convex_palm_candidate/geometry_report.json').read_text())
    output=BASE/'report';output.mkdir(exist_ok=True)
    rows=[]
    for label,(body,values) in zip(LABELS,report['bodies'].items(),strict=True):
        row=dict(body=body,label=label)
        for mode in ('old','new'):
            for key in ('uncovered_p95_mm','proxy_to_mesh_p95_mm','primitives'):
                row[f'{mode}_{key}']=values[mode][key]
        rows.append(row)
    with (output/'geometry_errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,9),layout='constrained',sharey=True)
    y=np.arange(len(rows))
    for ax,key,title in zip(axes,('uncovered_p95_mm','proxy_to_mesh_p95_mm'),
                             ('STL outside collision proxy (p95)','Exposed proxy to STL (p95)')):
        for mode,offset,color in [('old',-.16,'#667580'),('new',.16,'#168369')]:
            values=np.array([r[f'{mode}_{key}'] for r in rows])
            ax.scatter(np.maximum(values,.01),y+offset,label=mode,c=color,s=35)
        ax.set_xscale('log');ax.set_xlim(.009,80)
        ax.set_xlabel('Distance, mm (zero plotted at 0.01)');ax.set_title(title)
        ax.set_yticks(y,LABELS);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    axes[0].invert_yaxis()
    fig.suptitle('Orca collision refit: fixed STL samples, same link frames')
    fig.savefig(output/'geometry_errors.png',dpi=160);plt.close(fig)
    print(output/'geometry_errors.png')


if __name__=='__main__':
    main()
