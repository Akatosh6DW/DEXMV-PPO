"""Read-only aggregation of completed refit runs and held-out episode scores."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'artifacts/orca_collision_refit'


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def wilson(k,n):
    z=1.959963984540054
    center=(k/n+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius),float(center+radius)]


def main():
    out=ROOT/'report';out.mkdir(exist_ok=True)
    rows=[];curves={}
    for folder in sorted(ROOT.glob('R*')):
        if not (folder/'summary.json').exists():
            continue
        config=json.loads((folder/'run_config.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        metrics=read_csv(folder/'metrics.csv')
        model=Path(config['model']).parent.name
        row=dict(run=folder.name,model=model,arm=config['arm'],seed=config['seed'],
                 steps=config['steps'],first=summary['first'],last=summary['last'],max=summary['max'],
                 best_step=summary['best_step'],last_mean_hold=float(metrics[-1]['mean_hold']),
                 elapsed_sec=float(metrics[-1]['elapsed_sec']))
        rows.append(row);curves[folder.name]=(model,metrics)
    with (out/'training_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for name,(model,metrics) in curves.items():
        col=0 if model.startswith('G07') else 1
        x=np.array([int(r['step']) for r in metrics])/1000
        axes[0,col].plot(x,[100*float(r['success30']) for r in metrics],marker='o',label=name)
        axes[1,col].plot(x,[float(r['mean_hold']) for r in metrics],marker='o',label=name)
    for col,title in enumerate(('G07: global surface refit','G08: local thumb refinement')):
        axes[0,col].set_title(title)
        for row in (0,1):
            axes[row,col].set_xlabel('Environment steps (thousands)')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7)
        axes[0,col].set_ylim(-3,103);axes[0,col].set_ylabel('30 s + valid final state (%)')
        axes[1,col].set_ylim(-1,40);axes[1,col].set_ylabel('Mean longest clean hold (s)')
    fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    text=['# Refit Experiment Tables','','Development set: 9200-9215. See the Chinese log for protocol and limitations.','',
          '| Run | Model | First % | Last % | Max % | Best step | Last mean hold s |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['run']} | {r['model'][:3]} | {100*r['first']:.2f} | {100*r['last']:.2f} | {100*r['max']:.2f} | {r['best_step']} | {r['last_mean_hold']:.2f} |")
    held=ROOT/'T01_heldout'
    if (held/'metrics.csv').exists():
        tests=read_csv(held/'metrics.csv');statistics={}
        text+=['','## Held-Out States','','Intervals describe variation across initial states for a fixed checkpoint, not across training seeds.','',
               '| Policy | Success / N | Wilson 95% interval | Mean hold s | Final hold >=30 % |',
               '|---|---:|---:|---:|---:|']
        for row in tests:
            episodes=json.loads((held/row['name']/'episodes.json').read_text())
            successes=np.asarray(episodes['success30'],bool)
            ci=wilson(int(successes.sum()),len(successes))
            statistics[row['name']]=dict(successes=int(successes.sum()),n=len(successes),wilson95=ci)
            text.append(f"| {row['name']} | {successes.sum()} / {len(successes)} | {100*ci[0]:.1f}-{100*ci[1]:.1f}% | {float(row['mean_hold']):.2f} | {100*float(row['final_hold30']):.2f} |")
        (out/'heldout_statistics.json').write_text(json.dumps(statistics,indent=2))
    text+=['','![Learning curves](learning_curves.png)','']
    (out/'results.md').write_text('\n'.join(text))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
