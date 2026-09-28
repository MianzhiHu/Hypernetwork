from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

root = Path('Results/HyperNN_Grid_Search_Evaluation')
output = Path('Figures/Uniform_Prediction')
output.mkdir(parents=True, exist_ok=True)
same = pd.read_csv(root / 'test_results_same_pred.csv')
runs = pd.read_csv(root / 'test_results_all.csv')
tasks = ['ant', 'cct', 'dd', 'motor', 'stroop', 'dpx', 'stopsignal', 'twobytwo']
labels = ['ANT', 'CCT', 'DD', 'Motor', 'Stroop', 'DPX', 'Stop Signal', 'Two-by-Two']
keys = ['config_id', 'task']
assert not runs.duplicated(keys + ['random_split']).any()
assert not same.duplicated(keys).any()
counts = runs.groupby(keys).size().unstack('task').reindex(columns=tasks)
assert counts.notna().all().all() and counts.eq(10).all().all()
assert runs.groupby(keys).random_split.apply(lambda x: set(x) == set(range(10))).all()
# Missing rows in same_pred mean zero affected folds, not missing evaluations.
table = same.pivot(index='config_id', columns='task', values='n_same_pred')
table = table.reindex(index=counts.index, columns=tasks).fillna(0).astype(int)
check = runs.assign(uniform=runs.pred_imbalance.eq(1)).groupby(keys).uniform.sum().unstack('task')
assert np.array_equal(table.to_numpy(), check.reindex(index=table.index, columns=tasks).to_numpy())

# Pool affected folds within each task and hyperparameter level.
columns = ['config_id', 'num_hidden_layers', 'hidden_dims', 'rank', 'participant_emb_dim', 'mapping', 'hyper_hidden_dim', 'hyper_reg']
configs = runs[columns].drop_duplicates()
assert not configs.config_id.duplicated().any()
configs['hidden_width'] = configs.hidden_dims.astype(str).str.split('-').str[0].astype(int)
configs['rank_label'] = configs['rank'].fillna('Full').astype(str).str.replace('.0', '', regex=False)
configs['nodes_label'] = configs.hyper_hidden_dim.fillna('N/A').astype(str).str.replace('.0', '', regex=False)
data = table.rename_axis(columns='task').stack().rename('n_same_pred').reset_index().merge(configs, on='config_id', validate='many_to_one')
data['n_folds'] = 10
parameters = [('num_hidden_layers', 'Hidden layers'), ('hidden_width', 'Hidden units per layer'),
              ('rank_label', 'Rank'), ('participant_emb_dim', 'Embedding dimensions'),
              ('mapping', 'Mapping'), ('nodes_label', 'Hypernetwork nodes')]
# Node counts apply only to nonlinear hypernetworks; linear models have no hidden nodes.
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axes = plt.subplots(8, len(parameters), figsize=(16, 18), sharey=True, layout='constrained')
records = []
for col, (parameter, title) in enumerate(parameters):
    subset = data if parameter != 'nodes_label' else data.loc[data.mapping.eq('nonlinear')]
    levels = sorted(subset[parameter].unique(), key=lambda v: (not str(v).isdigit(), int(v) if str(v).isdigit() else str(v)))
    summary = subset.groupby(['task', parameter], dropna=False).agg(uniform_folds=('n_same_pred', 'sum'), total_folds=('n_folds', 'sum'), n_configs=('config_id', 'size')).reset_index()
    summary['rate_percent'] = 100 * summary.uniform_folds / summary.total_folds
    records.append(summary.rename(columns={parameter: 'level'}).assign(hyperparameter=title))
    for row, (task, label) in enumerate(zip(tasks, labels)):
        ax = axes[row, col]
        values = summary.loc[summary.task.eq(task)].set_index(parameter).reindex(levels)
        ax.bar(range(len(levels)), values.rate_percent, color='#39739D', width=.65)
        ax.set_xticks(range(len(levels)), [str(v) for v in levels], rotation=25 if parameter == 'mapping' else 0)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        if row == 0:
            ax.set_title(title + ('\n(nonlinear only)' if parameter == 'nodes_label' else ''))
        if col == 0:
            ax.set_ylabel(label + '\nUniform prediction rate (%)')
        for i, value in enumerate(values.rate_percent):
            ax.text(i, value + 1.5, f'{value:.1f}', ha='center', va='bottom', fontsize=7)
fig.suptitle('Uniform prediction rate by hyperparameter and task (including ANT)\nAffected folds / all evaluated folds at each level; other hyperparameters pooled', fontsize=13)
fig.savefig(output / 'hyperparameters_by_task.png', dpi=300)
fig.savefig(output / 'hyperparameters_by_task.pdf')
fig.savefig(output / 'hyperparameters_by_task.svg')
plt.close(fig)
pd.concat(records, ignore_index=True).to_csv(output / 'hyperparameters_by_task.csv', index=False)
print('Saved hyperparameters_by_task: 8 tasks x 6 hyperparameters; fold-weighted rates, zero-count configurations included.')
print('Regularization levels:', configs.hyper_reg.unique())
