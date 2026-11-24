"""Compare held-out performance of neural and cognitive models.

Each task has a shared-scale NLL panel. Points give equally weighted fold means;
error bars give fold-wise SE: sample SD (ddof=1) / sqrt(number of folds). Individual and group fits occupy separate labelled rows.
Vertical lines mark HyperNN means; right-hand values report model means.
The input is the selected-model table exported by model_evaluation.py.
Exports editable SVG/PDF, a 600-dpi PNG preview, source data and summary data.
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from task_config import selected_config


BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Results/Cognitive_Model_Evaluation/test_performance_by_fold.csv'
OUTPUT = BASE / 'Figures/Test_Performance'

TASK_LABELS = {
    'cct': 'Columbia Card Task',
    'dd': 'Delay Discounting Task',
    'stopsignal': 'Stop-Signal Task',
    'motor': 'Motor Selective Stop-Signal Task',
    'twobytwo': 'Cued Task Switching Task',
}

MODEL_LABELS = {
    'pt': 'PT',
    'pt_prob': 'PT + probability',
    'pt_loss_shape': 'PT + loss shape',
    'exponential': 'Exponential',
    'hyperbolic': 'Hyperbolic',
    'hyperboloid': 'Hyperboloid',
    'logistic': 'Logistic',
    'hr_exgau': 'HR ex-Gaussian',
    'rdex': 'RDEX',
    'ddm': 'DDM',
    'lba': 'LBA',
    'ccr': 'CCR',
}
COGNITIVE_ORDER = {
    'cct': ['pt', 'pt_prob', 'pt_loss_shape'],
    'dd': ['exponential', 'hyperbolic', 'hyperboloid'],
    'motor': ['logistic', 'hr_exgau', 'rdex'],
    'stopsignal': ['logistic', 'hr_exgau', 'rdex'],
    'twobytwo': ['ddm', 'lba', 'ccr'],
}
# Row label, color, marker, marker face.
STYLES = {
    'HyperNN': ('HyperNN', '#2F808D', 'D', '#2F808D'),
    'BasicNN': ('Basic NN', '#687F91', 's', 'white'),
    'cognitive_individual': ('individual', '#82649E', 'o', '#82649E'),
    'cognitive_group': ('group', '#82649E', 's', 'white'),
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'text.color': '#263743',
    'axes.labelcolor': '#263743',
    'axes.edgecolor': '#B5C4D0',
    'xtick.color': '#687F91',
    'ytick.color': '#263743',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.size': 0,
    'legend.frameon': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})


data = pd.read_csv(INPUT)
summary = data.groupby(['task', 'model_family', 'model'], sort=False).agg(
    test_nll_mean=('test_nll', 'mean'),
    test_nll_sd=('test_nll', 'std'),
    n_folds=('random_split', 'nunique'),
).reset_index()
summary['test_nll_se'] = summary.test_nll_sd / np.sqrt(summary.n_folds)

# Quantitative grid: 183 mm wide; every task uses the same NLL scale.
tasks = sorted(data.task.unique(), key=list(TASK_LABELS).index)
fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 1.48 * len(tasks) + 1.5),
                         sharex=True, squeeze=False)
fig.subplots_adjust(left=0.335, right=0.905, top=0.84, bottom=0.065, hspace=0.27)
upper = np.ceil((summary.test_nll_mean + summary.test_nll_se).max() * 10) / 10

for task, ax in zip(tasks, axes[:, 0]):
    task_summary = summary.loc[summary.task.eq(task)]
    labels = [''] * len(task_summary)
    hypernn_mean = task_summary.loc[task_summary.model_family.eq('HyperNN'), 'test_nll_mean'].item()
    ax.axvline(hypernn_mean, color=STYLES['HyperNN'][1], linewidth=0.9, zorder=1)

    for result in task_summary.itertuples():
        family = result.model_family
        label, color, marker, face = STYLES[family]
        if family.startswith('cognitive'):
            variant = result.model.split('_', 1)[1]
            y = 2 + 2 * COGNITIVE_ORDER[task].index(variant) + (family == 'cognitive_group')
            label = f'{MODEL_LABELS[variant]} ({label})'
        else:
            y = ['HyperNN', 'BasicNN'].index(family)
        labels[y] = label
        ax.errorbar(result.test_nll_mean, y, xerr=result.test_nll_se,
                    fmt=marker, color=color, markerfacecolor=face, markersize=4.5,
                    markeredgewidth=0.8, elinewidth=0.65, capsize=1.8, capthick=0.65,
                    zorder=3)
        ax.text(1.035, y, f'{result.test_nll_mean:.3f}', va='center', ha='left',
                transform=ax.get_yaxis_transform(), fontsize=7)

    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, upper)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.tick_params(axis='x', labelsize=7, length=2.5)
    fig.text(0.055, ax.get_position().y1 + 0.012, TASK_LABELS[task],
             fontsize=8, fontweight='bold')

fig.suptitle('Model comparison', x=0.055, y=0.98,
             ha='left', fontsize=12, fontweight='bold')
fig.text(0.055, 0.935, 'Error bars: ±1 SE across 10 test folds',
         fontsize=10, color='#687F91')
reference = Line2D([], [], color=STYLES['HyperNN'][1], linewidth=0.9,
                   label='Vertical line: HyperNN mean NLL')
group_fit = Line2D([], [], linestyle='none', marker='s', color=STYLES['cognitive_group'][1],
                   markerfacecolor='white', markersize=4.5, markeredgewidth=0.8,
                   label='Square: Group Fit')
individual_fit = Line2D([], [], linestyle='none', marker='o', color=STYLES['cognitive_individual'][1],
                        markerfacecolor=STYLES['cognitive_individual'][1], markersize=4.5,
                        markeredgewidth=0.8, label='Circle: Individual Fit')
fig.legend(handles=[reference, group_fit, individual_fit], loc='upper left',
           bbox_to_anchor=(0.045, 0.917), ncol=3, columnspacing=1.2,
           handletextpad=0.5, labelcolor=[STYLES['HyperNN'][1], STYLES['cognitive_group'][1],
                                        STYLES['cognitive_individual'][1]], fontsize=7)
fig.text(0.945, 0.865, 'Mean NLL', ha='center', fontsize=7, color='#687F91')
axes[-1, 0].set_xlabel('Mean held-out NLL per trial (lower is better)', fontsize=10)
OUTPUT.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT / 'test_performance_source.csv', index=False)
summary.to_csv(OUTPUT / 'test_performance_summary.csv', index=False)
fig.savefig(OUTPUT / 'test_performance.png', dpi=600)
plt.close(fig)


# Pairwise CCA: separate full-data and CV-consensus figures.
# Symmetric cells show the same saved task-pair estimate; diagonal values are
# self-correlations (1 by definition), with no significance test on the diagonal.
CCA_INPUT = BASE / 'Results/Main_Model_Evaluation' / selected_config / 'pairwise_cca_results.csv'
CCA_OUTPUT = BASE / 'Figures/Pairwise_CCA'
cca_data = pd.read_csv(CCA_INPUT, usecols=[
    'name', 'task_a', 'task_b', 'observed_cca', 'cca_p_adj',
    'observed_cca_cv', 'cca_p_cv_adj',
])
cca_tasks = sorted(set(cca_data.task_a) | set(cca_data.task_b), key=list(TASK_LABELS).index)
cca_labels = [fill(TASK_LABELS[task], width=50, break_long_words=False, break_on_hyphens=False)
              for task in cca_tasks]
cca_positions = {task: i for i, task in enumerate(cca_tasks)}
cca_cmap = LinearSegmentedColormap.from_list('task_cca', ['#F2F7F8', '#83B6BE', '#175563'])
CCA_OUTPUT.mkdir(parents=True, exist_ok=True)
cca_data.to_csv(CCA_OUTPUT / 'pairwise_cca_source.csv', index=False)

for metric, p_column, title, filename in [
    ('observed_cca', 'cca_p_adj', 'Full-data embeddings', 'pairwise_cca_full_data'),
    ('observed_cca_cv', 'cca_p_cv_adj', 'Cross-Task Embedding Correlation', 'pairwise_cca_cv_consensus'),
]:
    matrix = np.eye(len(cca_tasks))
    annotations = np.full(matrix.shape, '1.00', dtype=object)
    for result in cca_data.itertuples():
        i, j = cca_positions[result.task_a], cca_positions[result.task_b]
        value, q = getattr(result, metric), getattr(result, p_column)
        matrix[i, j] = matrix[j, i] = value
        stars = '*' * sum(q < threshold for threshold in [0.05, 0.01, 0.001])
        annotations[i, j] = annotations[j, i] = f'{value:.2f}' + (rf'$^{{{stars}}}$' if stars else '')

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.23, right=0.82, bottom=0.28, top=0.92)
    image = ax.imshow(matrix, cmap=cca_cmap, vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, annotations[i, j], ha='center', va='center', fontsize=8,
                color='white' if value > 0.65 else '#263743')
    ax.set_xticks(range(len(cca_tasks)), cca_labels, rotation=45, ha='right', rotation_mode='anchor',
                  multialignment='center', fontsize=7)
    ax.set_yticks(range(len(cca_tasks)), cca_labels, rotation=0, ha='right',
                  multialignment='center', fontsize=7)
    ax.tick_params(axis='both', length=0, pad=5, labelcolor='#263743')
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cca_tasks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cca_tasks), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', length=0)
    colorbar_ax = fig.add_axes([0.86, 0.40, 0.018, 0.49])
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    colorbar.set_label('Canonical correlation r', labelpad=7)
    colorbar.outline.set_visible(False)
    fig.text(0.055, 0.035, 'Permutation tests, BH-FDR: * p < .05; ** p < .01; *** p < .001.',
             fontsize=10, color='#687F91')
    fig.savefig(CCA_OUTPUT / f'{filename}.png', dpi=600)
    plt.close(fig)
