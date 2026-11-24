"""Plot significant cognitive parameters from cognitive-parameter/embedding PLS."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils.PLS_Reader import get_pls_results


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    result_dir = Path(r'C:\Users\User\OneDrive\Desktop\胡勉之\Texas A&M University\PLS\Result\Hypernetwork')
    task = 'dd'
    parameter_filter = 'corrected'  # 'corrected' or 'bsr'.
    alpha = 0.05
    bsr_threshold = 3.0
    correction_method = 'fdr_bh'
    result_file = result_dir / f'PLS_behav{task}.mat'
    lv_file = result_dir / f'PLS_behav{task}_lv_vals.mat'
    output_dir = project_dir / 'Results' / 'Embedding_Interpretation_PLS'
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    parameter_names = pd.read_csv(
        project_dir / 'Data' / 'PLS_data' / f'{task}_cognitive_parameters.csv', nrows=0
    ).columns.tolist()
    result = loadmat(result_file, simplify_cells=True)['result']
    permutation_p = np.asarray(result['perm_result']['sprob']).ravel()
    singular_values = np.asarray(result['s']).ravel()
    covariance_pct = 100 * singular_values ** 2 / np.sum(singular_values ** 2)
    n = int(np.sum(result['num_subj_lst']))
    n_permutations = int(result['perm_result']['num_perm'])
    task_labels = {
        'cct': 'Columbia card task', 'dd': 'Delayed discounting',
        'stopsignal': 'Stop signal', 'motor': 'Motor selective stop',
        'twobytwo': 'Two-by-two task switching',
    }
    tables = []

    for lv, p_value in enumerate(permutation_p, start=1):
        # The reader orients the largest absolute cognitive weight positive.
        # Its corrected p-values use a normal approximation to BSRs, within this LV.
        parameters = get_pls_results(
            lv_file, result_file, var_names=parameter_names,
            method=correction_method, p=alpha, LV=lv,
        )
        parameters.insert(0, 'LV', lv)
        parameters['lv_permutation_p'] = p_value
        parameters['cross_block_covariance_pct'] = covariance_pct[lv - 1]
        parameters['n'] = n
        parameters['orientation'] = parameters.attrs['orientation']
        if parameter_filter == 'bsr':
            parameters['selected'] = parameters['boot_ratio'].abs() > bsr_threshold
            selection_label = f'|BSR| > {bsr_threshold:g}'
            selection_tag = f'bsr{bsr_threshold:g}'
        else:
            parameters['selected'] = parameters['significant']
            selection_label = f'BH-adjusted BSR-derived p < {alpha:g} within LV'
            selection_tag = f'{correction_method}_{alpha:g}'
        parameters['selected'] &= p_value < alpha
        tables.append(parameters)

        selected = parameters.loc[parameters['selected']].sort_values(
            'u1', key=abs, ascending=False
        )
        print(f'{task} LV{lv}: permutation p={p_value:.6g}; '
              f'{len(selected)}/{len(parameters)} parameters selected')
        if selected.empty:
            continue

        p_label = f'p < {1 / n_permutations:g}' if p_value == 0 else f'p = {p_value:.4g}'
        fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.5 * len(selected) + 1.4)),
                               layout='constrained')
        colors = np.where(selected['u1'] > 0, '#B64B49', '#4778A8')
        ax.barh(selected['Variable'], selected['u1'], color=colors, height=0.55)
        ax.invert_yaxis()
        ax.axvline(0, color='0.35', linewidth=0.7)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel(f'Cognitive parameter weight\n{selection_label}')
        ax.set_title(
            f'{task_labels[task]} | LV{lv}\n'
            f'Permutation {p_label} | Cross-block covariance {covariance_pct[lv - 1]:.1f}% | n = {n}',
            fontsize=9,
        )
        prefix = output_dir / f'{task}_LV{lv}_{selection_tag}_cognitive_parameters'
        fig.savefig(f'{prefix}.png', dpi=600, bbox_inches='tight')
        fig.savefig(f'{prefix}.svg', bbox_inches='tight')
        fig.savefig(f'{prefix}.pdf', bbox_inches='tight')
        plt.close(fig)

    all_parameters = pd.concat(tables, ignore_index=True)
    all_parameters.to_csv(output_dir / f'{task}_{selection_tag}_all_parameters.csv', index=False)
    all_parameters.loc[all_parameters['selected']].to_csv(
        output_dir / f'{task}_{selection_tag}_significant_parameters.csv', index=False
    )
    print(f'PLS cognitive-parameter figures and tables saved to {output_dir}')
