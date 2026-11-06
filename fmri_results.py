"""Read PLS results and plot edges using a BSR or corrected-p-value filter."""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy.io import loadmat
from utils.PLS_Reader import get_pls_results, plot_outcome_results

pls_task_name_mapping = {
    'CCTHot': 'cct',
    'discountFix': 'dd',
    'motorSelectiveStop': 'motor',
    'stopSignal': 'ss',
    'twoByTwo': 'twobytwo',
}

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    result_dir = Path(r'C:\Users\User\OneDrive\Desktop\胡勉之\Texas A&M University\PLS\Result\Hypernetwork')
    task = 'twoByTwo'   # 'CCTHot', 'discountFix', 'motorSelectiveStop', 'stopSignal', 'twoByTwo'
    mapped_task = pls_task_name_mapping[task]
    result_file = f'PLS_single_task_edge_{mapped_task}.mat'
    lv_file = f'PLS_single_task_edge_{mapped_task}_lv_vals.mat'
    lv = 1
    edge_filter = 'bsr'  # Choose 'bsr' or 'corrected'.
    bsr_threshold = 3.0        # 'bsr': keep BOTH signs where |BSR| > this value or 'corrected': keep adjusted p-values < alpha.
    correction_method = 'fdr_bh'
    alpha = 0.05
    output_dir = project_dir / 'Results' / 'fmri_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})

    with (project_dir / 'Data' / 'fc_filtered.pkl').open('rb') as handle:
        fc_data = pickle.load(handle)
    fc_table = fc_data['fc_filtered'][task]['edges']
    original_columns = fc_table.drop(columns='worker_id').columns.to_numpy(dtype=int)

    edges = get_pls_results(result_dir / lv_file, result_dir / result_file, var_names=original_columns,
                            method=correction_method, p=alpha, LV=lv)

    # The updated reader's attrs contain sign orientation, not LV metadata.
    result = loadmat(result_dir / result_file, simplify_cells=True)['result']
    singular_values = np.asarray(result['s']).ravel()
    info = {
        'lv': lv,
        'p_value': float(np.asarray(result['perm_result']['sprob']).ravel()[lv - 1]),
        'cross_block_covariance_pct': float(100 * singular_values[lv - 1] ** 2 / np.sum(singular_values ** 2)),
        'n_subjects': int(np.sum(result['num_subj_lst'])),
        'n_permutations': int(result['perm_result']['num_perm']),
        'n_bootstraps': int(result['boot_result']['num_boot']),
        'orientation': edges.attrs['orientation'],
    }
    print(f"{task} LV{lv}: permutation p = {info['p_value']:.6f}; "
          f"cross-block covariance = {info['cross_block_covariance_pct']:.2f}%; "
          f"n = {info['n_subjects']}")
    if info['p_value'] >= 0.05:
        raise ValueError(f'LV{lv} is not significant at permutation p < 0.05')
    if len(fc_table) != info['n_subjects']:
        raise ValueError('PLS participant count does not match the filtered task FC data')

    # Map retained edge labels back to the ORIGINAL upper triangle before plotting.
    edge_indices = edges.pop('Variable').to_numpy(dtype=int)
    roi_pairs = np.asarray(fc_data['edge_indices'])[edge_indices] - fc_data['edge_index_base'] + 1
    edges.insert(0, 'original_edge_index', edge_indices)
    edges.insert(1, 'roi_i', roi_pairs[:, 0])
    edges.insert(2, 'roi_j', roi_pairs[:, 1])
    edges = edges.rename(columns={'u1': 'brain_weight', 'boot_ratio': 'bsr',
                                  'significant': 'fdr_significant'})
    if edge_filter == 'bsr':
        edges['selected'] = np.isfinite(edges['bsr']) & (edges['bsr'].abs() > bsr_threshold)
        selection_label = f'|BSR| > {bsr_threshold:g} (uncorrected stability threshold)'
        selection_tag = f'bsr{bsr_threshold:g}'
    else:
        edges['selected'] = np.isfinite(edges['bsr']) & edges['fdr_significant']
        selection_label = f'{correction_method}, adjusted p < {alpha:g} across {len(edges):,} edges'
        selection_tag = f'{correction_method}_{alpha:g}'
    selected_edges = edges.loc[edges['selected']].copy()
    prefix = f'{task}_LV{lv}_{selection_tag}'
    edges.to_csv(output_dir / f'{prefix}_all_edges.csv', index=False)
    selected_edges.to_csv(output_dir / f'{prefix}_selected_edges.csv', index=False)
    pd.DataFrame([dict(info, edge_filter=edge_filter, bsr_threshold=bsr_threshold,
                       correction_method=correction_method, alpha=alpha,
                       edge_p_value_method='Normal approximation from bootstrap ratios',
                       n_retained_edges=len(edges), n_selected_edges=len(selected_edges))]
                 ).to_csv(output_dir / f'{prefix}_summary.csv', index=False)
    if selected_edges.empty:
        print(f'No edges pass {selection_label}; CSV results saved to {output_dir}')
    else:
        coord_path = project_dir / 'atlasTransform' / 'atlasTransform' / 'data' / 'shen_268' / 'Shen_atlas_MNI.txt'
        coords = pd.read_csv(coord_path, sep='\t')
        coords.columns = coords.columns.str.strip()
        coords = coords.set_index('Node #').loc[fc_data['roi_ids'], ['MNI x', 'MNI y', 'MNI z']].to_numpy()
        adjacency = np.zeros((len(coords), len(coords)))
        i = selected_edges['roi_i'].to_numpy(dtype=int) - 1
        j = selected_edges['roi_j'].to_numpy(dtype=int) - 1
        adjacency[i, j] = selected_edges['bsr']
        adjacency[j, i] = selected_edges['bsr']
        node_sizes = np.where(np.any(adjacency != 0, axis=0), 5, 0)
        vmax = float(np.max(np.abs(adjacency)))

        fig = plt.figure(figsize=(12, 5))
        display = plotting.plot_connectome(
            adjacency, coords, figure=fig, display_mode='lr',
            node_color='#555555', node_size=node_sizes,
            edge_cmap='RdBu_r', edge_vmin=-vmax, edge_vmax=vmax, edge_threshold=bsr_threshold,
            edge_kwargs={'linewidth': 0.6, 'alpha': 0.45},
            colorbar=True, annotate=True,
        )
        fig.suptitle(f"{task} LV{lv} | p = {info['p_value']:.3f} | "
                     f"cross-block covariance {info['cross_block_covariance_pct']:.1f}%", fontsize=13)
        fig.text(0.5, 0.06, f'{len(selected_edges):,} edges; {selection_label}', ha='center', fontsize=10)
        fig.text(0.5, 0.015, 'Edge p-values use a normal approximation to BSRs. '
                 'Red/blue = positive/negative LV weight direction, not FC sign.', ha='center', fontsize=9)
        fig.savefig(output_dir / f'{prefix}_connectome.png', dpi=300, bbox_inches='tight')
        display.close()

    # Match the outcome plotter's orientation to the reader, including sign changes.
    raw_u = np.asarray(result['u'])[:, lv - 1]
    outcome_orientation = 1 if raw_u[np.nanargmax(np.abs(raw_u))] >= 0 else -1
    plot_outcome_results(
        str(result_dir), result_file, method=3, conditions=[task], LV_Vis=lv,
        BehavLabels=['GCCA 1', 'GCCA 2', 'GCCA 3', 'GCCA 4'],
        ylabel='Correlation with FC latent score',
        save_path=str(output_dir / f'{prefix}_embeddings_'),
        reverse_sign=(outcome_orientation != info['orientation']),
    )
    print(f'Saved {len(selected_edges)} edges using {selection_label}; outputs: {output_dir}')
