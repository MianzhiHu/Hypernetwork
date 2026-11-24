"""Plot selected PLS loadings using a filled Shen atlas and pial surfaces."""

from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting, surface
from scipy.io import loadmat

from utils.PLS_Reader import get_pls_results, plot_outcome_results


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    result_dir = Path(r'C:\Users\User\OneDrive\Desktop\胡勉之\Texas A&M University\PLS\Result\Hypernetwork')
    result_file = 'PLS_single_task_neg_ss.mat'
    lv_file = 'PLS_single_task_neg_ss_lv_vals.mat'
    task = 'stopSignal'
    metric = 'neg'  # Node-strength table: 'overall', 'pos', or 'neg'.
    task_label = 'DD overall strength'
    lv = 1
    node_filter = 'bsr'       # Choose 'bsr' or 'corrected'.
    bsr_threshold = 3.0       # Keep both signs with |BSR| > this value.
    correction_method = 'fdr_bh'
    alpha = 0.05              # 'corrected': adjusted p-value < alpha.
    if node_filter not in ('bsr', 'corrected'):
        raise ValueError("node_filter must be 'bsr' or 'corrected'")

    output_dir = project_dir / 'Results' / 'fmri_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

    with (project_dir / 'Data' / 'fc_filtered.pkl').open('rb') as handle:
        fc_data = pickle.load(handle)
    fc_table = fc_data['fc_filtered'][task][metric]
    original_columns = fc_table.drop(columns='worker_id').columns.to_numpy(dtype=int)
    # Parser column names are ORIGINAL zero-based positions, not consecutive
    # positions after exclusions. Map them to the atlas's one-based ROI IDs.
    roi_ids = np.asarray(fc_data['roi_ids'])[original_columns]
    nodes = get_pls_results(
        result_dir / lv_file, result_dir / result_file, var_names=roi_ids,
        method=correction_method, p=alpha, LV=lv,
    )
    orientation = nodes.attrs['orientation']
    nodes = nodes.rename(columns={'Variable': 'roi_id', 'u1': 'brain_weight',
                                  'boot_ratio': 'bsr', 'significant': 'fdr_significant'})
    nodes.insert(0, 'original_node_index', original_columns)

    result = loadmat(result_dir / result_file, simplify_cells=True)['result']
    singular_values = np.asarray(result['s']).ravel()
    info = {
        'lv': lv,
        'permutation_p': float(np.asarray(result['perm_result']['sprob']).ravel()[lv - 1]),
        'cross_block_covariance_pct': float(100 * singular_values[lv - 1] ** 2 / np.sum(singular_values ** 2)),
        'n_subjects': int(np.sum(result['num_subj_lst'])),
        'n_permutations': int(result['perm_result']['num_perm']),
        'n_bootstraps': int(result['boot_result']['num_boot']),
        'orientation': orientation,
    }
    if len(fc_table) != info['n_subjects']:
        raise ValueError('PLS participant count does not match the filtered node-strength data')
    print(f"{task_label} LV{lv}: permutation p = {info['permutation_p']:.6f}; "
          f"cross-block covariance = {info['cross_block_covariance_pct']:.2f}%; "
          f"n = {info['n_subjects']}")
    if info['permutation_p'] >= 0.05:
        raise ValueError(f'LV{lv} is not significant at permutation p < 0.05')

    if node_filter == 'bsr':
        nodes['selected'] = np.isfinite(nodes['bsr']) & (nodes['bsr'].abs() > bsr_threshold)
        selection_label = f'|BSR| > {bsr_threshold:g} (uncorrected stability threshold)'
        selection_tag = f'bsr{bsr_threshold:g}'
    else:
        nodes['selected'] = np.isfinite(nodes['bsr']) & nodes['fdr_significant']
        selection_label = f'{correction_method}, adjusted p < {alpha:g} across {len(nodes)} tested nodes'
        selection_tag = f'{correction_method}_{alpha:g}'

    coord_path = project_dir / 'atlasTransform' / 'atlasTransform' / 'data' / 'shen_268' / 'Shen_atlas_MNI.txt'
    coords = pd.read_csv(coord_path, sep='\t')
    coords.columns = coords.columns.str.strip()
    node_coords = coords.set_index('Node #').loc[nodes['roi_id'], ['MNI x', 'MNI y', 'MNI z']].to_numpy()
    nodes[['mni_x', 'mni_y', 'mni_z']] = node_coords

    atlas_path = coord_path.parent / 'shen_1mm_268_parcellation.nii.gz'
    atlas = nib.load(atlas_path)
    atlas_labels = np.asarray(atlas.dataobj).astype(int)
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    selected_nodes = nodes.loc[nodes['selected']].copy()
    missing_roi_ids = np.setdiff1d(fc_data['roi_ids'], roi_ids)
    prefix = f'{Path(result_file).stem}_LV{lv}_{selection_tag}_pial_weights'
    nodes.to_csv(output_dir / f'{prefix}_all_nodes.csv', index=False)
    selected_nodes.to_csv(output_dir / f'{prefix}_selected_nodes.csv', index=False)
    pd.DataFrame({'roi_id': missing_roi_ids}).to_csv(output_dir / f'{prefix}_excluded_nodes.csv', index=False)
    pd.DataFrame([dict(info, task=task, metric=metric, node_filter=node_filter,
                       bsr_threshold=bsr_threshold, correction_method=correction_method, alpha=alpha,
                       node_p_value_method='Normal approximation from bootstrap ratios',
                       n_atlas_nodes=len(fc_data['roi_ids']), n_tested_nodes=len(nodes),
                       n_excluded_nodes=len(missing_roi_ids), n_selected_nodes=len(selected_nodes),
                       plotted_value='Oriented PLS loading (u1)', atlas_resolution_mm=1)]
                 ).to_csv(output_dir / f'{prefix}_summary.csv', index=False)

    if selected_nodes.empty:
        print(f'No nodes pass {selection_label}; CSV results saved to {output_dir}')
    else:
        # Restore all 268 positions: excluded and unselected parcels stay NaN.
        # BSR/corrected p selects parcels; the displayed values are u1 loadings.
        full_weights = np.full(len(fc_data['roi_ids']), np.nan)
        selected = nodes['selected'].to_numpy(dtype=bool)
        full_weights[original_columns[selected]] = nodes.loc[selected, 'brain_weight']
        np.save(output_dir / f'{prefix}_268_loadings.npy', full_weights)
        parcel_values = np.full(int(atlas_labels.max()) + 1, np.nan)
        parcel_values[0] = 0.0
        parcel_values[np.asarray(fc_data['roi_ids'], dtype=int)] = full_weights
        parcel_img = nib.Nifti1Image(parcel_values[atlas_labels].astype(np.float32), atlas.affine)
        nib.save(parcel_img, output_dir / f'{prefix}_parcel_loadings.nii.gz')

        # Follow the example: project the filled loading volume onto each pial
        # mesh, then mask NaN/zero samples before passing textures to plot_surf.
        textures = {}
        with warnings.catch_warnings():
            # Empty samples are expected over excluded/unselected parcels.
            warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
            for hemi in ('left', 'right'):
                texture = surface.vol_to_surf(parcel_img, fsaverage[f'pial_{hemi}'])
                textures[hemi] = np.ma.masked_where(~np.isfinite(texture) | (texture == 0), texture)

        vmax = float(np.max(np.abs(selected_nodes['brain_weight'])))
        fig, axes = plt.subplots(figsize=(10, 10), nrows=2, ncols=2,
                                 subplot_kw={'projection': '3d'})
        for row, hemi in enumerate(('left', 'right')):
            for column, view in enumerate(('medial', 'lateral')):
                plotting.plot_surf(
                    fsaverage[f'pial_{hemi}'], surf_map=textures[hemi],
                    bg_map=fsaverage[f'sulc_{hemi}'], hemi=hemi, view=view,
                    engine='matplotlib', cmap='RdBu_r', alpha=1, bg_on_data=True,
                    threshold=1e-12, vmin=-vmax, vmax=vmax,
                    colorbar=False, axes=axes[row, column], figure=fig,
                )
                axes[row, column].set_title(f'{hemi.capitalize()} {view}', fontsize=12,
                                            y=0.86 if row else 1.0)
        fig.subplots_adjust(bottom=0.08, top=0.90, left=0.05, right=0.95, wspace=0.001, hspace=0.15)
        colorbar_ax = fig.add_axes([0.2, 0.49, 0.6, 0.02])
        fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap='RdBu_r'),
                     cax=colorbar_ax, orientation='horizontal', label='PLS loading (u1)',
                     ticks=np.linspace(-vmax, vmax, 5), format='%.3f')
        fig.suptitle(f"{task_label} LV{lv} | p = {info['permutation_p']:.3f} | "
                     f"cross-block covariance {info['cross_block_covariance_pct']:.1f}%", fontsize=13)
        fig.text(0.5, 0.045, f'{len(selected_nodes)} selected / {len(nodes)} tested parcels; {selection_label}',
                 ha='center', fontsize=10)
        fig.text(0.5, 0.02, 'Color represents oriented PLS loadings; selection uses BSR or corrected p-values.',
                 ha='center', fontsize=9)
        fig.savefig(output_dir / f'{prefix}_surface.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    # The companion embedding plot uses the same LV sign orientation.
    raw_u = np.asarray(result['u'])[:, lv - 1]
    outcome_orientation = 1 if raw_u[np.nanargmax(np.abs(raw_u))] >= 0 else -1
    plot_outcome_results(
        str(result_dir), result_file, method=3, conditions=[task_label], LV_Vis=lv,
        BehavLabels=['GCCA 1', 'GCCA 2', 'GCCA 3', 'GCCA 4'],
        ylabel='Correlation with node-strength latent score',
        save_path=str(output_dir / f'{prefix}_embeddings_'),
        reverse_sign=(outcome_orientation != orientation),
    )
    print(f'Saved {len(selected_nodes)} selected nodes; excluded {len(missing_roi_ids)} unavailable ROIs. '
          f'Outputs: {output_dir}')
