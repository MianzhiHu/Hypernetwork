from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import CCA

# ======================================================================================================================
# Settings
# ======================================================================================================================
BASE = Path(__file__).resolve().parent
config_id = 'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8'
root = BASE / 'Results/HyperNN_Grid_Search'
full_data = BASE / 'Final_Models' / config_id / 'full_data'
gcca_root = BASE / 'Results/HyperNN_Grid_Search_Evaluation/Cross_Task_Convergence/Consensus_Embeddings'
output = BASE / 'Results/Full_Data_Procrustes_Comparison' / config_id
tasks = ['ant', 'cct', 'dd', 'motor', 'stroop', 'dpx', 'stopsignal', 'twobytwo']
n_splits = 10
max_iterations = 1000
tolerance = 1e-10

if __name__ == '__main__':
    output.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(root / 'participant_map.csv').sort_values('participant_id')
    if (participants.worker_id.duplicated().any() or
            not np.array_equal(participants.participant_id, np.arange(len(participants)))):
        raise ValueError('Invalid grid-search participant map')
    records = []
    component_records = []

    for task in tasks:
        # ==============================================================================================================
        # Generalized orthogonal Procrustes across folds only. Full-data embeddings are not the alignment target.
        # Center each view and give it unit Frobenius norm (one scale per view, not one per dimension).
        # Rotations/reflections preserve the within-view geometry; no shearing or dimension-specific rescaling.
        # ==============================================================================================================
        views = np.stack([np.load(root / config_id / f'random_split_{split}' / f'{task}_embeddings.npy')
                          for split in range(n_splits)]).astype(float)
        if views.shape[1] != len(participants) or not np.isfinite(views).all():
            raise ValueError(f'Invalid fold embeddings: {task}')
        views -= views.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(views, axis=(1, 2), keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError(f'Degenerate fold embeddings: {task}')
        views /= norms

        # Try each fold as the initial reference; choose the smallest within-fold alignment error.
        # This choice never uses correspondence with the full-data model.
        best_error = np.inf
        for initial_fold in range(n_splits):
            reference = views[initial_fold].copy()
            previous_error = np.inf
            converged = False
            for iteration in range(1, max_iterations + 1):
                rotations = np.stack([orthogonal_procrustes(view, reference)[0] for view in views])
                aligned = np.stack([view @ rotation for view, rotation in zip(views, rotations)])
                consensus = aligned.mean(axis=0)
                error = float(np.sum((aligned - consensus) ** 2))
                if abs(previous_error - error) < tolerance:
                    converged = True
                    break
                previous_error = error
                reference = consensus
            if error < best_error:
                best_error = error
                best_consensus = consensus.copy()
                best_aligned = aligned.copy()
                best_rotations = rotations.copy()
                best_iteration = iteration
                best_initial_fold = initial_fold
                best_converged = converged
        if not best_converged:
            raise RuntimeError(f'Procrustes did not converge for {task}; increase max_iterations')
        consensus_table = participants.copy()
        for k in range(best_consensus.shape[1]):
            consensus_table[f'procrustes_{k+1}'] = best_consensus[:, k]
        consensus_table.to_csv(output / f'{task}_consensus.csv', index=False)
        np.savez(output / f'{task}_alignment.npz', aligned=best_aligned, rotations=best_rotations,
                 participant_id=participants.participant_id.to_numpy(),
                 worker_id=participants.worker_id.to_numpy(dtype=str))

        # ==============================================================================================================
        # Compare both consensuses with the same full-data embeddings, using the same CCA calculation.
        # These are descriptive in-sample correlations, not independent validation or original-axis correlations.
        # ==============================================================================================================
        full_participants = pd.read_csv(full_data / f'{task}_participant_map.csv')
        full_embeddings = np.load(full_data / f'{task}_embeddings.npy')
        full_columns = [f'full_{k+1}' for k in range(full_embeddings.shape[1])]
        for k, column in enumerate(full_columns):
            full_participants[column] = full_embeddings[full_participants.participant_id.to_numpy(dtype=int), k]
        gcca = pd.read_csv(gcca_root / f'{config_id}_{task}.csv')
        gcca_columns = [c for c in gcca if c.startswith('gcca_')]
        table = consensus_table.merge(full_participants, on=['participant_id', 'worker_id'], how='left', validate='one_to_one')
        table = table.merge(gcca, on=['participant_id', 'worker_id'], how='left', validate='one_to_one')
        if len(full_participants) != len(participants) or len(gcca) != len(participants) or table.isna().any().any():
            raise ValueError(f'Mismatched participants or missing scores: {task}')
        procrustes_columns = [c for c in table if c.startswith('procrustes_')]
        matrices = {'GCCA': table[gcca_columns].to_numpy(), 'Procrustes': table[procrustes_columns].to_numpy()}
        full_view = StandardScaler().fit_transform(table[full_columns])
        dimensions = min(np.linalg.matrix_rank(full_view),
                         *[np.linalg.matrix_rank(StandardScaler().fit_transform(v)) for v in matrices.values()])
        if dimensions < 1:
            raise ValueError(f'Degenerate comparison: {task}')
        record = {'task': task, 'config_id': config_id, 'n_participants': len(table),
                  'n_components': dimensions, 'procrustes_error': best_error,
                  'procrustes_iterations': best_iteration, 'initial_fold': best_initial_fold}
        for method, matrix in matrices.items():
            cca_views = [full_view, StandardScaler().fit_transform(matrix)]
            cca = CCA(latent_dimensions=int(dimensions)).fit(cca_views)
            r = cca.score(cca_views)
            record[f'{method.lower()}_first_r'] = float(r[0])
            record[f'{method.lower()}_mean_r'] = float(np.mean(r))
            for k, value in enumerate(r, start=1):
                component_records.append({'task': task, 'method': method, 'component': k, 'cca_r': float(value)})
        record['difference_first_r'] = record['procrustes_first_r'] - record['gcca_first_r']
        record['difference_mean_r'] = record['procrustes_mean_r'] - record['gcca_mean_r']
        records.append(record)
        print(f"{task}: first r GCCA={record['gcca_first_r']:.4f}, Procrustes={record['procrustes_first_r']:.4f}; "
              f"mean r GCCA={record['gcca_mean_r']:.4f}, Procrustes={record['procrustes_mean_r']:.4f}", flush=True)

    summary = pd.DataFrame(records)
    summary.to_csv(output / 'comparison_by_task.csv', index=False)
    pd.DataFrame(component_records).to_csv(output / 'cca_by_component.csv', index=False)
    metrics = ['gcca_first_r', 'procrustes_first_r', 'difference_first_r',
               'gcca_mean_r', 'procrustes_mean_r', 'difference_mean_r']
    summary[metrics].mean().rename_axis('metric').reset_index(name='task_mean').to_csv(output / 'overall_comparison.csv', index=False)
    print('\nAcross-task means:\n', summary[metrics].mean().to_string())
