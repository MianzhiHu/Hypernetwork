"""Participant-level RSA using raw full-data model embeddings and task FC."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from task_config import selected_config


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    embedding_dir = project_dir / 'Results' / 'Final_Models' / selected_config
    output_dir = project_dir / 'Results' / 'fmri_rsa' / selected_config / 'full_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_columns = ['embedding_1', 'embedding_2', 'embedding_3', 'embedding_4']
    n_permutations = 10_000
    rng = np.random.default_rng(42)
    tasks = {
        'cct': 'CCTHot',
        'dd': 'discountFix',
        'motor': 'motorSelectiveStop',
        'stopsignal': 'stopSignal',
        'twobytwo': 'twoByTwo',
    }

    with (project_dir / 'Data' / 'fc_filtered.pkl').open('rb') as handle:
        fc_data = pickle.load(handle)

    embeddings = {}
    for task in tasks:
        values = pd.DataFrame(np.load(embedding_dir / f'{task}_embeddings.npy'),
                              columns=embedding_columns)
        participants = pd.read_csv(embedding_dir / f'{task}_participant_map.csv')
        embeddings[task] = values.join(participants.set_index('participant_id')['worker_id']) \
                                .set_index('worker_id')
    fc_tables = {
        task: fc_data['fc_filtered'][bids_task]['neg'].set_index('worker_id')
        for task, bids_task in tasks.items()
    }

    records = []
    for embedding_task, embedding_table in embeddings.items():
        for fc_task, fc_table in fc_tables.items():
            workers = fc_table.index.intersection(embedding_table.index, sort=False)
            E = embedding_table.loc[workers].to_numpy(dtype=float)
            F = fc_table.loc[workers].to_numpy(dtype=float)
            # E = (E - E.mean(axis=0)) / E.std(axis=0)

            embedding_distances = pdist(E, metric='seuclidean')
            fc_distances = pdist(F, metric='seuclidean')

            # Spearman correlation: Pearson correlation of the ranked distances.
            # Rank once; participant permutations only reorder these same values.
            rho = spearmanr(embedding_distances, fc_distances).statistic
            embedding_ranks = rankdata(embedding_distances)
            embedding_ranks -= embedding_ranks.mean()
            embedding_ranks /= np.linalg.norm(embedding_ranks)
            fc_ranks = rankdata(fc_distances)
            fc_ranks -= fc_ranks.mean()
            fc_ranks /= np.linalg.norm(fc_ranks)

            rank_rdm = squareform(embedding_ranks)
            i, j = np.triu_indices(len(workers), k=1)
            null_rho = np.empty(n_permutations)
            for permutation in range(n_permutations):
                order = rng.permutation(len(workers))
                null_rho[permutation] = rank_rdm[order[i], order[j]] @ fc_ranks
            p_value = (1 + np.sum(np.abs(null_rho) >= abs(rho))) / (n_permutations + 1)

            records.append({
                'embedding_task': embedding_task, 'fc_task': fc_task,
                'embedding_columns': ','.join(embedding_columns),
                'n_participants': len(workers), 'n_pairs': len(fc_distances),
                'spearman_r': rho, 'p_permutation': p_value,
            })
            print(f'{embedding_task} embedding / {fc_task} FC: '
                  f'n = {len(workers)}, rho = {rho:.4f}, p = {p_value:.4f}', flush=True)

    results = pd.DataFrame(records)
    results['p_fdr'] = multipletests(results['p_permutation'], method='fdr_bh')[1]
    results['significant_fdr'] = results['p_fdr'] < 0.05
    rho_matrix = results.pivot(index='embedding_task', columns='fc_task', values='spearman_r')
    rho_matrix = rho_matrix.reindex(index=list(tasks), columns=list(tasks))

    prefix = '_'.join(embedding_columns)
    results.to_csv(output_dir / f'{prefix}_rsa_results.csv', index=False)
    rho_matrix.to_csv(output_dir / f'{prefix}_spearman_r.csv')
    print(rho_matrix.round(4).to_string())
    print(f'Saved RSA results to {output_dir}')
