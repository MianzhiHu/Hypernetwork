from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import GCCA, CCA
from cca_zoo.model_selection import permutation_test_significance
from CKA.CKA import CKA
from joblib import Parallel, delayed, parallel_config
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from HyperNetwork import evaluate_cka_permutation
from task_config import selected_config
from utils.permcca import permcca
from permcca.inference import permutation_inference, can_corr


# Settings: only the exported main model is evaluated here.
# All: layers_1_dims_8_rank_full_emb_2_nonlinear_nodes_8_reg_0.0001 (relatively low stability for cct, twobytwo, and motor; relatively bad cct perf; fine distinguish)
# No embedding 2: layers_1_dims_32_rank_full_emb_4_nonlinear_nodes_4_reg_0.01 (fine stability > .8; better cct, but worse dd and stop-signal; weaker distinguish)
# No reg: layers_1_dims_16_rank_full_emb_2_nonlinear_nodes_16_reg_0.0 (bad stability for cct and twobytwo; relatively good stop-signal, average cct and dd; strong distinguish)
# No reg or embedding 2: layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_16_reg_0.0 (great stability; good performance; moderate distinguish)
output = Path('./Results/Main_Model_Evaluation') / selected_config
n_splits = 10
n_permutation = 10000
n_jobs = 32
permutation_seed = 42
permutation_alpha = 0.05
task_names = ['cct', 'dd', 'motor', 'stopsignal', 'twobytwo']

# Saved predictive results (independent of the embedding/permutation analysis below).
BASE = Path(__file__).resolve().parent
hypernn_root = BASE / 'Results/HyperNN_Grid_Search'
cognitive_root = BASE / 'Results/Final_Models/CogModels'
cognitive_group_root = BASE / 'Results/Final_Models/CogModels_Group'
comparison_output = BASE / 'Results/Cognitive_Model_Evaluation'
basic_nn_results = BASE / 'Results/Final_Models/BasicNN/layers_1_dims_8/test_by_split.csv'
hyper_nn_results = BASE / 'Results/HyperNN_Grid_Search_Evaluation/test_results_all.csv'

if __name__ == '__main__':
    # ==================================================================================================================
    # Test Performance Evaluation: Cognitive Models, BasicNN, and HyperNN from 10-fold CV
    # ==================================================================================================================
    # Test performance of individually trained cognitive models
    cognitive_records = []
    for task in ['cct', 'dd', 'stopsignal', 'motor', 'twobytwo']:
        paths_ind = sorted((cognitive_root / task).glob('*_eval_*.csv'))
        paths_group = sorted((cognitive_group_root / task).glob('*_eval_*.csv'))
        for family, paths in [('cognitive_individual', paths_ind), ('cognitive_group', paths_group)]:
            for path in paths:
                model, split = path.stem.rsplit('_eval_', 1)
                family = family
                table = pd.read_csv(path)
                values = table[['total_nll', 'n_trials', 'n_correct']]
                if (not np.isfinite(values).all().all() or (table.n_trials <= 0).any()
                        or (table.total_nll < 0).any() or (table.n_correct < 0).any()
                    or (table.n_correct > table.n_trials).any()):
                    raise ValueError(f'Invalid saved metrics: {path}')
                cognitive_records.append(dict(task=task, model=model, random_split=int(split), model_family=family,
                    n_participants=len(table), n_test_trials=int(table.n_trials.sum()),
                    test_total_nll=table.total_nll.sum(), n_test_correct=int(table.n_correct.sum())))
    cognitive_by_split = pd.DataFrame(cognitive_records)
    for key, group in cognitive_by_split.groupby(['task', 'model', 'model_family']):
        if sorted(group.random_split.tolist()) != list(range(n_splits)):
            raise ValueError(f'{key}: expected exactly one evaluation file per fold 0–9')

    # Pool trials within a fold, then give each fold equal weight.
    cognitive_by_split['test_nll'] = cognitive_by_split.test_total_nll / cognitive_by_split.n_test_trials
    cognitive_by_split['test_accuracy'] = cognitive_by_split.n_test_correct / cognitive_by_split.n_test_trials
    cognitive_by_model = cognitive_by_split.groupby(['task', 'model', 'model_family'], as_index=False).agg(
        test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
        test_nll_sd=('test_nll', 'std'), n_splits=('random_split', 'nunique'),
        n_test_trials=('n_test_trials', 'sum'))
    comparison_output.mkdir(parents=True, exist_ok=True)
    cognitive_by_split.to_csv(comparison_output / 'cognitive_test_by_split.csv', index=False)
    cognitive_by_model.to_csv(comparison_output / 'cognitive_test_by_model.csv', index=False)
    comparison = [cognitive_by_model]
    fold_comparison = [cognitive_by_split]

    # Evaluate the saved BasicNN and HyperNN results, and compare with the cognitive models.
    for family, path in [('BasicNN', basic_nn_results), ('HyperNN', hyper_nn_results)]:
        neural = pd.read_csv(path)
        neural = neural.loc[neural.task.isin(['cct', 'dd', 'stopsignal', 'motor', 'twobytwo'])].copy()
        neural = neural.rename(columns={'test_n_trials': 'n_test_trials'})  # Rename for consistency with cognitive model results.
        if family == 'HyperNN':
            neural = neural.loc[neural.config_id.eq(selected_config)].copy()
        counts = cognitive_by_split.merge(neural[['task', 'random_split', 'n_test_trials']], on=['task', 'random_split'], suffixes=('_cognitive', '_neural'))
        fold_comparison.append(neural.rename(columns={'config_id': 'model'}).assign(model_family=family))
        summary = neural.groupby(['task', 'config_id'], as_index=False).agg(
            test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
            test_nll_sd=('test_nll', 'std'), n_splits=('random_split', 'nunique'),
            n_test_trials=('n_test_trials', 'sum')).rename(columns={'config_id': 'model'})
        summary['model_family'] = family
        comparison.append(summary)
    model_comparison = pd.concat(comparison, ignore_index=True)

    # Long-format table: one row per model, task and held-out fold; no fold averaging.
    model_fold_results = pd.concat(fold_comparison, ignore_index=True)
    model_fold_results = model_fold_results[['model_family', 'model', 'task', 'random_split',
                                             'test_nll', 'test_accuracy', 'n_test_trials']]
    model_fold_results.to_csv(comparison_output / 'test_performance_by_fold.csv', index=False)
    print(f'Saved {len(model_fold_results)} model/task/fold rows')
    model_comparison = model_comparison.sort_values(['task', 'test_nll']).reset_index(drop=True)
    model_comparison.to_csv(comparison_output / 'test_performance_by_model.csv', index=False)
    print(model_comparison.to_string(index=False))

    # ==================================================================================================================
    # Embedding Stability Evaluation: HyperNN embeddings across 10 random splits
    # ==================================================================================================================
    participants = pd.read_csv(hypernn_root / 'participant_map.csv').sort_values('participant_id').reset_index(drop=True)
    dimensions = int(selected_config.split('_')[7])
    split_pairs = list(combinations(range(n_splits), 2))

    # Embedding Stability
    stability_tests = []
    for task in task_names:
        raw = [np.load(hypernn_root / selected_config / f'random_split_{split}' / f'{task}_embeddings.npy').astype(float)
               for split in range(n_splits)]
        views = [StandardScaler().fit_transform(v) for v in raw]
        gcca = GCCA(n_components=dimensions).fit(views)
        scores = gcca.transform(views)
        result = permutation_test_significance(GCCA(n_components=dimensions), views, n_permutations=n_permutation,
                                               random_state=permutation_seed, n_jobs=n_jobs)
        observed_cca = result.correlations[0]
        cca_p = result.p_values[0]
        observed_cka = np.array([CKA().linear_CKA(raw[a], raw[b]) for a, b in split_pairs])

        # Permutation for CKA
        cka_permutation_results = []
        cka_permutation_pairwise = []
        rng = np.random.default_rng(permutation_seed)
        for _ in range(n_permutation):
            shuffled_views = [raw[0]] + [rng.permutation(v) for v in raw[1:]]
            cka_values = np.array([CKA().linear_CKA(shuffled_views[a], shuffled_views[b]) for a, b in split_pairs])
            cka_permutation_pairwise.append(cka_values)
            cka_permutation_results.append(cka_values.mean())
        cka_permutation_mean_null = np.array(cka_permutation_results).mean()
        cka_permutation_pairwise_null = np.stack(cka_permutation_pairwise).mean(axis=0)
        cka_permutation_p = (1 + (np.array(cka_permutation_results) >= observed_cka.mean()).sum()) / (n_permutation + 1)
        cka_permutation_pairwise_p = (1 + (np.stack(cka_permutation_pairwise) >= observed_cka[None, :]).sum(axis=0)) / (n_permutation + 1)
        stability_tests.append(dict(name=task, params=gcca.get_params(), views=views, cka_views=raw,
                                    observed_cca=observed_cca, cca_p=cca_p, cca_null=result.null_correlations[:, 0],
                                    observed_cka=observed_cka.mean(), cka_p=cka_permutation_p, cka_null=cka_permutation_mean_null,
                                    pair_cka=observed_cka, pair_cka_p=cka_permutation_pairwise_p, pair_cka_null=cka_permutation_pairwise_null))

    # Save stability results for each task
    stability_output = output / 'Stability'
    stability_output.mkdir(parents=True, exist_ok=True)
    stability_tests_df = pd.DataFrame(stability_tests)
    stability_tests_df.to_csv(stability_output / 'stability_tests.csv', index=False)

    # ==================================================================================================================
    # Cross-task GCCA: Full-data embeddings
    # ==================================================================================================================
    full_embedding_path = output.parent.parent / 'Final_Models' / selected_config
    full_embeddings = [np.load(full_embedding_path / f'{task}_embeddings.npy').astype(float) for task in task_names]

    # Cross-task GCCA on the full-data embeddings
    views = [StandardScaler().fit_transform(v) for v in full_embeddings]
    gcca = GCCA(n_components=4).fit(views)
    scores = gcca.transform(views)
    result = permutation_test_significance(GCCA(n_components=4), views, n_permutations=n_permutation,
                                           random_state=permutation_seed, n_jobs=n_jobs)
    observed_cca = result.correlations
    cca_p = result.p_values
    null = result.null_correlations[:, 0]
    pd.DataFrame({'null_cca': null}).to_csv(output / 'cross_task_gcca_null.csv', index=False)

    # Robustness check: GCCA on the consensus embeddings from 10-fold CV (should be similar to the full-data GCCA above).
    cv_embedding_path = output.parent.parent / 'HyperNN_Grid_Search_Evaluation' / 'Cross_Task_Convergence' / 'Consensus_Embeddings'
    cv_embeddings = [pd.read_csv(cv_embedding_path / f'{selected_config}_{task}.csv') for task in task_names]
    embedding_cols = [c for c in cv_embeddings[0].columns if c.startswith('gcca_')]

    # Cross-task GCCA on the consensus embeddings
    views_cv = [StandardScaler().fit_transform(v) for v in [consensus[embedding_cols].to_numpy() for consensus in cv_embeddings]]
    gcca_cv = GCCA(n_components=4).fit(views_cv)
    scores_cv = gcca_cv.transform(views_cv)
    result_cv = permutation_test_significance(GCCA(n_components=4), views_cv, n_permutations=n_permutation,
                                              random_state=permutation_seed, n_jobs=n_jobs)
    observed_cca_cv = result_cv.correlations
    cca_p_cv = result_cv.p_values


    cross_task_gcca = [dict(comp=i, params=gcca.get_params(), observed_cca=observed_cca[i], cca_p=cca_p[i],
                            observed_cca_cv=observed_cca_cv[i], cca_p_cv=cca_p_cv[i]) for i in range(len(observed_cca))]
    cross_task_gcca_df = pd.DataFrame(cross_task_gcca)
    cross_task_gcca_df.to_csv(output / 'cross_task_gcca_results.csv', index=False)

    # Create cross-task consensus results
    cross_task_consensus = np.mean([StandardScaler().fit_transform(s) for s in scores], axis=0)

    # Save both the aligned embeddings and the consensus embeddings for further analysis
    np.save(full_embedding_path / 'cross_task_consensus_embeddings.npy', cross_task_consensus)
    for task, score in zip(task_names, scores):
        np.save(full_embedding_path / f'{task}_cross_task_gcca_embeddings.npy', score)
    print(f'Cross-task GCCA results saved to {output / "cross_task_gcca_results.csv"}')

    # ==================================================================================================================
    # Pairwise CCA across tasks using the consensus embeddings
    # ==================================================================================================================
    pairwise_cca_results = []
    for a, b in combinations(range(len(task_names)), 2):
        views = [StandardScaler().fit_transform(full_embeddings[a]), StandardScaler().fit_transform(full_embeddings[b])]
        views_cv = [StandardScaler().fit_transform(cv_embeddings[a][embedding_cols].to_numpy()), StandardScaler().fit_transform(cv_embeddings[b][embedding_cols].to_numpy())]
        cca = CCA(n_components=4).fit(views)
        cca_cv = CCA(n_components=4).fit(views_cv)
        scores = cca.transform(views)
        result = permutation_test_significance(CCA(n_components=4), views, n_permutations=n_permutation, random_state=permutation_seed, n_jobs=n_jobs)
        result_cv = permutation_test_significance(CCA(n_components=4), views_cv, n_permutations=n_permutation, random_state=permutation_seed, n_jobs=n_jobs)
        observed_cca = result.correlations
        observed_cca_cv = result_cv.correlations
        cca_p = result.p_values
        cca_p_cv = result_cv.p_values
        perm_p = permcca(views[0], views[1], n_perm=n_permutation, seed=permutation_seed)
        perm_p_cv = permcca(views_cv[0], views_cv[1], n_perm=n_permutation, seed=permutation_seed)
        pairwise_cca_results.append(dict(name=f'{task_names[a]}__{task_names[b]}', task_a=task_names[a], task_b=task_names[b],
                                         observed_cca=observed_cca[0], cca_p=cca_p[0], cca_p_perm=perm_p,
                                         observed_cca_cv=observed_cca_cv[0], cca_p_cv=cca_p_cv[0], cca_p_cv_perm=perm_p_cv,
                                         views_a=views[0], views_b=views[1], views_cv_a=views_cv[0], views_cv_b=views_cv[1]))

        # Create pairwise cross-task consensus results
        pairwise_consensus = np.mean([StandardScaler().fit_transform(s) for s in scores], axis=0)

        # Save both the aligned embeddings and the consensus embeddings for further analysis
        np.save(full_embedding_path / f'{task_names[a]}__{task_names[b]}_pairwise_consensus_embeddings.npy', pairwise_consensus)
        for task, score in zip([task_names[a], task_names[b]], scores):
            np.save(full_embedding_path / f'{task}_{task_names[a]}__{task_names[b]}_pairwise_cca_embeddings.npy', score)

    pairwise_cca_results_df = pd.DataFrame(pairwise_cca_results)
    pairwise_cca_results_df['cca_p_adj'] = multipletests(pairwise_cca_results_df['cca_p'], alpha=permutation_alpha, method='fdr_bh')[1]
    pairwise_cca_results_df['cca_p_cv_adj'] = multipletests(pairwise_cca_results_df['cca_p_cv'], alpha=permutation_alpha, method='fdr_bh')[1]
    pairwise_cca_results_df.to_csv(output / 'pairwise_cca_results.csv', index=False)