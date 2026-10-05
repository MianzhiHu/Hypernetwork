from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import GCCA, CCA
from CKA.CKA import CKA
from joblib import Parallel, delayed, parallel_config
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from HyperNetwork import evaluate_permutation

# Settings: only the exported main model is evaluated here.
selected_config = 'layers_1_dims_16_rank_full_emb_2_nonlinear_nodes_16_reg_0.0'
root = Path('./Final_Models') / selected_config
output = Path('./Results/Main_Model_Evaluation') / selected_config
n_splits = 10
n_permutation = 10000
n_jobs = 8
permutation_seed = 42
permutation_alpha = 0.05
task_names = ['ant', 'cct', 'dd', 'motor', 'stroop', 'dpx', 'stopsignal', 'twobytwo']

# Saved predictive results (independent of the embedding/permutation analysis below).
BASE = Path(__file__).resolve().parent
run_cognitive_results = True
run_embedding_permutations = False
cognitive_root = BASE / 'Results/Final_Models/CogModels'
comparison_output = BASE / 'Results/Cognitive_Model_Evaluation'
basic_nn_results = BASE / 'Results/Final_Models/BasicNN/layers_1_dims_64/test_by_split.csv'
hyper_nn_results = BASE / 'Results/HyperNN_Grid_Search_Evaluation/test_results_all.csv'

if __name__ == '__main__':
    cognitive_records = []
    for task in ['cct', 'dd', 'stopsignal', 'motor']:
        paths = sorted((cognitive_root / task).glob('*_eval_*.csv'))
        if not paths:
            raise FileNotFoundError(f'No saved evaluation files for {task}')
        for path in paths:
            model, split = path.stem.rsplit('_eval_', 1)
            table = pd.read_csv(path)
            if table.empty or table.participant_id.duplicated().any():
                raise ValueError(f'Empty or duplicate participant results: {path}')
            values = table[['total_nll', 'n_trials', 'n_correct']]
            if (not np.isfinite(values).all().all() or (table.n_trials <= 0).any()
                    or (table.total_nll < 0).any() or (table.n_correct < 0).any()
                    or (table.n_correct > table.n_trials).any()):
                raise ValueError(f'Invalid saved metrics: {path}')
            if not np.allclose(table.mean_nll, table.total_nll / table.n_trials):
                raise ValueError(f'Mean and total NLL disagree: {path}')
            cognitive_records.append(dict(task=task, model=model, random_split=int(split),
                n_participants=len(table), n_test_trials=int(table.n_trials.sum()),
                test_total_nll=table.total_nll.sum(), n_test_correct=int(table.n_correct.sum())))
    cognitive_by_split = pd.DataFrame(cognitive_records)
    for key, group in cognitive_by_split.groupby(['task', 'model']):
        if sorted(group.random_split.tolist()) != list(range(n_splits)):
            raise ValueError(f'{key}: expected exactly one evaluation file per fold 0–9')
    # Pool trials within a fold, then give each fold equal weight.
    cognitive_by_split['test_nll'] = cognitive_by_split.test_total_nll / cognitive_by_split.n_test_trials
    cognitive_by_split['test_accuracy'] = cognitive_by_split.n_test_correct / cognitive_by_split.n_test_trials
    cognitive_by_model = cognitive_by_split.groupby(['task', 'model'], as_index=False).agg(
        test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
        test_nll_sd=('test_nll', 'std'), n_splits=('random_split', 'nunique'),
        n_test_trials=('n_test_trials', 'sum'))
    comparison_output.mkdir(parents=True, exist_ok=True)
    cognitive_by_split.to_csv(comparison_output / 'cognitive_test_by_split.csv', index=False)
    cognitive_by_model.to_csv(comparison_output / 'cognitive_test_by_model.csv', index=False)
    comparison = [cognitive_by_model.assign(model_family='Cognitive')]
    fold_comparison = [cognitive_by_split.assign(model_family='Cognitive')]


    for family, path in [('BasicNN', basic_nn_results), ('HyperNN', hyper_nn_results)]:
        if not path.exists():
            print(f'{family}: no saved results at {path}; skipping comparison')
            continue
        neural = pd.read_csv(path)
        neural = neural.loc[neural.task.isin(['cct', 'dd', 'stopsignal', 'motor'])].copy()
        if family == 'HyperNN':
            neural = neural.loc[neural.config_id.eq(selected_config)].copy()
        if neural.empty:
            print(f'{family}: no matching saved configuration; skipping comparison')
            continue
        if 'test_n_trials' in neural:
            neural = neural.rename(columns={'test_n_trials': 'n_test_trials'})
        for key, group in neural.groupby(['task', 'config_id']):
            if sorted(group.random_split.tolist()) != list(range(n_splits)):
                raise ValueError(f'{family}/{key}: expected ten unique folds')
        counts = cognitive_by_split.merge(neural[['task', 'random_split', 'n_test_trials']],
            on=['task', 'random_split'], suffixes=('_cognitive', '_neural'))
        if not counts.n_test_trials_cognitive.eq(counts.n_test_trials_neural).all():
            raise ValueError(f'{family}: cognitive and neural test trial counts differ')
        fold_comparison.append(neural.rename(columns={'config_id': 'model'}).assign(model_family=family))
        summary = neural.groupby(['task', 'config_id'], as_index=False).agg(
            test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
            test_nll_sd=('test_nll', 'std'), n_splits=('random_split', 'nunique'),
            n_test_trials=('n_test_trials', 'sum')).rename(columns={'config_id': 'model'})
        summary['model_family'] = family
        comparison.append(summary)
    model_comparison = pd.concat(comparison, ignore_index=True)
    # Long-format table: one row per model, task and held-out fold; no fold averaging.
    model_fold_results = pd.concat(fold_comparison, ignore_index=True).rename(columns={'random_split': 'fold'})
    model_fold_results = model_fold_results[['model_family', 'model', 'task', 'fold',
                                             'test_nll', 'test_accuracy', 'n_test_trials']]
    if model_fold_results.duplicated(['model_family', 'model', 'task', 'fold']).any():
        raise ValueError('Duplicate model/task/fold rows in comparison')
    if not np.isfinite(model_fold_results[['test_nll', 'test_accuracy']]).all().all():
        raise ValueError('Nonfinite fold metrics in comparison')
    model_fold_results['task_fold'] = model_fold_results.task + '_' + model_fold_results.fold.astype(str)
    model_fold_results = model_fold_results.sort_values(['task', 'model_family', 'model', 'fold']).reset_index(drop=True)
    model_fold_results.to_csv(comparison_output / 'cct_dd_model_by_fold.csv', index=False)
    print(f'Saved {len(model_fold_results)} model/task/fold rows')
    model_comparison.to_csv(comparison_output / 'cct_dd_model_comparison.csv', index=False)
    print(model_comparison.to_string(index=False))

    # These are conditional tests of the selected representation, not corrections
    # for selecting a configuration on the same participants. Participants must be
    # exchangeable. First-component significance does not establish extra shared axes.
    if run_embedding_permutations:
        participants = pd.read_csv(root / 'participant_map.csv').sort_values('participant_id').reset_index(drop=True)
        if not np.array_equal(participants.participant_id, np.arange(len(participants))):
            raise ValueError('Participant IDs must be contiguous and match embedding row order')
        dimensions = int(selected_config.split('_')[7])
        split_pairs = list(combinations(range(n_splits), 2))
        metric = CKA()
        consensus = {}
        tests = []

        # GCCA consensus matches the grid evaluation: standardize each view, align
        # all embedding components, standardize each aligned view, then average.
        for task in task_names:
            raw = [np.load(root / f'random_split_{split}' / f'{task}_embeddings.npy', allow_pickle=False).astype(float)
                   for split in range(n_splits)]
            if any(v.shape != (len(participants), dimensions) or not np.isfinite(v).all() for v in raw):
                raise ValueError(f'Invalid embeddings for {task}')
            if any(np.any(v.std(axis=0) < 1e-12) for v in raw):
                raise ValueError(f'Degenerate embedding dimension for {task}')
            views = [StandardScaler().fit_transform(v) for v in raw]
            with threadpool_limits(limits=1):
                gcca = GCCA(latent_dimensions=dimensions).fit(views)
                scores = gcca.transform(views)
                observed_r = float(gcca.score(views)[0])
                observed_cka = np.array([metric.linear_CKA(raw[a], raw[b]) for a, b in split_pairs])
            consensus[task] = np.mean([StandardScaler().fit_transform(v) for v in scores], axis=0)
            tests.append(dict(name=task, family='stability', model='GCCA', params=gcca.get_params(),
                              views=views, cka_views=raw, observed_r=observed_r,
                              observed_cka=observed_cka.mean(), pair_cka=observed_cka))

        for a, b in combinations(task_names, 2):
            views = [consensus[a], consensus[b]]
            with threadpool_limits(limits=1):
                cca = CCA(latent_dimensions=1).fit(views)
                observed_r = float(cca.score(views)[0])
                observed_cka = float(metric.linear_CKA(*views))
            tests.append(dict(name=f'{a}__{b}', family='cross_task', model='CCA', params=cca.get_params(),
                              views=views, cka_views=views, observed_r=observed_r,
                              observed_cka=observed_cka, pair_cka=None))

        views = [StandardScaler().fit_transform(consensus[task]) for task in task_names]
        with threadpool_limits(limits=1):
            gcca = GCCA(latent_dimensions=1).fit(views)
            observed_r = float(gcca.score(views)[0])
        tests.append(dict(name='all_tasks', family='joint', model='GCCA', params=gcca.get_params(),
                          views=views, cka_views=None, observed_r=observed_r,
                          observed_cka=None, pair_cka=None))

        permutation_dir = output / 'Permutations'
        permutation_dir.mkdir(parents=True, exist_ok=True)
        records, stability_pair_records = [], []
        for test_number, test in enumerate(tests):
            print(f"{test_number + 1}/{len(tests)}: {test['family']} / {test['name']}", flush=True)
            rng = np.random.default_rng(permutation_seed + test_number)
            null_r = np.empty(n_permutation)
            null_cka = np.empty(n_permutation) if test['cka_views'] is not None else None
            null_pairs = np.empty((n_permutation, len(split_pairs))) if test['family'] == 'stability' else None
            # Fix the first view; independently shuffle participant rows in all others.
            # Refit alignment for EVERY permutation. CKA uses the same row shuffles.
            with parallel_config(backend='loky', inner_max_num_threads=1):
                results = Parallel(n_jobs=n_jobs, return_as='generator')(
                    delayed(evaluate_permutation)(test['model'], test['params'], test['views'],
                        [np.arange(len(participants))] + [rng.permutation(len(participants)) for _ in test['views'][1:]],
                        cka_views=test['cka_views']) for _ in range(n_permutation))
                for permutation, (r, cka) in enumerate(results):
                    null_r[permutation] = r[0]
                    if null_cka is not None:
                        null_cka[permutation] = cka.mean()
                    if null_pairs is not None:
                        null_pairs[permutation] = cka
            null = pd.DataFrame({'first_component_r': null_r})
            if null_cka is not None:
                null['mean_linear_cka'] = null_cka
            if null_pairs is not None:
                for j, (a, b) in enumerate(split_pairs):
                    null[f'split_{a}_{b}_cka'] = null_pairs[:, j]
                    stability_pair_records.append(dict(task=test['name'], split_a=a, split_b=b,
                        linear_cka=test['pair_cka'][j], null_mean=null_pairs[:, j].mean(),
                        permutation_p=(1 + (null_pairs[:, j] >= test['pair_cka'][j]).sum()) / (n_permutation + 1)))
            null.index = np.arange(1, n_permutation + 1)
            null.to_csv(permutation_dir / f"{test['family']}_{test['name']}.csv", index_label='permutation')
            for label, observed, values in [('first_component_r', test['observed_r'], null_r),
                                             ('linear_cka', test['observed_cka'], null_cka)]:
                if values is None:
                    continue
                if not np.isfinite(observed) or not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite statistic: {test['name']}/{label}")
                records.append(dict(config_id=selected_config, family=test['family'], name=test['name'], metric=label,
                    observed=observed, null_mean=values.mean(), null_sd=values.std(ddof=1),
                    permutation_p=(1 + (values >= observed).sum()) / (n_permutation + 1),
                    n_permutation=n_permutation, permutation_seed=permutation_seed + test_number))

        summary = pd.DataFrame(records)
        # Separate families: 8 stability GCCA, 8 stability CKA, 28 pairwise CCA,
        # 28 pairwise CKA, and one joint GCCA. The joint test is not duplicated 28 times.
        for _, indices in summary.groupby(['family', 'metric']).groups.items():
            significant, adjusted, _, _ = multipletests(summary.loc[indices, 'permutation_p'],
                                                       alpha=permutation_alpha, method='fdr_bh')
            summary.loc[indices, 'permutation_p_adjusted'] = adjusted
            summary.loc[indices, 'significant'] = significant
        stability_pairs = pd.DataFrame(stability_pair_records)
        significant, adjusted, _, _ = multipletests(stability_pairs.permutation_p, alpha=permutation_alpha, method='fdr_bh')
        stability_pairs['permutation_p_adjusted'] = adjusted
        stability_pairs['significant'] = significant
        stability_pairs.to_csv(output / 'stability_split_pair_permutation_results.csv', index=False)
        summary.to_csv(output / 'permutation_results.csv', index=False)
        print(summary.to_string(index=False))
        print(f'Main-model permutation results saved to {output}')
