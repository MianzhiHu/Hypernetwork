from pathlib import Path
import shutil
from itertools import combinations
import hypernn_grid_search as training
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from cca_zoo.linear import GCCA, CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from CKA.CKA import CKA
from HyperNetwork import *

# ======================================================================================================================
# Settings
# ======================================================================================================================
# Saving directories
root = Path('./Results/HyperNN_Grid_Search')  # Change only this folder to switch searches.
output = root.with_name(f'{root.name}_Evaluation')
test_results_path = output / "test_results_all.csv"
stability_dir = output / 'Embedding_Stability'
aligned_dir = stability_dir / 'Aligned_Embeddings'
svd_aligned_dir = stability_dir / 'SVD_Aligned_Embeddings'
cross_task_dir = output / 'Cross_Task_Convergence'
consensus_dir = cross_task_dir / 'Consensus_Embeddings'
gcca_score_dir = cross_task_dir / 'GCCA_Scores'
typicality_dir = cross_task_dir / 'Typicality'

plots = Path('./Figures') / root.name
test_plots = plots / 'Test_Performance'
stability_plots = plots / 'Embedding_Stability'
cross_task_plots = plots / 'Cross_Task_Convergence'

for directory in [output, plots, stability_dir, aligned_dir, svd_aligned_dir, stability_plots, test_plots,
                  cross_task_dir, cross_task_plots, consensus_dir, gcca_score_dir, typicality_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Other parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
kept_variance = 0.99
n_splits = 10
n_participant_folds = 5
participant_split_seed = 42
task_names = ['cct', 'dd', 'motor', 'stopsignal', 'twobytwo']
prevalance = [0.2066, 0.4950, 0.8476, 0.7716, 0.9145]
# Fixed task mapping keeps rates correct when task_names selects a subset.
prevalence_by_task = dict(zip(task_names, prevalance, strict=True))
trial_keys = ['worker_id', 'participant_id', 'trial']
figure_size = (7.2, 4.2)
figure_dpi = 300
selected_config = 'layers_1_dims_8_rank_full_emb_2_nonlinear_nodes_8_reg_0.0001'

# Exclude a configuration if ANY listed hyperparameter value matches.
# Empty lists exclude nothing. rank=None means full rank.
exclude_hyperparameters = {
    'num_hidden_layers': [],  # Example: [2]
    'hidden_dim': [],        # Example: [8, 16]; common width of each layer
    'participant_emb_dim': [],  # Example: [2]
    'rank': [],              # Example: [2] or [None]
    'mapping': [],           # Example: ['linear']
    'hyper_hidden_dim': [],  # Example: [4]; None for linear mapping
    'hyper_reg': [],         # Example: [0.0001, 0.001, 0.01]
}

# Select configurations once, so all later analyses use the same exclusions.
configuration_rows = []
for folder in sorted(root.glob('layers_*')):
    if not folder.is_dir():
        continue
    parts = folder.name.split('_')
    configuration_rows.append({
        'config_id': folder.name,
        'num_hidden_layers': int(parts[1]),
        'hidden_dim': int(parts[3].split('x')[0]),
        'participant_emb_dim': int(parts[7]),
        'rank': None if parts[5] == 'full' else int(parts[5]),
        'mapping': parts[8],
        'hyper_hidden_dim': None if parts[10] == 'na' else int(parts[10]),
        'hyper_reg': float(parts[12]) if len(parts) > 12 else 0.0,
    })
configuration_table = pd.DataFrame(configuration_rows, dtype=object)
if configuration_table.empty:
    raise ValueError(f'No configuration folders in {root}')
keep_configuration = pd.Series(True, index=configuration_table.index)
for parameter, values in exclude_hyperparameters.items():
    if parameter not in configuration_table.columns or parameter == 'config_id':
        raise ValueError(f'Unknown exclusion hyperparameter: {parameter}')
    keep_configuration &= ~configuration_table[parameter].isin(values)
included_config_ids = set(configuration_table.loc[keep_configuration, 'config_id'])
if not included_config_ids:
    raise ValueError('Hyperparameter exclusions removed every configuration')
print(f'Configurations included: {len(included_config_ids)}; excluded: {(~keep_configuration).sum()}')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

stability_labels = {'first_component_r': 'Mean pairwise first-component r (GCCA)',
                    'linear_cka': 'Mean pairwise linear CKA',
                    'heldout_first_component_r': 'Mean participant-held-out first-component r',
                    'first_component_svd_r': 'Mean pairwise first-component r (SVD-GCCA)',
                    'linear_svd_cka': 'Mean pairwise linear CKA (SVD-GCCA)',
                    'heldout_first_component_svd_r': 'Mean participant-held-out first-component r (SVD-GCCA)'}

cross_task_labels = {'first_cca': 'First canonical correlation (r)',
                     'first_svd_cca': 'First canonical correlation (r) after SVD',
                     'gcca_first_component_r': 'Joint GCCA mean pairwise first-component r'}

cka_metric = CKA()
split_pairs = list(combinations(range(n_splits), 2))

# ======================================================================================================================
# Test Performance
# ======================================================================================================================
tasks = {task['name']: task for task in training.tasks if task['name'] in task_names}
participant_map = pd.read_csv(root / 'participant_map.csv')

# Load and validate the saved random splits for each task, and create DataLoaders for each split.
loaders = {}
for name, task in tasks.items():
    data = task["data"]
    saved = pd.read_csv(root / 'random_splits' / f'{name}_random_splits.csv')
    aligned = data[trial_keys].merge(saved, on=trial_keys, how='left', validate='one_to_one', indicator=True, sort=False)
    current_map = data[["participant_id", "worker_id"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)

    # Check that the participant mapping has not changed and that participant indices are contiguous.
    if not current_map.astype({"participant_id": "int64"}).equals(participant_map.sort_values("participant_id").reset_index(drop=True)):
        raise ValueError(f"Participant mapping changed for {name}")
    if not np.array_equal(current_map.participant_id, np.arange(len(current_map))):
        raise ValueError(f"Participant indices are not contiguous for {name}")

    # Validate that the saved trial identities match the current data and that fold numbers are valid.
    if len(saved) != len(data) or not aligned["_merge"].eq("both").all():
        raise ValueError(f"Saved trial identities do not match current data for {name}")
    if not aligned["fold"].isin(range(n_splits)).all():
        raise ValueError(f"Invalid fold numbers for {name}")

    # Create DataLoaders for each random split, using the saved fold assignments to select test trials.
    for split in range(n_splits):
        selected = aligned[f'random_split_{split}'].eq("test").to_numpy()
        dataset = BehavioralDataset(data.iloc[selected].copy(), task["x_col"], task["y_col"])
        loaders[name, split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)

# Load and evaluate each saved checkpoint, validating its configuration and computing test test_metrics.
paths = sorted(path for path in root.glob("*/random_split_*/*_model.pt")
               if path.name.removesuffix('_model.pt') in task_names
               and path.parent.parent.name in included_config_ids)
if not paths:
    raise FileNotFoundError(f"No saved checkpoints in {root}")

test_results = []
for number, path in enumerate(paths, 1):
    name = path.name.removesuffix('_model.pt')
    split = int(path.parent.name.removeprefix('random_split_'))
    config_id = path.parent.parent.name
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint['extra_config']
    dims = tuple(config['hidden_dims'])
    val_loss = checkpoint["best_val_loss"]
    best_epoch = checkpoint["best_epoch"]
    last_epoch = checkpoint["history"][-1]["epoch"]
    early_stopped = last_epoch - best_epoch >= checkpoint["patience"]
    task = tasks[name]

    model = HyperNN(n_participants=len(participant_map), input_dim=len(task["x_col"]), hidden_dim=dims, output_dim=2,
                    participant_emb_dim=config["participant_emb_dim"], hyper_hidden_dim=config["hyper_hidden_dim"],
                    rank=config["rank"], shared_right=config["shared_right"], mapping=config["mapping"],
                    num_hidden_layers=config["num_hidden_layers"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    test_metrics = test_model(model, loaders[name, split], device, prevalence_rate=prevalence_by_task[name])
    nll, accuracy, n_trials, pred_imbalance = test_metrics["raw_loss"], test_metrics["acc"], test_metrics["n_trials"], test_metrics["pred_imbalance"]
    if n_trials <= 0 or not np.isfinite([nll, accuracy]).all() or nll < 0 or not 0 <= accuracy <= 1:
        raise ValueError(f"Invalid test metrics for {path}: {test_metrics}")
    test_results.append({
        "config_id": config_id, "task": name, "random_split": split, "val_loss": val_loss,
        "test_nll": nll, "test_accuracy": accuracy, "n_test_trials": n_trials,
        "test_total_nll": nll * n_trials, "n_test_correct": int(round(accuracy * n_trials)),
        "best_epoch": checkpoint["best_epoch"], "last_epoch": last_epoch, "early_stopped": early_stopped,
        "num_hidden_layers": config["num_hidden_layers"], "hidden_dims": "-".join(map(str, dims)),
        "rank": config["rank"], "participant_emb_dim": config["participant_emb_dim"],
        "mapping": config["mapping"], "hyper_hidden_dim": config["hyper_hidden_dim"], "hyper_reg": config["hyper_reg"],
        "pred_imbalance": pred_imbalance,
        "pred_probability_1_mean": test_metrics["pred_probability_1_mean"],
        "pred_probability_1_std": test_metrics["pred_probability_1_std"],
        "prevalence_distance": test_metrics["prevalence_distance"],
    })
    if number % 100 == 0 or number == len(paths):
        pd.DataFrame(test_results[-100:] if number % 100 == 0 else test_results[-(number % 100):]).to_csv(
            test_results_path, mode="w" if number <= 100 else "a", header=number <= 100, index=False)
        print(f"Evaluated {number}/{len(paths)} checkpoints", flush=True)

# Consolidate the test_by_split test test_metrics and compute aggregated statistics by task and configuration
test_results_all = pd.DataFrame(test_results)
test_results_all.to_csv(test_results_path, index=False)

test_by_task = test_results_all.groupby(["config_id", "task"], as_index=False).agg(
    val_nll=("val_loss", "mean"), test_nll=("test_nll", "mean"), test_accuracy=("test_accuracy", "mean"),
    test_nll_sd=("test_nll", "std"), test_accuracy_sd=("test_accuracy", "std"), n_splits=("random_split", "nunique"),
    n_test_trials=("n_test_trials", "sum"), test_total_nll=("test_total_nll", "sum"),
    n_test_correct=("n_test_correct", "sum"), pred_imbalance=("pred_imbalance", "mean"), pred_dist=("prevalence_distance", "mean"))
test_by_config = test_results_all.groupby(["config_id"], as_index=False).agg(
    val_nll=("val_loss", "mean"), test_nll=("test_nll", "mean"), test_accuracy=("test_accuracy", "mean"),
    test_nll_sd=("test_nll", "std"), test_accuracy_sd=("test_accuracy", "std"), n_splits=("random_split", "nunique"),
    n_test_trials=("n_test_trials", "sum"), test_total_nll=("test_total_nll", "sum"),
    n_test_correct=("n_test_correct", "sum"), pred_imbalance=("pred_imbalance", "mean"), pred_dist=("prevalence_distance", "mean"))

# For many participants, the total number of trials cannot be evenly divided across 10 splits.
# So the rotating test set may have slightly different sizes.
# Therefore, when we average the test NLL across splits, this may cause slight weighting problems.
# Here we compute the pooled test NLL and accuracy across all splits for each task/configuration.
# It turns out that this is essentially the same as the mean test NLL and accuracy, but we include it for completeness.
test_by_task["pooled_test_nll"] = test_by_task.test_total_nll / test_by_task.n_test_trials
test_by_task["pooled_test_accuracy"] = test_by_task.n_test_correct / test_by_task.n_test_trials
test_config_columns = ["config_id", "num_hidden_layers", "hidden_dims", "rank", "participant_emb_dim", "mapping", "hyper_hidden_dim", "hyper_reg"]
test_configurations = test_results_all[test_config_columns].drop_duplicates()
test_by_task = test_by_task.merge(test_configurations, on="config_id", validate="many_to_one")
test_by_config = test_by_config.merge(test_configurations, on="config_id", validate="one_to_one")
test_by_task.to_csv(output / "test_results_summary.csv", index=False)
test_by_config.to_csv(output / "test_results_by_config.csv", index=False)
test_by_config_sorted = test_by_config.sort_values(["test_nll", "config_id"], ascending=[True, True]).reset_index(drop=True)
print(test_by_config_sorted.head(10).to_string(index=False), flush=True)

# Print the selected configuration's test performance for each task and overall.
test_selected = test_results_all.loc[test_results_all.config_id.eq(selected_config)].groupby("task", as_index=False).agg(
test_nll=("test_nll", "mean"), test_accuracy=("test_accuracy", "mean"), pred_imbalance=("pred_imbalance", "mean"), pred_dist=("prevalence_distance", "mean"))
print("Selected configuration test performance:")
print(test_selected)

# Track test NLL along each hyperparameter dimension, for each task and overall. This is a simple univariate analysis.
test_by_task["rank"] = test_by_task["rank"].apply(lambda x: f"LORA-{int(x)}" if pd.notnull(x) else "full-rank")
test_parameters = ['num_hidden_layers', 'hidden_dims', 'rank', 'participant_emb_dim', 'mapping', 'hyper_hidden_dim', 'hyper_reg']
for config in test_parameters:
    test_by_parameter = test_by_task.groupby([config, 'task'], as_index=False).agg(
        mean_test_nll=("test_nll", "mean"), mean_test_accuracy=("test_accuracy", "mean"),
        n_configs=("config_id", "nunique"))
    test_by_parameter.to_csv(output / f"test_results_by_param_{config}.csv", index=False)

# Plot test NLL and accuracy, separately sorted for each task and overall.
for name in list(tasks) + ['all_tasks']:
    test_plot_data = test_by_config if name == 'all_tasks' else test_by_task.loc[test_by_task.task.eq(name)].copy()
    if name != 'all_tasks':
        test_plot_data.to_csv(output / f'{name}_test_results.csv', index=False)
    for metric, ascending, ylabel in [('test_nll', True, 'Mean test NLL per valid trial'),
                                      ('test_accuracy', False, 'Mean test accuracy'),
                                      ('val_nll', True, 'Mean validation NLL per valid trial')]:
        test_ranked = test_plot_data.sort_values([metric, 'config_id'], ascending=[ascending, True]).reset_index(drop=True)
        test_ranked.insert(0, 'performance_rank', np.arange(1, len(test_ranked)+1))
        fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
        ax.plot(test_ranked.performance_rank, test_ranked[metric], color='#39739D')
        for selected_rank in test_ranked.index[test_ranked.config_id.eq(selected_config)] + 1:
            ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
            ax.legend(frameon=False)
        ax.set(title=name, xlabel='Configurations, best to worst', ylabel=ylabel)
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(test_plots / f'{name}_{metric}.png', dpi=figure_dpi)
        plt.close(fig)

# ======================================================================================================================
# Embedding Stability (GCCA across random splits)
# ======================================================================================================================
participants = pd.read_csv(root / "participant_map.csv").sort_values("participant_id").reset_index(drop=True)
if not np.array_equal(participants.participant_id, np.arange(len(participants))):
    raise ValueError("Saved participant IDs must be contiguous")
participants.to_csv(stability_dir / "participant_map.csv", index=False)

# Load the saved embeddings for each task and random split, validating their presence and structure.
stability_groups = {}
for path in sorted(root.glob("*/random_split_*/*_embeddings.npy")):
    name = path.name.removesuffix("_embeddings.npy")
    split = int(path.parent.name.removeprefix("random_split_"))
    if name not in task_names or path.parent.parent.name not in included_config_ids:
        continue
    if split not in range(n_splits):
        raise ValueError(f"Unexpected split: {path}")
    stability_groups.setdefault((path.parent.parent.name, name), {})[split] = path
if not stability_groups:
    raise ValueError("No saved embeddings for stability analysis")

# Run GCCA on each task/configuration group, then compute pairwise correlations and descriptive similarity.
stability_results, stability_component_results, stability_pair_results, stability_heldout_results = [], [], [], []

for index, ((config_id, task), files) in enumerate(sorted(stability_groups.items()), 1):
    views = []
    svd_views = []
    max_dimension = int(config_id.split('_')[7])
    for split in range(n_splits):
        view = np.load(files[split], allow_pickle=False).astype(float)

        # SVD through PCA. This is the first step of SVCCA, a variant of GCCA that uses SVD to reduce dimensionality before alignment
        svd_view = PCA(n_components=kept_variance, svd_solver='full').fit_transform(view)
        views.append(view)
        svd_views.append(svd_view)

    svd_dimension = min(v.shape[1] for v in svd_views)

    # Regular GCCA
    scaled = [StandardScaler().fit_transform(v) for v in views]
    gcca = GCCA(latent_dimensions=max_dimension).fit(scaled)
    aligned = gcca.transform(scaled)
    score = gcca.score(scaled)
    corr = np.array([[np.corrcoef(aligned[a][:, k], aligned[b][:, k])[0, 1]
                      for k in range(max_dimension)] for a, b in split_pairs])
    if not np.allclose(score, corr.mean(axis=0), atol=1e-10):
        raise ValueError("GCCA score does not match mean pairwise correlation")

    # SVD-GCCA
    svd_scaled = [StandardScaler().fit_transform(v) for v in svd_views]
    svd_gcca = GCCA(latent_dimensions=svd_dimension).fit(svd_scaled)
    svd_aligned = svd_gcca.transform(svd_scaled)
    svd_score = svd_gcca.score(svd_scaled)
    svd_corr = np.array([[np.corrcoef(svd_aligned[a][:, k], svd_aligned[b][:, k])[0, 1]
                      for k in range(svd_dimension)] for a, b in split_pairs])
    if not np.allclose(svd_score, svd_corr.mean(axis=0), atol=1e-10):
        raise ValueError("SVD-GCCA score does not match mean pairwise correlation")
    # Match the dimensionality of the SVD-GCCA correlations to the original GCCA dimensionality for consistency in reporting.
    if svd_corr.shape[1] < max_dimension:
        svd_corr = np.pad(svd_corr, ((0, 0), (0, max_dimension - svd_corr.shape[1])), mode='constant', constant_values=np.nan)


    # CKA
    cka = np.array([cka_metric.linear_CKA(views[a], views[b]) for a, b in split_pairs])
    svd_cka = np.array([cka_metric.linear_CKA(svd_views[a], svd_views[b]) for a, b in split_pairs])
    for j,(a,b) in enumerate(split_pairs):
        stability_pair_results.append(dict(config_id=config_id, task=task, split_a=a, split_b=b,
                                           first_component_r=corr[j,0], first_component_r2=corr[j,0]**2,
                                           linear_cka=cka[j], svd_first_component_r=svd_corr[j,0],
                                           svd_first_component_r2=svd_corr[j,0]**2, svd_linear_cka=svd_cka[j]))

    # Held-out participant GCCA
    heldout_correlations, svd_heldout_correlations = [], []
    participant_folds = list(KFold(n_splits=n_participant_folds, shuffle=True, random_state=participant_split_seed).split(participants))
    fold_labels = np.empty(len(participants), dtype=int)
    for fold, (_, heldout) in enumerate(participant_folds):
        fold_labels[heldout] = fold
    participants.assign(gcca_validation_fold=fold_labels).to_csv(stability_dir / 'gcca_participant_folds.csv', index=False)

    for fold, (train, test) in enumerate(participant_folds):
        # GCCA
        scalers = [StandardScaler().fit(v[train]) for v in views]
        train_views = [s.transform(v[train]) for s,v in zip(scalers,views)]
        test_views = [s.transform(v[test]) for s,v in zip(scalers,views)]
        fitted = GCCA(**gcca.get_params()).fit(train_views)
        heldout_scores = fitted.transform(test_views)
        if np.any(np.std(np.stack(heldout_scores), axis=1) < 1e-12):
            raise ValueError(f'Degenerate held-out GCCA component: {config_id}/{task}')
        corr_heldout = np.array([[np.corrcoef(heldout_scores[a][:, k], heldout_scores[b][:, k])[0, 1]
                                  for k in range(max_dimension)] for a, b in split_pairs])
        heldout_correlations.append(corr_heldout)
        cka_heldout = np.array([cka_metric.linear_CKA(views[a][test], views[b][test]) for a, b in split_pairs])

        # SVD-GCCA
        svd_scalers = [StandardScaler().fit(v[train]) for v in svd_views]
        svd_train_views = [s.transform(v[train]) for s,v in zip(svd_scalers,svd_views)]
        svd_test_views = [s.transform(v[test]) for s,v in zip(svd_scalers,svd_views)]
        svd_fitted = GCCA(**svd_gcca.get_params()).fit(svd_train_views)
        svd_heldout_scores = svd_fitted.transform(svd_test_views)
        if np.any(np.std(np.stack(svd_heldout_scores), axis=1) < 1e-12):
            raise ValueError(f'Degenerate held-out SVD-GCCA component: {config_id}/{task}')
        svd_corr_heldout = np.array([[np.corrcoef(svd_heldout_scores[a][:, k], svd_heldout_scores[b][:, k])[0, 1]
                                      for k in range(svd_dimension)] for a, b in split_pairs])
        # Match the dimensionality of the SVD-GCCA held-out correlations to the original GCCA dimensionality for consistency in reporting.
        if svd_corr_heldout.shape[1] < max_dimension:
            svd_corr_heldout = np.pad(svd_corr_heldout, ((0, 0), (0, max_dimension - svd_corr_heldout.shape[1])), mode='constant', constant_values=np.nan)
        svd_heldout_correlations.append(svd_corr_heldout)
        svd_cka_heldout = np.array([cka_metric.linear_CKA(svd_views[a][test], svd_views[b][test]) for a, b in split_pairs])

        # Compute the first component correlation for each split pair and record it for the held-out participants.
        for j,(a,b) in enumerate(split_pairs):
            stability_heldout_results.append(dict(config_id=config_id, task=task, participant_fold=fold,
                                                  split_a=a, split_b=b, first_component_r=corr_heldout[j,0],
                                                  first_component_r2=corr_heldout[j,0]**2, linear_cka=cka_heldout[j],
                                                  svd_first_component_r=svd_corr_heldout[j,0],
                                                  svd_first_component_r2=svd_corr_heldout[j,0]**2, svd_linear_cka=svd_cka_heldout[j]))
    # Combine all five participant folds: 225 split-pair correlations per component.
    corr_heldout = np.concatenate(heldout_correlations, axis=0)
    corr_svd_heldout = np.concatenate(svd_heldout_correlations, axis=0)

    # Component-level summaries
    for k in range(max_dimension):
        stability_component_results.append(
            dict(config_id=config_id, task=task, component=k + 1,
                 mean_pairwise_r=float(corr[:,k].mean()), mean_pairwise_r2=float((corr[:,k]**2).mean()),
                 mean_svd_pairwise_r=float(svd_corr[:,k].mean()), mean_svd_pairwise_r2=float((svd_corr[:,k]**2).mean()),
                 heldout_mean_pairwise_r=float(corr_heldout[:,k].mean()), heldout_mean_pairwise_r2=float((corr_heldout[:,k]**2).mean()),
                 heldout_mean_svd_pairwise_r=float(corr_svd_heldout[:,k].mean()), heldout_mean_svd_pairwise_r2=float((corr_svd_heldout[:,k]**2).mean())))

    stability_results.append(
        dict(config_id=config_id, task=task, n_participants=len(participants),
             participant_emb_dim=views[0].shape[1], n_components=max_dimension,
             first_component_r=float(corr[:,0].mean()),first_component_r2=float(np.mean(corr[:,0]**2)),
             all_components_r2=float(np.mean(corr**2)), linear_cka=float(cka.mean()),
             first_component_svd_r=float(svd_corr[:,0].mean()), first_component_svd_r2=float(np.mean(svd_corr[:,0]**2)),
             all_components_svd_r2=float(np.nanmean(svd_corr**2)), linear_svd_cka=float(svd_cka.mean()),
             heldout_first_component_r=float(np.mean(corr_heldout[:,0])),
             heldout_first_component_r2=float(np.mean(np.square(corr_heldout[:,0]))),
             heldout_first_component_svd_r=float(np.mean(corr_svd_heldout[:,0])),
             heldout_first_component_svd_r2=float(np.mean(np.square(corr_svd_heldout[:,0])))))

    means_svd = np.stack([np.pad(v.mean(axis=0), (0, max_dimension - v.shape[1]), mode='constant', constant_values=np.nan) for v in svd_views])
    scales_svd = np.stack([np.pad(v.std(axis=0), (0, max_dimension - v.shape[1]), mode='constant', constant_values=np.nan) for v in svd_views])
    np.savez_compressed(aligned_dir / f'{config_id}_{task}.npz',
                        scores=np.stack(aligned), svd_scores=np.stack(svd_aligned), weights=np.stack(gcca.weights_),
                        means=np.stack([v.mean(axis=0) for v in views]),
                        scales=np.stack([v.std(axis=0) for v in views]),
                        means_svd=means_svd,
                        scales_svd=scales_svd,
                        participant_id=participants.participant_id.to_numpy(),
                        worker_id=participants.worker_id.to_numpy(dtype=str), random_splits=np.arange(n_splits))
    if len(stability_results) % 50 == 0:
        print(f'GCCA: {len(stability_results)} processed task/configuration groups', flush=True)

stability_results_all = pd.DataFrame(stability_results)
stability_results_all.to_csv(stability_dir / 'stability_results_all.csv', index=False)
pd.DataFrame(stability_component_results).to_csv(stability_dir / 'stability_by_component.csv', index=False)
pd.DataFrame(stability_pair_results).to_csv(stability_dir / 'stability_by_split_pair.csv', index=False)
pd.DataFrame(stability_heldout_results).to_csv(stability_dir / 'stability_participant_heldout.csv', index=False)

stability_metrics = ['first_component_r', 'first_component_svd_r', 'linear_cka', 'linear_svd_cka', 'heldout_first_component_r', 'heldout_first_component_svd_r']
stability_by_config = stability_results_all.groupby('config_id', as_index=False)[stability_metrics].mean()
stability_by_config.to_csv(stability_dir / 'stability_by_config.csv', index=False)
stability_by_config_sorted = stability_by_config.sort_values(['linear_cka', 'config_id'], ascending=[False, True]).reset_index(drop=True)
print(stability_by_config_sorted.head(20).to_string(index=False), flush=True)

# Print the selected configuration's stability performance for each task and overall.
print(f'Selected configuration: {selected_config}', flush=True)
print('Stability performance by task:', flush=True)
stability_selected = stability_results_all.loc[stability_results_all.config_id.eq(selected_config)].groupby('task', as_index=False)[stability_metrics].mean()
print(stability_selected.to_string(index=False), flush=True)


for task in task_names + ['all_tasks']:
    stability_plot_data = stability_by_config if task == 'all_tasks' else stability_results_all.loc[stability_results_all.task.eq(task)]
    for metric, ylabel in stability_labels.items():
        stability_ranked = stability_plot_data.sort_values([metric,'config_id'], ascending=[False,True]).reset_index(drop=True)
        stability_ranked.insert(0,'performance_rank',np.arange(1,len(stability_ranked)+1))
        stem = stability_plots / f'{task}_{metric}'
        fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
        ax.plot(stability_ranked.performance_rank, stability_ranked[metric], color='#39739D', linewidth=1.5)
        for selected_rank in stability_ranked.index[stability_ranked.config_id.eq(selected_config)] + 1:
            ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
            ax.legend(frameon=False)
        ax.set(title=f'{task} | split/seed stability | {len(stability_ranked)} configurations',
               xlabel='Configurations, highest to lowest', ylabel=ylabel)
        ax.set_xticks([])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.18)
        fig.savefig(stem.with_suffix('.png'), dpi=figure_dpi)
        plt.close(fig)
print(f'Embedding stability saved to {stability_dir}', flush=True)

# ======================================================================================================================
# Cross-task convergence (pairwise CCA/CKA and joint eight-task GCCA)
# ======================================================================================================================
# All within-task GCCA components are inputs; cross-task CCA fits only one latent component.
cross_task_config_ids = sorted({path.stem.rsplit('_', 1)[0] for path in aligned_dir.glob('*.npz')}
                               & included_config_ids)
identities = pd.read_csv(stability_dir / 'participant_map.csv').sort_values('participant_id').reset_index(drop=True)
# diff = set(cross_task_config_ids) - set(same_pred_config.config_id.unique())
cross_task_results = []
for config_id in cross_task_config_ids:
    embedding_dim = int(config_id.split('_')[7])
    consensus = {}
    consensus_svd = {}
    for task in task_names:
        with np.load(aligned_dir / f'{config_id}_{task}.npz', allow_pickle=False) as saved:
            scores = saved['scores']
            svd_scores = saved['svd_scores']
            if scores.shape != (n_splits, len(identities), embedding_dim) or not np.isfinite(scores).all():
                raise ValueError(f'Invalid aligned embeddings: {config_id}/{task}')
            # Standardize each split's scores before averaging to create a consensus embedding for each task.
            scores_z = np.empty_like(scores)
            svd_scores_z = np.empty_like(svd_scores)
            for split in range(n_splits):
                scores_z[split] = StandardScaler().fit_transform(scores[split])
                svd_scores_z[split] = StandardScaler().fit_transform(svd_scores[split])
            x = scores_z.mean(axis=0)
            x_svd = svd_scores_z.mean(axis=0)
        consensus[task] = x
        consensus_svd[task] = x_svd
        pd.concat([identities, pd.DataFrame(x, columns=[f'gcca_{k+1}' for k in range(x.shape[1])])],axis=1).to_csv(
            consensus_dir / f'{config_id}_{task}.csv', index=False)
        pd.concat([identities, pd.DataFrame(x_svd, columns=[f'gcca_svd_{k+1}' for k in range(x_svd.shape[1])])],axis=1).to_csv(
            consensus_dir / f'{config_id}_{task}_svd.csv', index=False)

    # Cross-task GCCA
    views = [StandardScaler().fit_transform(consensus[task]) for task in task_names]
    gcca = GCCA(latent_dimensions=1).fit(views)
    gcca_r = float(gcca.score(views)[0])
    gcca_correlations = gcca.pairwise_correlations(views)[:, :, 0]
    gcca_pair_r = gcca_correlations[np.triu_indices(len(task_names), k=1)]
    if not np.isfinite(gcca_pair_r).all():
        raise ValueError(f'Undefined joint GCCA: {config_id}')
    gcca_scores = np.column_stack([v[:, 0] for v in gcca.transform(views)])
    pd.concat([identities, pd.DataFrame(gcca_scores, columns=task_names)], axis=1).to_csv(
        gcca_score_dir / f'{config_id}.csv', index=False)

    # Cross-task SVD-GCCA
    views_svd = [StandardScaler().fit_transform(consensus_svd[task]) for task in task_names]
    svd_gcca = GCCA(latent_dimensions=1).fit(views_svd)
    svd_gcca_r = float(svd_gcca.score(views_svd)[0])
    svd_gcca_correlations = svd_gcca.pairwise_correlations(views_svd)[:, :, 0]
    svd_gcca_pair_r = svd_gcca_correlations[np.triu_indices(len(task_names), k=1)]
    if not np.isfinite(svd_gcca_pair_r).all():
        raise ValueError(f'Undefined joint SVD-GCCA: {config_id}')
    svd_gcca_scores = np.column_stack([v[:, 0] for v in svd_gcca.transform(views_svd)])
    pd.concat([identities, pd.DataFrame(svd_gcca_scores, columns=task_names)], axis=1).to_csv(
        gcca_score_dir / f'{config_id}_svd.csv', index=False)

    # Compute pairwise CCA and CKA between tasks.
    for a,b in combinations(task_names,2):
        # Pair-wise CCA
        cca = CCA(latent_dimensions=1).fit([consensus[a], consensus[b]])
        r = float(cca.score([consensus[a], consensus[b]])[0])
        if not np.isfinite(r):
            raise ValueError(f'Undefined CCA: {config_id}/{a}/{b}')
        pair = f'{a}__{b}'
        cka = cka_metric.linear_CKA(consensus[a], consensus[b])

        # Pair-wise SVD-GCCA
        svd_cca = CCA(latent_dimensions=1).fit([consensus_svd[a], consensus_svd[b]])
        svd_r = float(svd_cca.score([consensus_svd[a], consensus_svd[b]])[0])
        if not np.isfinite(svd_r):
            raise ValueError(f'Undefined SVD-CCA: {config_id}/{a}/{b}')
        svd_cka = cka_metric.linear_CKA(consensus_svd[a], consensus_svd[b])

        # Save one row per configuration and task pair.
        cross_task_results.append(
            dict(config_id=config_id, task_a=a, task_b=b, task_pair=pair,
            n_participants=len(identities), first_cca=r, first_cca_r2=r**2, linear_cka=cka, gcca_first_component_r=gcca_r,
            gcca_mean_pairwise_r2=np.mean(gcca_pair_r**2), first_svd_cca=svd_r, first_svd_cca_r2=svd_r**2, linear_svd_cka=svd_cka,
            svd_gcca_first_component_r=svd_gcca_r, svd_gcca_mean_pairwise_r2=np.mean(svd_gcca_pair_r**2)))

    if len(cross_task_results) % 50 == 0:
        print(f'Cross-task CCA: {len(cross_task_results)} processed configurations', flush=True)

cross_task_by_pair = pd.DataFrame(cross_task_results)
cross_task_by_pair = cross_task_by_pair.sort_values(['task_pair', 'config_id']).reset_index(drop=True)
cross_task_by_pair.to_csv(cross_task_dir / 'cca_by_task_pair.csv',index=False)

# Highest first-component CCA per task pair; config_id breaks exact ties.
best_cca_by_pair = (cross_task_by_pair.sort_values(['task_pair', 'first_cca', 'config_id'], ascending=[True, False, True])
                   .drop_duplicates('task_pair').reset_index(drop=True))
best_cca_by_pair.to_csv(cross_task_dir / 'best_cca_by_pair.csv', index=False)

cross_task_by_config = cross_task_by_pair.groupby('config_id', as_index=False).agg(
    mean_first_cca=('first_cca','mean'), mean_first_cca_r2=('first_cca_r2','mean'), mean_linear_cka=('linear_cka','mean'),
    mean_svd_first_cca=('first_svd_cca','mean'), mean_svd_first_cca_r2=('first_svd_cca_r2','mean'), mean_svd_linear_cka=('linear_svd_cka','mean'),
    gcca_first_component_r=('gcca_first_component_r','first'), gcca_mean_pairwise_r2=('gcca_mean_pairwise_r2','first'),
    svd_gcca_first_component_r=('svd_gcca_first_component_r','first'), svd_gcca_mean_pairwise_r2=('svd_gcca_mean_pairwise_r2','first'))

cross_task_by_config = cross_task_by_config.sort_values(['mean_first_cca', 'config_id'], ascending=[False, True]).reset_index(drop=True)
cross_task_by_config.insert(0, 'performance_rank', np.arange(1, len(cross_task_by_config) + 1))
cross_task_by_config.to_csv(cross_task_dir / 'cca_by_config.csv', index=False)

# 28 pairwise first-CCA rankings and one joint GCCA ranking across configurations.
for pair in list(cross_task_by_pair.task_pair.unique()) + ['all_tasks']:
    metric = 'gcca_first_component_r' if pair == 'all_tasks' else 'first_cca'
    cross_task_plot_data = cross_task_by_pair.drop_duplicates('config_id') if pair == 'all_tasks' else cross_task_by_pair.loc[cross_task_by_pair.task_pair.eq(pair)]
    cross_task_ranked = cross_task_plot_data.sort_values([metric, 'config_id'], ascending=[False, True]).reset_index(drop=True)
    title = f'Joint GCCA across {len(task_names)} tasks' if pair == 'all_tasks' else pair.replace('__', ' vs ')
    fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
    ax.plot(np.arange(1, len(cross_task_ranked) + 1), cross_task_ranked[metric], color='#39739D')
    for selected_rank in cross_task_ranked.index[cross_task_ranked.config_id.eq(selected_config)] + 1:
        ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
        ax.legend(frameon=False)
    ax.set(title=title, xlabel='Configurations, best to worst', ylabel=cross_task_labels[metric])
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(cross_task_plots / f'{pair}_{metric}.png', dpi=figure_dpi)
    plt.close(fig)
print(cross_task_by_config.head(5).to_string(index=False), flush=True)
print(f'Cross-task CCA saved to {cross_task_dir}',flush=True)

# ======================================================================================================================
# Typicality of the 28 cross-task CCA associations across configurations
# ======================================================================================================================
# Now we find the config that has the most typical cross-task associations.
typicality_pairs = [f'{a}__{b}' for a, b in combinations(task_names, 2)]
typicality_data = pd.read_csv(cross_task_dir / 'cca_by_task_pair.csv')
if set(typicality_data.task_pair) != set(typicality_pairs):
    raise ValueError(f'Expected all {len(typicality_pairs)} task pairs for configuration typicality')

for representation, metric in [('original', 'first_cca'), ('svd', 'first_svd_cca')]:
    # Rows are configurations; columns are identically ordered task pairs.
    association_vectors = typicality_data.pivot(index='config_id', columns='task_pair', values=metric)
    association_vectors = association_vectors.reindex(columns=typicality_pairs).sort_index()
    values = association_vectors.to_numpy(dtype=float)
    config_similarity = np.corrcoef(values)
    np.fill_diagonal(config_similarity, np.nan)  # Exclude self-comparisons.
    config_similarity = np.clip(config_similarity, -1 + 1e-12, 1 - 1e-12)  # Avoid infinite Fisher z.
    typicality_z = np.arctanh(config_similarity)
    typicality_summary = pd.DataFrame({
        'config_id': association_vectors.index,
        'typicality_r': np.tanh(np.nanmean(typicality_z, axis=1)),
        'mean_cross_task_cca_r': values.mean(axis=1),
        'n_reference_configs': len(values) - 1,
    }).sort_values(['typicality_r', 'config_id'], ascending=[False, True]).reset_index(drop=True)
    association_vectors.to_csv(typicality_dir / f'{representation}_association_vectors.csv')
    typicality_summary.to_csv(typicality_dir / f'{representation}_typicality_ranking.csv', index=False)
    fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
    ax.plot(typicality_summary.index + 1, typicality_summary.typicality_r, color='#39739D')
    for selected_rank in typicality_summary.index[typicality_summary.config_id.eq(selected_config)] + 1:
        ax.axvline(selected_rank, color='#D55E00', linestyle='--', label='Selected configuration')
        ax.legend(frameon=False)
    ax.set(title=f'{representation} | typicality of cross-task associations',
           xlabel='Configurations, most to least typical', ylabel='Fisher-z averaged Pearson r')
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(typicality_dir / f'{representation}_typicality.png', dpi=figure_dpi)
    plt.close(fig)
    print(f'{representation}: most typical cross-task association patterns', flush=True)
    print(typicality_summary.head(20).to_string(index=False), flush=True)


