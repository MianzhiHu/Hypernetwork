from pathlib import Path
from itertools import combinations
import hypernn_grid_search as training
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from cca_zoo.linear import GCCA, CCA
from cca_zoo.metrics import pairwise_correlations, average_pairwise_correlations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from CKA.CKA import CKA
from HyperNetwork import BehavioralDataset, behavioral_collate_fn, HyperNN, test_model
from task_config import selected_config

# ======================================================================================================================
# Settings
# ======================================================================================================================
# Saving directories
root = Path('./Results/HyperNN_Grid_Search')  # Change only this folder to switch searches.
output = root.with_name(f'{root.name}_Evaluation')
test_results_path = output / "test_results_all.csv"
stability_dir = output / 'Embedding_Stability'
aligned_dir = stability_dir / 'GCCA_Aligned_Embeddings'
cross_task_dir = output / 'Cross_Task_Convergence'
consensus_dir = cross_task_dir / 'Consensus_Embeddings'
gcca_score_dir = cross_task_dir / 'GCCA_Scores'
typicality_dir = cross_task_dir / 'Typicality'

plots = Path('./Figures') / root.name
test_plots = plots / 'Test_Performance'
stability_plots = plots / 'Embedding_Stability'
cross_task_plots = plots / 'Cross_Task_Convergence'

for directory in [output, plots, stability_dir, aligned_dir, stability_plots, test_plots, cross_task_dir,
                  cross_task_plots, consensus_dir, gcca_score_dir, typicality_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Other parameters
skip_configurations = {'emb': [2], 'reg': [0.0001, 0.001, 0.01]}  # Example: exclude small embeddings and low regularization.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
kept_variance = 0.99
n_splits = 10
n_participant_folds = 10
participant_split_seed = 42
cka_metric = CKA()
split_pairs = list(combinations(range(n_splits), 2))
task_names = ['cct', 'dd', 'motor', 'stopsignal', 'twobytwo']
trial_keys = ['worker_id', 'participant_id', 'trial']
stability_metrics = ['first_component_r', 'first_component_svd_r', 'linear_cka', 'linear_svd_cka', 'heldout_first_component_r']
prevalence_by_task = {
    'cct': 0.2066,
    'dd': 0.4950,
    'motor': 0.8476,
    'stopsignal': 0.7716,
    'twobytwo': 0.9145,
}

# Apply the skip_configurations filter to identify which configurations to include in the evaluation.
included_config_ids = set()
for folder in sorted(root.glob("layers_*")):
    if not folder.is_dir():
        continue

    parts = folder.name.split("_")
    parameters = {
        "layers": int(parts[1]),
        "dims": parts[3],
        "rank": parts[5],
        "emb": int(parts[7]),
        "mapping": parts[8],
        "nodes": None if parts[10] == "na" else int(parts[10]),
        "reg": float(parts[12]) if len(parts) > 12 else 0.0,
    }

    if not any(parameters[key] in values for key, values in skip_configurations.items()):
        included_config_ids.add(folder.name)

print(f"Configurations included: {len(included_config_ids)}")

# ======================================================================================================================
# Test Performance
# ======================================================================================================================
tasks = {task['name']: task for task in training.tasks if task['name'] in task_names}
participant_map = (pd.read_csv(root / "participant_map.csv").sort_values("participant_id").reset_index(drop=True))

# Load and validate the saved random splits for each task, and create DataLoaders for each split.
loaders = {}
for name, task in tasks.items():
    data = task["data"]
    saved_data = pd.read_csv(root / 'random_splits' / f'{name}_random_splits.csv')
    aligned_data = data[trial_keys].merge(saved_data, on=trial_keys, how='left', validate='one_to_one', indicator=True, sort=False)

    # Match participants and trials to the saved training data.
    if len(saved_data) != len(data) or not aligned_data["_merge"].eq("both").all():
        raise ValueError(f"Saved trial participants do not match current data for {name}")

    # Create DataLoaders for each random split, using the saved fold assignments to select test trials.
    for split in range(n_splits):
        selected = aligned_data[f'random_split_{split}'].eq("test").to_numpy()
        dataset = BehavioralDataset(data.iloc[selected].copy(), task["x_col"], task["y_col"])
        loaders[name, split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)

# Load and evaluate each saved checkpoint, validating its configuration and computing test test_metrics.
paths = sorted(path for path in root.glob("*/random_split_*/*_model.pt")
               if path.name.removesuffix("_model.pt") in task_names and path.parent.parent.name in included_config_ids)

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

    # Load the saved model
    model = HyperNN(n_participants=len(participant_map), input_dim=len(task["x_col"]), hidden_dim=dims, output_dim=2,
                    participant_emb_dim=config["participant_emb_dim"], hyper_hidden_dim=config["hyper_hidden_dim"],
                    rank=config["rank"], shared_right=config["shared_right"], mapping=config["mapping"],
                    num_hidden_layers=config["num_hidden_layers"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    # Testing the model on the test set
    test_metrics = test_model(model, loaders[name, split], device, prevalence_rate=prevalence_by_task[name])
    nll, accuracy, n_trials = test_metrics["raw_loss"], test_metrics["acc"], test_metrics["n_trials"]
    if n_trials == 0:
        raise ValueError(f"No valid test trials: {path}")
    test_results.append({
        "config_id": config_id, "task": name, "random_split": split, "val_loss": val_loss,
        "test_nll": nll, "test_accuracy": accuracy, "n_test_trials": n_trials,
        "test_total_nll": nll * n_trials, "n_test_correct": int(round(accuracy * n_trials)),
        "best_epoch": checkpoint["best_epoch"], "last_epoch": last_epoch, "early_stopped": early_stopped,
        "num_hidden_layers": config["num_hidden_layers"], "hidden_dims": "-".join(map(str, dims)),
        "rank": config["rank"], "participant_emb_dim": config["participant_emb_dim"],
        "mapping": config["mapping"], "hyper_hidden_dim": config["hyper_hidden_dim"], "hyper_reg": config["hyper_reg"],
        "pred_imbalance": test_metrics["pred_imbalance"],
        "pred_probability_1_mean": test_metrics["pred_probability_1_mean"],
        "pred_probability_1_std": test_metrics["pred_probability_1_std"],
        "prevalence_distance": test_metrics["prevalence_distance"],
    })
    if number % 100 == 0 or number == len(paths):
        pd.DataFrame(test_results).to_csv(test_results_path, index=False)
        print(f"Evaluated {number}/{len(paths)} checkpoints", flush=True)

# Consolidate the test_by_split test test_metrics and compute aggregated statistics by task and configuration
test_results_all = pd.DataFrame(test_results)
test_results_all.to_csv(test_results_path, index=False)
test_config_columns = ["config_id", "num_hidden_layers", "hidden_dims", "rank", "participant_emb_dim", "mapping", "hyper_hidden_dim", "hyper_reg"]
test_configurations = test_results_all[test_config_columns].drop_duplicates()

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
test_by_task = test_by_task.merge(test_configurations, on="config_id", validate="many_to_one")
test_by_config = test_by_config.merge(test_configurations, on="config_id", validate="one_to_one")
print(test_by_config.sort_values(["test_nll", "config_id"], ascending=[True, True]).reset_index(drop=True).head(10).to_string(index=False), flush=True)
test_by_task.to_csv(output / "test_results_summary.csv", index=False)
test_by_config.to_csv(output / "test_results_by_config.csv", index=False)

# Track test NLL along each hyperparameter dimension, for each task and overall. This is a simple univariate analysis.
test_parameters = ['hidden_dims', 'participant_emb_dim', 'hyper_hidden_dim']
for config in test_parameters:
    test_by_parameter = test_by_task.groupby([config, 'task'], as_index=False).agg(
        mean_test_nll=("test_nll", "mean"), mean_test_accuracy=("test_accuracy", "mean"),
        n_configs=("config_id", "nunique"))
    print(f"\nTest NLL by {config}:\n{test_by_parameter.to_string(index=False)}", flush=True)
    test_by_parameter.to_csv(output / f"test_results_by_param_{config}.csv", index=False)

# ======================================================================================================================
# Embedding Stability (GCCA across random splits)
# ======================================================================================================================
# Transfer the participant map
participants = pd.read_csv(root / "participant_map.csv").sort_values("participant_id").reset_index(drop=True)
if not np.array_equal(participants.participant_id, np.arange(len(participants))):
    raise ValueError("Saved participant IDs must be contiguous")
participants.to_csv(stability_dir / "participant_map.csv", index=False)

# Load the saved embeddings for each task and random split, validating their presence and structure.
stability_groups = {}
paths_embeddings = sorted(path for path in root.glob("*/random_split_*/*_embeddings.npy")
                          if path.name.removesuffix("_embeddings.npy") in task_names and path.parent.parent.name in included_config_ids)

for path in paths_embeddings:
    name = path.name.removesuffix("_embeddings.npy")
    split = int(path.parent.name.removeprefix("random_split_"))
    stability_groups.setdefault((path.parent.parent.name, name), {})[split] = path

# Prepare to compute stability metrics for each configuration and task, including GCCA, SVD-GCCA, and CKA, with or without held-out participants.
stability_results = []
stability_component_results = []
i = 0
participant_folds = list(KFold(n_splits=n_participant_folds, shuffle=True, random_state=participant_split_seed).split(participants))
fold_labels = np.empty(len(participants), dtype=int)
for fold, (_, heldout) in enumerate(participant_folds):
    fold_labels[heldout] = fold
participants.assign(gcca_validation_fold=fold_labels).to_csv(stability_dir / 'gcca_participant_folds.csv', index=False)

# Compute stability metrics
for (config_id, task), files in sorted(stability_groups.items()):
    i += 1
    print(f"Computing stability metrics for {task} | {config_id} ({i}/{len(stability_groups)})", flush=True)
    views = []
    svd_views = []
    max_dimension = int(config_id.split('_')[7])
    for split in range(n_splits):
        # Original embeddings
        view = np.load(files[split], allow_pickle=False).astype(float)
        views.append(view)

        # SVD through PCA. This is the first step of SVCCA, a variant of GCCA that uses SVD to reduce dimensionality before alignment
        svd_view = PCA(n_components=kept_variance, svd_solver='full').fit_transform(view)
        svd_views.append(svd_view)

    svd_dimension = min(v.shape[1] for v in svd_views)

    # Regular GCCA
    scaled = [StandardScaler().fit_transform(v) for v in views]
    gcca = GCCA(n_components=max_dimension).fit(scaled)
    aligned = gcca.transform(scaled)
    score = average_pairwise_correlations(pairwise_correlations(aligned))

    # SVD-GCCA
    svd_scaled = [StandardScaler().fit_transform(v) for v in svd_views]
    svd_gcca = GCCA(n_components=svd_dimension).fit(svd_scaled)
    svd_aligned = svd_gcca.transform(svd_scaled)
    svd_score = average_pairwise_correlations(pairwise_correlations(svd_aligned))

    # CKA
    cka = np.array([cka_metric.linear_CKA(views[a], views[b]) for a, b in split_pairs])
    svd_cka = np.array([cka_metric.linear_CKA(svd_views[a], svd_views[b]) for a, b in split_pairs])

    # Held-out participant GCCA
    heldout_correlations = []

    for fold, (train, test) in enumerate(participant_folds):
        # GCCA
        scalers = [StandardScaler().fit(v[train]) for v in views]
        train_views = [s.transform(v[train]) for s,v in zip(scalers,views)]
        test_views = [s.transform(v[test]) for s,v in zip(scalers,views)]
        fitted = GCCA(**gcca.get_params()).fit(train_views)
        heldout_scores = fitted.transform(test_views)
        heldout_r = average_pairwise_correlations(pairwise_correlations(heldout_scores))
        heldout_correlations.append(heldout_r)

    # Combine all five participant folds: 225 split-pair correlations per component.
    corr_heldout = np.column_stack(heldout_correlations).mean(axis=1)

    # Component-level summaries
    for k in range(max_dimension):
        stability_component_results.append(
            dict(config_id=config_id, task=task, component=k + 1, mean_pairwise_r=score[k],
                 mean_svd_pairwise_r=svd_score[k] if k < svd_dimension else np.nan, heldout_mean_pairwise_r=corr_heldout[k]))

    stability_results.append(
        dict(config_id=config_id, task=task, n_participants=len(participants),
             participant_emb_dim=views[0].shape[1], n_components=max_dimension,
             first_component_r=score[0], all_component_r=float(np.mean(score)), linear_cka=float(cka.mean()),
             first_component_svd_r=svd_score[0], all_components_svd_r=float(np.mean(svd_score)), linear_svd_cka=float(svd_cka.mean()),
             heldout_first_component_r=corr_heldout[0], heldout_all_components_r=float(np.mean(corr_heldout))))

    means_svd = np.stack([np.pad(v.mean(axis=0), (0, max_dimension - v.shape[1]), mode='constant', constant_values=np.nan) for v in svd_views])
    scales_svd = np.stack([np.pad(v.std(axis=0), (0, max_dimension - v.shape[1]), mode='constant', constant_values=np.nan) for v in svd_views])
    np.savez_compressed(aligned_dir / f'{config_id}_{task}.npz',
                        scores=np.stack(aligned), svd_scores=np.stack(svd_aligned), weights=np.stack(gcca.weights_),
                        means=np.stack([v.mean(axis=0) for v in views]), scales=np.stack([v.std(axis=0) for v in views]),
                        means_svd=means_svd, scales_svd=scales_svd,
                        participant_id=participants.participant_id.to_numpy(), worker_id=participants.worker_id.to_numpy(dtype=str), random_splits=np.arange(n_splits))

# Save the stability results to CSV
stability_results_all = pd.DataFrame(stability_results)
stability_results_all.to_csv(stability_dir / 'stability_results_all.csv', index=False)
pd.DataFrame(stability_component_results).to_csv(stability_dir / 'stability_by_component.csv', index=False)

stability_by_config = stability_results_all.groupby('config_id', as_index=False)[stability_metrics].mean()
stability_by_config.to_csv(stability_dir / 'stability_by_config.csv', index=False)
print(stability_by_config.sort_values(['linear_cka', 'config_id'], ascending=[False, True]).reset_index(drop=True).head(20).to_string(index=False), flush=True)

# ======================================================================================================================
# Cross-task convergence (pairwise CCA/CKA and joint eight-task GCCA)
# ======================================================================================================================
# All within-task GCCA components are inputs; cross-task CCA fits only one latent component.
cross_task_config_ids = sorted({path.stem.rsplit("_", 1)[0] for path in aligned_dir.glob("*.npz")} & included_config_ids)
participants = pd.read_csv(stability_dir / 'participant_map.csv').sort_values('participant_id').reset_index(drop=True)

# Compute cross-task GCCA and SVD-GCCA, then compute pairwise CCA and CKA between tasks.
cross_task_results = []
j = 0
for config_id in cross_task_config_ids:
    j += 1
    print(f'Computing cross-task convergence for {config_id} ({j}/{len(cross_task_config_ids)})', flush=True)
    embedding_dim = int(config_id.split('_')[7])
    consensus = {}
    consensus_svd = {}
    for task in task_names:
        with np.load(aligned_dir / f'{config_id}_{task}.npz', allow_pickle=False) as saved:
            scores = saved['scores']
            svd_scores = saved['svd_scores']
            if scores.shape != (n_splits, len(participants), embedding_dim) or not np.isfinite(scores).all():
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
        pd.concat([participants, pd.DataFrame(x, columns=[f'gcca_{k+1}' for k in range(x.shape[1])])],axis=1).to_csv(
            consensus_dir / f'{config_id}_{task}.csv', index=False)
        pd.concat([participants, pd.DataFrame(x_svd, columns=[f'gcca_svd_{k+1}' for k in range(x_svd.shape[1])])],axis=1).to_csv(
            consensus_dir / f'{config_id}_{task}_svd.csv', index=False)

    # Cross-task GCCA
    views = [StandardScaler().fit_transform(consensus[task]) for task in task_names]
    gcca = GCCA(n_components=1).fit(views)
    gcca_r = float(gcca.score(views))
    gcca_correlations = pairwise_correlations(gcca.transform(views))[:, :, 0]
    gcca_pair_r = gcca_correlations[np.triu_indices(len(task_names), k=1)]
    gcca_scores = np.column_stack([v[:, 0] for v in gcca.transform(views)])
    pd.concat([participants, pd.DataFrame(gcca_scores, columns=task_names)], axis=1).to_csv(
        gcca_score_dir / f'{config_id}.csv', index=False)

    # Cross-task SVD-GCCA
    views_svd = [StandardScaler().fit_transform(consensus_svd[task]) for task in task_names]
    svd_gcca = GCCA(n_components=1).fit(views_svd)
    svd_gcca_r = float(svd_gcca.score(views_svd))
    svd_gcca_correlations = pairwise_correlations(svd_gcca.transform(views_svd))[:, :, 0]
    svd_gcca_pair_r = svd_gcca_correlations[np.triu_indices(len(task_names), k=1)]
    svd_gcca_scores = np.column_stack([v[:, 0] for v in svd_gcca.transform(views_svd)])
    pd.concat([participants, pd.DataFrame(svd_gcca_scores, columns=task_names)], axis=1).to_csv(
        gcca_score_dir / f'{config_id}_svd.csv', index=False)

    # Compute pairwise CCA and CKA between tasks.
    for a,b in combinations(task_names,2):
        pair = f'{a}__{b}'

        # Pair-wise CCA
        cca = CCA(n_components=1).fit([consensus[a], consensus[b]])
        r = float(cca.score([consensus[a], consensus[b]]))
        cka = cka_metric.linear_CKA(consensus[a], consensus[b])

        # Pair-wise SVD-CCA
        svd_cca = CCA(n_components=1).fit([consensus_svd[a], consensus_svd[b]])
        svd_r = float(svd_cca.score([consensus_svd[a], consensus_svd[b]]))
        svd_cka = cka_metric.linear_CKA(consensus_svd[a], consensus_svd[b])

        # Save one row per configuration and task pair.
        cross_task_results.append(
            dict(config_id=config_id, task_a=a, task_b=b, task_pair=pair, n_participants=len(participants),
                 gcca_first_component_r=gcca_r, svd_gcca_first_component_r=svd_gcca_r,
                 pairwise_cca=r, pairwise_cka=cka, pairwise_svd_cca=svd_r, pairwise_svd_cka=svd_cka))

# Save the cross-task results to CSV
cross_task_by_pair = pd.DataFrame(cross_task_results)
cross_task_by_pair.to_csv(cross_task_dir / 'cca_by_task_pair.csv',index=False)

# Highest first-component CCA per task pair; config_id breaks exact ties.
best_cca_by_pair = (cross_task_by_pair.sort_values(['task_pair', 'pairwise_cca', 'config_id'], ascending=[True, False, True])
                   .drop_duplicates('task_pair').reset_index(drop=True))
best_cca_by_pair.to_csv(cross_task_dir / 'best_cca_by_pair.csv', index=False)

cross_task_by_config = cross_task_by_pair.groupby('config_id', as_index=False).agg(
    mean_pairwise_cca=('pairwise_cca','mean'), mean_pairwise_cka=('pairwise_cka','mean'),
    mean_svd_pairwise_cca=('pairwise_svd_cca','mean'), mean_svd_pairwise_cka=('pairwise_svd_cka','mean'),
    gcca_first_component_r=('gcca_first_component_r','first'), svd_gcca_first_component_r=('svd_gcca_first_component_r','first'))
cross_task_by_config.to_csv(cross_task_dir / 'cca_by_config.csv', index=False)

print(cross_task_by_config.sort_values(['mean_pairwise_cca', 'config_id'], ascending=[False, True]).reset_index(drop=True).head(5).to_string(index=False), flush=True)
print(f'Cross-task CCA saved to {cross_task_dir}',flush=True)

# ======================================================================================================================
# Typicality of the cross-task CCA associations across configurations
# ======================================================================================================================
# Reads saved cross-task metrics; does not retest models or refit CCA/GCCA.
typicality_data = pd.read_csv(cross_task_dir / "cca_by_task_pair.csv")
typicality_data = typicality_data.loc[typicality_data.config_id.isin(included_config_ids)].copy()

if typicality_data.config_id.nunique() < 2:
    raise ValueError("Typicality requires at least two configurations")

for representation, metric in [('original', 'pairwise_cca'), ('svd', 'pairwise_svd_cca')]:
    # Rows are configurations; columns are identically ordered task pairs.
    association_vectors = typicality_data.pivot(index='config_id', columns='task_pair', values=metric)
    association_values = association_vectors.to_numpy(dtype=float)
    config_similarity = np.corrcoef(association_values)
    np.fill_diagonal(config_similarity, np.nan)  # Exclude self-comparisons.
    config_similarity = np.clip(config_similarity, -1 + 1e-12, 1 - 1e-12)  # Avoid infinite Fisher z.
    typicality_z = np.arctanh(config_similarity)
    typicality_summary = pd.DataFrame({
        'config_id': association_vectors.index,
        'typicality_r': np.tanh(np.nanmean(typicality_z, axis=1)),
        'mean_cross_task_cca_r': association_values.mean(axis=1),
        'n_reference_configs': len(association_values) - 1,
    }).sort_values(['typicality_r', 'config_id'], ascending=[False, True]).reset_index(drop=True)
    association_vectors.to_csv(typicality_dir / f'{representation}_association_vectors.csv')
    typicality_summary.to_csv(typicality_dir / f'{representation}_typicality_ranking.csv', index=False)
    print(f'{representation}: most typical cross-task association patterns', flush=True)
    print(typicality_summary.head(20).to_string(index=False), flush=True)

# ======================================================================================================================
# Selected configuration: printing and plotting
# Change selected_config and rerun this section after the analyses finish.
# ======================================================================================================================
# Print the selected configuration's test performance for each task and overall.
test_results_all = pd.read_csv(test_results_path)
test_selected = test_results_all.loc[test_results_all.config_id.eq(selected_config)].groupby("task", as_index=False).agg(
test_nll=("test_nll", "mean"), test_accuracy=("test_accuracy", "mean"), pred_imbalance=("pred_imbalance", "mean"), pred_dist=("prevalence_distance", "mean"))
print(f'======================================================================================')
print("Selected configuration test performance:")
print(test_selected)

# Print the selected configuration's stability performance for each task and overall.
stability_results_all = pd.read_csv(stability_dir / 'stability_results_all.csv')
print(f'Selected configuration: {selected_config}', flush=True)
print('Stability performance by task:', flush=True)
stability_selected = stability_results_all.loc[stability_results_all.config_id.eq(selected_config)].groupby('task', as_index=False)[stability_metrics].mean()
print(stability_selected.to_string(index=False), flush=True)

# Print the selected configuration's cross-task convergence performance for each task pair and overall.
cross_task_by_pair = pd.read_csv(cross_task_dir / 'cca_by_task_pair.csv')
print('Cross-task convergence by pair:', flush=True)
print(cross_task_by_pair.loc[cross_task_by_pair.config_id.eq(selected_config)].to_string(index=False), flush=True)

for representation in ['original', 'svd']:
    typicality_summary = pd.read_csv(typicality_dir / f'{representation}_typicality_ranking.csv')
    print(f'{representation}: selected configuration typicality', flush=True)
    print(typicality_summary.loc[typicality_summary.config_id.eq(selected_config)].to_string(index=False), flush=True)

# # Plot test NLL and accuracy, separately sorted for each task and overall.
# figure_size = (7.2, 4.2)
# figure_dpi = 300
# plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
#
# stability_labels = {'first_component_r': 'Mean pairwise first-component r (GCCA)',
#                     'linear_cka': 'Mean pairwise linear CKA',
#                     'heldout_first_component_r': 'Mean participant-held-out first-component r',
#                     'first_component_svd_r': 'Mean pairwise first-component r (SVD-GCCA)',
#                     'linear_svd_cka': 'Mean pairwise linear CKA (SVD-GCCA)',
#                     'heldout_first_component_svd_r': 'Mean participant-held-out first-component r (SVD-GCCA)'}
#
# cross_task_labels = {'first_cca': 'First canonical correlation (r)',
#                      'first_svd_cca': 'First canonical correlation (r) after SVD',
#                      'gcca_first_component_r': 'Joint GCCA mean pairwise first-component r'}

# for name in list(tasks) + ['all_tasks']:
#     test_plot_data = test_by_config if name == 'all_tasks' else test_by_task.loc[test_by_task.task.eq(name)].copy()
#     if name != 'all_tasks':
#         test_plot_data.to_csv(output / f'{name}_test_results.csv', index=False)
#     for metric, ascending, ylabel in [('test_nll', True, 'Mean test NLL per valid trial'),
#                                       ('test_accuracy', False, 'Mean test accuracy'),
#                                       ('val_nll', True, 'Mean validation NLL per valid trial')]:
#         test_ranked = test_plot_data.sort_values([metric, 'config_id'], ascending=[ascending, True]).reset_index(drop=True)
#         test_ranked.insert(0, 'performance_rank', np.arange(1, len(test_ranked)+1))
#         fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
#         ax.plot(test_ranked.performance_rank, test_ranked[metric], color='#39739D')
#         for selected_rank in test_ranked.index[test_ranked.config_id.eq(selected_config)] + 1:
#             ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
#             ax.legend(frameon=False)
#         ax.set(title=name, xlabel='Configurations, best to worst', ylabel=ylabel)
#         ax.set_xticks([])
#         ax.spines[['top', 'right']].set_visible(False)
#         fig.savefig(test_plots / f'{name}_{metric}.png', dpi=figure_dpi)
#         plt.close(fig)
#
#
# for task in task_names + ['all_tasks']:
#     stability_plot_data = stability_by_config if task == 'all_tasks' else stability_results_all.loc[stability_results_all.task.eq(task)]
#     for metric, ylabel in stability_labels.items():
#         stability_ranked = stability_plot_data.sort_values([metric,'config_id'], ascending=[False,True]).reset_index(drop=True)
#         stability_ranked.insert(0,'performance_rank',np.arange(1,len(stability_ranked)+1))
#         stem = stability_plots / f'{task}_{metric}'
#         fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
#         ax.plot(stability_ranked.performance_rank, stability_ranked[metric], color='#39739D', linewidth=1.5)
#         for selected_rank in stability_ranked.index[stability_ranked.config_id.eq(selected_config)] + 1:
#             ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
#             ax.legend(frameon=False)
#         ax.set(title=f'{task} | split/seed stability | {len(stability_ranked)} configurations',
#                xlabel='Configurations, highest to lowest', ylabel=ylabel)
#         ax.set_xticks([])
#         ax.spines[['top','right']].set_visible(False)
#         ax.grid(axis='y', alpha=0.18)
#         fig.savefig(stem.with_suffix('.png'), dpi=figure_dpi)
#         plt.close(fig)
# print(f'Embedding stability saved to {stability_dir}', flush=True)
#
#
# # Pairwise first-CCA rankings and one joint GCCA ranking across configurations.
# for pair in list(cross_task_by_pair.task_pair.unique()) + ['all_tasks']:
#     metric = 'gcca_first_component_r' if pair == 'all_tasks' else 'first_cca'
#     cross_task_plot_data = cross_task_by_pair.drop_duplicates('config_id') if pair == 'all_tasks' else cross_task_by_pair.loc[cross_task_by_pair.task_pair.eq(pair)]
#     cross_task_ranked = cross_task_plot_data.sort_values([metric, 'config_id'], ascending=[False, True]).reset_index(drop=True)
#     title = f'Joint GCCA across {len(task_names)} tasks' if pair == 'all_tasks' else pair.replace('__', ' vs ')
#     fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
#     ax.plot(np.arange(1, len(cross_task_ranked) + 1), cross_task_ranked[metric], color='#39739D')
#     for selected_rank in cross_task_ranked.index[cross_task_ranked.config_id.eq(selected_config)] + 1:
#         ax.axvline(selected_rank, color='#D55E00', linestyle='--', linewidth=1.2, label='Selected configuration')
#         ax.legend(frameon=False)
#     ax.set(title=title, xlabel='Configurations, best to worst', ylabel=cross_task_labels[metric])
#     ax.set_xticks([])
#     ax.spines[['top', 'right']].set_visible(False)
#     fig.savefig(cross_task_plots / f'{pair}_{metric}.png', dpi=figure_dpi)
#     plt.close(fig)
#
# for representation in ['original', 'svd']:
#     typicality_summary = pd.read_csv(typicality_dir / f'{representation}_typicality_ranking.csv')
#     fig, ax = plt.subplots(figsize=figure_size, layout='constrained')
#     ax.plot(typicality_summary.index + 1, typicality_summary.typicality_r, color='#39739D')
#     for selected_rank in typicality_summary.index[typicality_summary.config_id.eq(selected_config)] + 1:
#         ax.axvline(selected_rank, color='#D55E00', linestyle='--', label='Selected configuration')
#         ax.legend(frameon=False)
#     ax.set(title=f'{representation} | typicality of cross-task associations',
#            xlabel='Configurations, most to least typical', ylabel='Fisher-z averaged Pearson r')
#     ax.set_xticks([])
#     ax.spines[['top', 'right']].set_visible(False)
#     fig.savefig(typicality_dir / f'{representation}_typicality.png', dpi=figure_dpi)
#     plt.close(fig)
