"""Participant-label permutation checks for selected HyperNN configurations.

The raw linear CKA baseline depends on sample size and representation dimension.
This script estimates that baseline by breaking participant correspondence while
preserving each task embedding's marginal geometry.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("Results/HyperNN_Grid_Search")
N_PERMUTATIONS = 1000
SEED = 20260902


def prepare(x):
    centered = x - x.mean(axis=0, keepdims=True)
    self_product = centered.T @ centered
    denominator_part = np.sqrt(np.sum(self_product * self_product))
    return centered, denominator_part


def feature_linear_cka(prepared_x, prepared_y, permutation=None):
    x, x_norm = prepared_x
    y, y_norm = prepared_y
    if permutation is not None:
        y = y[permutation]
    cross = x.T @ y
    return float(np.sum(cross * cross) / (x_norm * y_norm))


summary = pd.read_csv(RESULTS / "configuration_summary.csv")
manifest = pd.read_csv(RESULTS / "training_manifest.csv")
pairs = pd.read_csv(RESULTS / "pairwise_convergence.csv")

selected = {
    "raw_cka_winner": summary.loc[summary["mean_pairwise_linear_cka"].idxmax(), "config_id"],
    "cv_cca_winner": summary.loc[
        summary["mean_cross_validated_first_cca"].idxmax(), "config_id"
    ],
    "rsa_winner": summary.loc[summary["mean_rsa_spearman"].idxmax(), "config_id"],
    "nll_winner": summary.loc[summary["mean_validation_nll"].idxmin(), "config_id"],
}

rng = np.random.default_rng(SEED)
result_rows = []
pair_rows = []

for selection_label, config_id in selected.items():
    config_manifest = manifest[manifest["config_id"].eq(config_id)]
    prepared_by_split = {}
    for random_split, split_rows in config_manifest.groupby("random_split"):
        prepared_by_split[int(random_split)] = {
            row["task"]: prepare(np.load(row["embedding_path"]))
            for _, row in split_rows.iterrows()
        }

    observed_pairs = pairs[pairs["config_id"].eq(config_id)]
    observed_global = float(observed_pairs["linear_cka"].mean())
    observed_by_pair = observed_pairs.groupby(["task_a", "task_b"])["linear_cka"].mean()
    pair_keys = list(observed_by_pair.index)

    null_global = np.empty(N_PERMUTATIONS)
    null_by_pair = {pair: np.empty(N_PERMUTATIONS) for pair in pair_keys}
    for permutation_index in range(N_PERMUTATIONS):
        all_values = []
        values_by_pair = {pair: [] for pair in pair_keys}
        for random_split, task_data in prepared_by_split.items():
            tasks = list(task_data)
            n_participants = task_data[tasks[0]][0].shape[0]
            task_permutations = {
                task: rng.permutation(n_participants) for task in tasks
            }
            for task_a, task_b in combinations(tasks, 2):
                value = feature_linear_cka(
                    task_data[task_a],
                    task_data[task_b],
                    permutation=task_permutations[task_b],
                )
                all_values.append(value)
                values_by_pair[(task_a, task_b)].append(value)
        null_global[permutation_index] = np.mean(all_values)
        for pair in pair_keys:
            null_by_pair[pair][permutation_index] = np.mean(values_by_pair[pair])

    result_rows.append(
        {
            "selection_label": selection_label,
            "config_id": config_id,
            "participant_emb_dim": int(config_manifest.iloc[0]["participant_emb_dim"]),
            "observed_mean_cka": observed_global,
            "permutation_mean_cka": float(null_global.mean()),
            "permutation_sd": float(null_global.std(ddof=1)),
            "excess_over_permutation": float(observed_global - null_global.mean()),
            "standardized_excess": float(
                (observed_global - null_global.mean()) / null_global.std(ddof=1)
            ),
            "permutation_p_one_sided": float(
                (1 + np.count_nonzero(null_global >= observed_global))
                / (N_PERMUTATIONS + 1)
            ),
        }
    )

    for pair in pair_keys:
        null = null_by_pair[pair]
        observed = float(observed_by_pair.loc[pair])
        pair_rows.append(
            {
                "selection_label": selection_label,
                "config_id": config_id,
                "task_a": pair[0],
                "task_b": pair[1],
                "observed_mean_cka": observed,
                "permutation_mean_cka": float(null.mean()),
                "excess_over_permutation": float(observed - null.mean()),
                "permutation_p_one_sided": float(
                    (1 + np.count_nonzero(null >= observed))
                    / (N_PERMUTATIONS + 1)
                ),
            }
        )

results = pd.DataFrame(result_rows)
pair_results = pd.DataFrame(pair_rows)
results.to_csv(RESULTS / "selected_configuration_cka_permutation.csv", index=False)
pair_results.to_csv(RESULTS / "selected_configuration_pairwise_cka_permutation.csv", index=False)
print(results.to_string(index=False))
print("\nPAIRWISE RESULTS FOR CV-CCA WINNER")
print(
    pair_results[pair_results["selection_label"].eq("cv_cca_winner")]
    .sort_values("excess_over_permutation", ascending=False)
    .to_string(index=False)
)
