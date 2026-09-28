"""Participant-label permutation test for aggregate cross-validated CCA."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold


RESULTS = Path("Results/HyperNN_Grid_Search")
N_PERMUTATIONS = 200
SEED = 20260902


def cv_first_cca(x, y, folds):
    values = []
    for train, test in folds:
        model = CCA(n_components=1, max_iter=2000).fit(x[train], y[train])
        xs, ys = model.transform(x[test], y[test])
        values.append(pearsonr(xs[:, 0], ys[:, 0]).statistic)
    return float(np.mean(values))


summary = pd.read_csv(RESULTS / "configuration_summary.csv")
manifest = pd.read_csv(RESULTS / "training_manifest.csv")
pairs = pd.read_csv(RESULTS / "pairwise_convergence.csv")
config_id = summary.loc[
    summary["mean_cross_validated_first_cca"].idxmax(), "config_id"
]
config_manifest = manifest[manifest["config_id"].eq(config_id)]
embeddings_by_split = {
    int(random_split): {
        row.task: np.load(row.embedding_path)
        for row in split_rows.itertuples()
    }
    for random_split, split_rows in config_manifest.groupby("random_split")
}
n_participants = next(iter(next(iter(embeddings_by_split.values())).values())).shape[0]
folds = list(KFold(5, shuffle=True, random_state=42).split(np.arange(n_participants)))
observed = float(
    pairs[pairs["config_id"].eq(config_id)]["cross_validated_first_cca"].mean()
)

rng = np.random.default_rng(SEED)
null = np.empty(N_PERMUTATIONS)
for permutation_index in range(N_PERMUTATIONS):
    values = []
    for task_embeddings in embeddings_by_split.values():
        tasks = list(task_embeddings)
        shuffled = {
            task: task_embeddings[task][rng.permutation(n_participants)] for task in tasks
        }
        for task_a, task_b in combinations(tasks, 2):
            values.append(cv_first_cca(shuffled[task_a], shuffled[task_b], folds))
    null[permutation_index] = np.mean(values)

result = pd.DataFrame(
    [
        {
            "config_id": config_id,
            "n_permutations": N_PERMUTATIONS,
            "observed_mean_cross_validated_first_cca": observed,
            "permutation_mean": float(null.mean()),
            "permutation_sd": float(null.std(ddof=1)),
            "permutation_95_percentile": float(np.quantile(null, 0.95)),
            "permutation_p_one_sided": float(
                (1 + np.count_nonzero(null >= observed)) / (N_PERMUTATIONS + 1)
            ),
        }
    ]
)
result.to_csv(RESULTS / "best_configuration_cv_cca_permutation.csv", index=False)
np.save(RESULTS / "best_configuration_cv_cca_permutation_null.npy", null)
print(result.to_string(index=False))
