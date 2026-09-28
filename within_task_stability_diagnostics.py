"""Measure whether participant embeddings replicate across random trial splits."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
from sklearn.model_selection import KFold
from sklearn.preprocessing import KernelCenterer


RESULTS = Path("Results/HyperNN_Grid_Search")


def linear_cka(x, y):
    x_kernel = KernelCenterer().fit_transform(pairwise_kernels(x, metric="linear"))
    y_kernel = KernelCenterer().fit_transform(pairwise_kernels(y, metric="linear"))
    return float(cosine_similarity(x_kernel.reshape(1, -1), y_kernel.reshape(1, -1))[0, 0])


def first_cca(x, y):
    xs, ys = CCA(n_components=1, max_iter=2000).fit_transform(x, y)
    return float(pearsonr(xs[:, 0], ys[:, 0]).statistic)


def cv_first_cca(x, y):
    values = []
    for train, test in KFold(5, shuffle=True, random_state=42).split(x):
        model = CCA(n_components=1, max_iter=2000).fit(x[train], y[train])
        xs, ys = model.transform(x[test], y[test])
        values.append(pearsonr(xs[:, 0], ys[:, 0]).statistic)
    return float(np.mean(values))


summary = pd.read_csv(RESULTS / "configuration_summary.csv")
manifest = pd.read_csv(RESULTS / "training_manifest.csv")
config_id = summary.loc[summary["mean_cross_validated_first_cca"].idxmax(), "config_id"]
rows = []
for task, task_rows in manifest[manifest["config_id"].eq(config_id)].groupby("task"):
    embeddings = {
        int(row.random_split): np.load(row.embedding_path)
        for row in task_rows.itertuples()
    }
    for split_a, split_b in combinations(sorted(embeddings), 2):
        x, y = embeddings[split_a], embeddings[split_b]
        rows.append(
            {
                "config_id": config_id,
                "task": task,
                "split_a": split_a,
                "split_b": split_b,
                "linear_cka": linear_cka(x, y),
                "whole_sample_first_cca": first_cca(x, y),
                "cross_validated_first_cca": cv_first_cca(x, y),
                "rsa_spearman": float(
                    spearmanr(
                        pdist(x, metric="euclidean"),
                        pdist(y, metric="euclidean"),
                    ).statistic
                ),
            }
        )

detail = pd.DataFrame(rows)
task_summary = detail.groupby("task", as_index=False).agg(
    linear_cka_mean=("linear_cka", "mean"),
    linear_cka_sd=("linear_cka", "std"),
    whole_sample_first_cca_mean=("whole_sample_first_cca", "mean"),
    cross_validated_first_cca_mean=("cross_validated_first_cca", "mean"),
    cross_validated_first_cca_sd=("cross_validated_first_cca", "std"),
    rsa_spearman_mean=("rsa_spearman", "mean"),
)
detail.to_csv(RESULTS / "best_configuration_within_task_split_stability.csv", index=False)
task_summary.to_csv(
    RESULTS / "best_configuration_within_task_split_stability_summary.csv", index=False
)
print(task_summary.to_string(index=False))
print("\nOVERALL")
print(detail[["linear_cka", "whole_sample_first_cca", "cross_validated_first_cca", "rsa_spearman"]].agg(["mean", "std"]).to_string())
