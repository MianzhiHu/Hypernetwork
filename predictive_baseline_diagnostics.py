"""Compare saved HyperNN validation NLL with a train-prevalence baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

import hypernn_grid_search as training


RESULTS = Path("Results/HyperNN_Grid_Search")
EPSILON = 1e-8

summary = pd.read_csv(RESULTS / "configuration_summary.csv")
manifest = pd.read_csv(RESULTS / "training_manifest.csv")
validation = pd.read_csv(RESULTS / "task_validation_results.csv")
task_lookup = {task["name"]: task for task in training.tasks}

# Splits are shared by all configurations, so use one manifest row per task/split.
split_rows = manifest.drop_duplicates(["task", "random_split"])
baseline_rows = []
for row in split_rows.itertuples():
    task = task_lookup[row.task]
    assignments = pd.read_csv(row.split_assignment_path)[
        ["participant_id", "trial", row.split_column]
    ]
    data = task["data"].merge(
        assignments,
        on=["participant_id", "trial"],
        how="inner",
        validate="one_to_one",
    )
    valid = data["choice_mask"].astype(bool) & data[task["y_col"]].isin([0, 1])
    train = data[row.split_column].eq("train") & valid
    held_out = data[row.split_column].eq("validation") & valid
    p_one = float(np.clip(data.loc[train, task["y_col"]].mean(), EPSILON, 1 - EPSILON))
    y = data.loc[held_out, task["y_col"]].to_numpy(dtype=float)
    nll = float(-(y * np.log(p_one) + (1 - y) * np.log(1 - p_one)).mean())
    prediction = int(p_one >= 0.5)
    baseline_rows.append(
        {
            "task": row.task,
            "random_split": int(row.random_split),
            "training_prevalence": p_one,
            "validation_nll_baseline": nll,
            "validation_accuracy_baseline": float((y == prediction).mean()),
            "validation_trials": len(y),
        }
    )

baseline = pd.DataFrame(baseline_rows)
baseline.to_csv(RESULTS / "train_prevalence_validation_baseline.csv", index=False)

candidate_ids = {
    "raw_cka_winner": summary.loc[summary["mean_pairwise_linear_cka"].idxmax(), "config_id"],
    "cv_cca_winner": summary.loc[
        summary["mean_cross_validated_first_cca"].idxmax(), "config_id"
    ],
    "nll_winner": summary.loc[summary["mean_validation_nll"].idxmin(), "config_id"],
}
comparison_rows = []
for label, config_id in candidate_ids.items():
    model_rows = validation[validation["config_id"].eq(config_id)]
    compared = model_rows.merge(baseline, on=["task", "random_split"], validate="one_to_one")
    compared["nll_improvement"] = (
        compared["validation_nll_baseline"] - compared["validation_nll"]
    )
    compared["accuracy_improvement"] = (
        compared["validation_accuracy"] - compared["validation_accuracy_baseline"]
    )
    for task, rows in compared.groupby("task"):
        comparison_rows.append(
            {
                "selection_label": label,
                "config_id": config_id,
                "task": task,
                "model_validation_nll": rows["validation_nll"].mean(),
                "baseline_validation_nll": rows["validation_nll_baseline"].mean(),
                "nll_improvement": rows["nll_improvement"].mean(),
                "model_validation_accuracy": rows["validation_accuracy"].mean(),
                "baseline_validation_accuracy": rows[
                    "validation_accuracy_baseline"
                ].mean(),
                "accuracy_improvement": rows["accuracy_improvement"].mean(),
            }
        )

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(RESULTS / "candidate_predictive_baseline_comparison.csv", index=False)
print(comparison[comparison["selection_label"].eq("cv_cca_winner")].to_string(index=False))
