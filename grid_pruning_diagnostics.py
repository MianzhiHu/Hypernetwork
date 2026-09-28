"""Matched comparisons for pruning a second-stage HyperNN grid."""

from pathlib import Path

import pandas as pd


RESULTS = Path("Results/HyperNN_Grid_Search")
SUMMARY = pd.read_csv(RESULTS / "configuration_summary.csv")
METRICS = {
    "mean_pairwise_linear_cka": "higher",
    "mean_cross_validated_first_cca": "higher",
    "mean_validation_nll": "lower",
}


def paired_comparison(name, left_filter, right_filter, match_columns):
    left = SUMMARY.loc[left_filter].copy()
    right = SUMMARY.loc[right_filter].copy()
    merged = left.merge(right, on=match_columns, suffixes=("_left", "_right"), validate="one_to_one")
    records = []
    for metric, direction in METRICS.items():
        difference = merged[f"{metric}_left"] - merged[f"{metric}_right"]
        left_better = difference > 0 if direction == "higher" else difference < 0
        records.append(
            {
                "comparison": name,
                "metric": metric,
                "n_matched": len(merged),
                "left_minus_right_mean": difference.mean(),
                "left_better_fraction": left_better.mean(),
                "left_better_count": int(left_better.sum()),
            }
        )
    return records


rows = []

# Mapping: linear versus each nonlinear width, matched on architecture.
mapping_match = ["num_hidden_layers", "hidden_dims", "rank", "participant_emb_dim"]
for nodes in [4.0, 8.0]:
    rows += paired_comparison(
        f"linear_vs_nonlinear_nodes_{int(nodes)}",
        SUMMARY["mapping"].eq("linear"),
        SUMMARY["mapping"].eq("nonlinear") & SUMMARY["hyper_hidden_dim"].eq(nodes),
        mapping_match,
    )

# Nonlinear hidden width 8 versus 4.
rows += paired_comparison(
    "nonlinear_nodes_8_vs_4",
    SUMMARY["mapping"].eq("nonlinear") & SUMMARY["hyper_hidden_dim"].eq(8),
    SUMMARY["mapping"].eq("nonlinear") & SUMMARY["hyper_hidden_dim"].eq(4),
    mapping_match,
)

# Full update versus rank 2, matched on everything else.
rank_match = [
    "num_hidden_layers",
    "hidden_dims",
    "participant_emb_dim",
    "mapping",
    "hyper_hidden_dim",
]
rows += paired_comparison(
    "full_rank_vs_rank_2",
    SUMMARY["rank"].isna(),
    SUMMARY["rank"].eq(2),
    rank_match,
)

# Four versus eight embedding dimensions.
embedding_match = [
    "num_hidden_layers",
    "hidden_dims",
    "rank",
    "mapping",
    "hyper_hidden_dim",
]
rows += paired_comparison(
    "embedding_4_vs_8",
    SUMMARY["participant_emb_dim"].eq(4),
    SUMMARY["participant_emb_dim"].eq(8),
    embedding_match,
)

# One versus two main-network hidden layers of the same width.
depth_data = SUMMARY.copy()
depth_data["width"] = depth_data["hidden_dims"].str.split("-").str[0].astype(int)
depth_match = ["width", "rank", "participant_emb_dim", "mapping", "hyper_hidden_dim"]
old_summary = SUMMARY
SUMMARY = depth_data
rows += paired_comparison(
    "one_layer_vs_two_layers",
    SUMMARY["num_hidden_layers"].eq(1),
    SUMMARY["num_hidden_layers"].eq(2),
    depth_match,
)

# Width comparisons, matched on every other factor.
width_match = [
    "num_hidden_layers",
    "rank",
    "participant_emb_dim",
    "mapping",
    "hyper_hidden_dim",
]
for left_width, right_width in [(8, 16), (8, 32), (16, 32)]:
    rows += paired_comparison(
        f"width_{left_width}_vs_{right_width}",
        SUMMARY["width"].eq(left_width),
        SUMMARY["width"].eq(right_width),
        width_match,
    )

comparison = pd.DataFrame(rows)
comparison.to_csv(RESULTS / "grid_pruning_matched_comparisons.csv", index=False)
print(comparison.to_string(index=False))
