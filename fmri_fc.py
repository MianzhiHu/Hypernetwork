"""Calculate per-run Pearson FC from cleaned Shen-268 regional time series."""

from pathlib import Path
import pickle
import numpy as np
import bct


def calculate_functional_connectivity(runs):
    """Keep the parcellation's column order and leave invalid-parcel edges as NaN."""
    n_rois = 268
    edge_rows, edge_cols = np.triu_indices(n_rois, k=1)
    fc_runs = {}

    for index, (key, record) in enumerate(parcellated_runs.items(), start=1):
        print(f"[{index}/{len(parcellated_runs)}] {key}: {record['time_series'].shape[0]} volumes", flush=True)
        time_series = np.asarray(record["time_series"])

        # Exclude missing parcels first
        valid_rois = ~np.isin(np.arange(1, n_rois + 1), record["missing_roi_ids"])
        valid_columns = np.flatnonzero(valid_rois)

        # Calculate the Pearson correlation matrix
        valid_fc = np.corrcoef(time_series[:, valid_rois], rowvar=False)
        fc_matrix = np.full((n_rois, n_rois), np.nan, dtype=np.float64)
        fc_matrix[np.ix_(valid_columns, valid_columns)] = valid_fc

        # Fisher-z transformation of the correlation coefficients
        fc_matrix_z = np.arctanh(fc_matrix)
        np.fill_diagonal(fc_matrix_z, 1.00)

        # Take edges from the upper triangle of the Fisher-z transformed matrix
        edges = fc_matrix_z[edge_rows, edge_cols]

        # Now take averaged FC across brain nodes
        fc_matrix_z_adj = fc_matrix_z - np.eye(fc_matrix_z.shape[0])
        W = fc_matrix_z_adj[np.ix_(valid_columns, valid_columns)]
        pos, neg, _, _ = bct.strengths_und_sign(W)
        overall = bct.strengths_und(W)

        # Restore the full ROI layout afterward.
        positive_strength = np.full(n_rois, np.nan)
        negative_strength = np.full(n_rois, np.nan)
        overall_strength = np.full(n_rois, np.nan)

        positive_strength[valid_columns] = pos
        negative_strength[valid_columns] = neg
        overall_strength[valid_columns] = overall

        retained_indices = np.asarray(record["retained_volume_indices"], dtype=int)
        if retained_indices.ndim != 1 or retained_indices.size != time_series.shape[0]:
            raise ValueError(f"{key}: retained indices do not match time-series rows")

        fc_runs[key] = {
            "worker_id": record["worker_id"],
            "task": record["task"],
            "session": record["session"],
            "run": record["run"],
            "fc_matrix": fc_matrix.astype(np.float32),
            "fc_matrix_z": fc_matrix_z.astype(np.float32),
            "fisher_z_edges": edges.astype(np.float32),
            "positive_strength": positive_strength.astype(np.float32),
            "negative_strength": negative_strength.astype(np.float32),
            "overall_strength": overall_strength.astype(np.float32),
            "valid_roi_mask": valid_rois,
            "invalid_roi_columns": np.flatnonzero(~valid_rois),
            "n_retained_volumes": time_series.shape[0],
            "retained_volume_indices": retained_indices,
            "tr": record["tr"],
            "bold_path": record["bold_path"],
        }

    if not fc_runs:
        raise ValueError("The parcellation pickle contains no runs")

    return {
        "atlas": "shen268",
        "method": "Empirical Pearson correlation of stored cleaned/censored time_series",
        "n_rois": n_rois,
        "edge_indices": np.column_stack((edge_rows, edge_cols)),
        "edge_index_base": 0,
        "runs": fc_runs,
    }


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "Data" / "shen268_parcellated.pkl"
    output_path = project_dir / "Data" / "shen268_fc.pkl"

    print(f"Loading {input_path}", flush=True)
    with input_path.open("rb") as handle:
        parcellated_runs = pickle.load(handle)

    fc_results = calculate_functional_connectivity(parcellated_runs)
    fc_results["source_pickle"] = str(input_path)
    del parcellated_runs

    # save the results to a new pickle file
    print(f"Saving {output_path}", flush=True)
    with output_path.open("wb") as handle:
        pickle.dump(fc_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved connectivity for {len(fc_results['runs'])} runs: {output_path}", flush=True)