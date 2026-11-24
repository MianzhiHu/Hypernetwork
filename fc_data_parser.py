from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
import bct
from collections import defaultdict
from task_config import selected_config

task_name_mapping = {
    'CCTHot': 'CCT',
    'discountFix': 'DD',
    'motorSelectiveStop': 'Motor',
    'stopSignal': 'StopSignal',
    'twoByTwo': 'TwoByTwo',
}


def to_dfs(data):
    return {
        task: pd.DataFrame.from_dict(subjects, orient='index').rename_axis('worker_id').reset_index()
        for task, subjects in data.items()
    }

def fc_data_parser(fc_results, pls_data_path=Path('./Data/PLS_data/')):
    task_fc_edges = defaultdict(dict)
    task_fc_pos = defaultdict(dict)
    task_fc_neg = defaultdict(dict)
    task_fc_overall = defaultdict(dict)

    # Read in task relevant FC data
    for key, record in fc_results['runs'].items():
        subject = record['worker_id']
        task = record['task']
        session = record['session']
        run = record['run']

        task_fc_edges[task].setdefault(subject, {}).setdefault(session, {})[run] = record['fisher_z_edges']
        task_fc_pos[task].setdefault(subject, {}).setdefault(session, {})[run] = record['positive_strength']
        task_fc_neg[task].setdefault(subject, {}).setdefault(session, {})[run] = record['negative_strength']
        task_fc_overall[task].setdefault(subject, {}).setdefault(session, {})[run] = record['overall_strength']

    # In cases where a subject completed the same task (mostly resting state) in multiple sessions,
    # average across sessions to get a single FC representation per subject per task
    for data in (task_fc_edges, task_fc_pos, task_fc_neg, task_fc_overall):
        for task, subjects in data.items():
            for subject, sessions in subjects.items():
                runs = [values for session_runs in sessions.values() for values in session_runs.values()]
                data[task][subject] = np.nanmean(np.stack(runs), axis=0)

    # Convert the nested dictionaries to DataFrames
    task_fc_edges = to_dfs(task_fc_edges)
    task_fc_pos = to_dfs(task_fc_pos)
    task_fc_neg = to_dfs(task_fc_neg)
    task_fc_overall = to_dfs(task_fc_overall)

    # Record missing nodes/edges for each task
    missing_features = defaultdict(dict)
    fc_filtered = defaultdict(dict)

    # Filter the DataFrames to remove rows with any NaN values
    for task in task_fc_edges.keys():
        edges_df = task_fc_edges[task]
        pos_df = task_fc_pos[task]
        neg_df = task_fc_neg[task]
        overall_df = task_fc_overall[task]

        # Identify columns with all NaN values
        invalid_edge_columns = edges_df.columns[edges_df.isna().any()]
        invalid_pos_columns = pos_df.columns[pos_df.isna().any()]
        invalid_neg_columns = neg_df.columns[neg_df.isna().any()]
        invalid_overall_columns = overall_df.columns[overall_df.isna().any()]

        # pos, neg, and overall strength should have the same invalid columns
        # Check that the invalid columns are consistent across all three DataFrames
        if not (invalid_pos_columns.equals(invalid_neg_columns) and invalid_pos_columns.equals(invalid_overall_columns)):
            raise ValueError(f"Inconsistent invalid columns for task {task}")

        # Record the invalid columns for reference
        missing_features[task] = {
            'edges': invalid_edge_columns.tolist(),
            'nodes': invalid_pos_columns.tolist()  # pos, neg, and overall strength have the same invalid columns
        }

        # Filter the DataFrames to keep only valid columns
        task_fc_edges[task] = edges_df.drop(columns=invalid_edge_columns)
        task_fc_pos[task] = pos_df.drop(columns=invalid_pos_columns)
        task_fc_neg[task] = neg_df.drop(columns=invalid_neg_columns)
        task_fc_overall[task] = overall_df.drop(columns=invalid_overall_columns)
        fc_filtered[task] = {
            'edges': task_fc_edges[task],
            'pos': task_fc_pos[task],
            'neg': task_fc_neg[task],
            'overall': task_fc_overall[task]
        }

        # Save the filtered DataFrames to CSV files
        task_fc_edges[task].to_csv(pls_data_path / f"{task}_edges_{selected_config}.csv", index=False)
        task_fc_pos[task].to_csv(pls_data_path / f"{task}_pos_{selected_config}.csv", index=False)
        task_fc_neg[task].to_csv(pls_data_path / f"{task}_neg_{selected_config}.csv", index=False)
        task_fc_overall[task].to_csv(pls_data_path / f"{task}_overall_{selected_config}.csv", index=False)

    # Save the missing features information and filtered DataFrames to a pickle file
    with open(pls_data_path.parent / f"fc_filtered.pkl", "wb") as handle:
        pickle.dump(
            {
                'missing_features': missing_features,
                'fc_filtered': fc_filtered,
                'edge_indices': fc_results['edge_indices'],
                'edge_index_base': fc_results['edge_index_base'],
                'roi_ids': np.arange(1, fc_results['n_rois'] + 1),
            }, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )

def match_embeddings_with_fc(embedding_path, fc_data, task, pls_data_path=Path('./Data/PLS_data/'),
                             prefix='single_task', embedding='full'):

    # Load the FC metrics and embeddings for the specified task
    metric_dfs = fc_data['fc_filtered'][task]
    embedding_task = task_name_mapping[task].lower()

    if embedding == 'cv':
        embedding_data = pd.read_csv(embedding_path / f"{selected_config}_{embedding_task}.csv")

    elif embedding == 'full':
        # Load the CV embeddings for the specified task
        embedding_data = pd.DataFrame(np.load(embedding_path / f"{embedding_task}_embeddings.npy", allow_pickle=False))
        embedding_data = embedding_data.drop(columns=['participant_id'])

        # Make the index a column for merging
        embedding_data['participant_id'] = embedding_data.index
        participants = pd.read_csv(embedding_path / f"{embedding_task}_participant_map.csv")
        embedding_data = embedding_data.merge(participants, on='participant_id', how='inner', sort=False, validate='one_to_one')
        embedding_data = embedding_data.drop(columns=['participant_id'])

    elif embedding == 'pca':
        # Load the PCA embeddings for the specified task
        embedding_data = pd.read_csv(embedding_path / f"{embedding_task}_pca_results.csv")

    # Keep workers present in embeddings and every FC metric, preserving FC order.
    common_workers = metric_dfs['edges'][['worker_id']]
    embedding_data = common_workers.merge(embedding_data, on='worker_id', how='inner', sort=False, validate='one_to_one')

    # Drop the worker_id column from the embedding data to avoid redundancy
    embedding_data = embedding_data.drop(columns=['worker_id'])

    # Remove the participant ID column from the embedding data
    embedding_data.to_csv(pls_data_path / f"{prefix}_{task}_embeddings_{selected_config}.csv", index=False)

if __name__ == "__main__":

    fc_path = Path('./Data/shen268_fc.pkl')
    pls_data_path = Path('./Data/PLS_data/')
    os.makedirs(pls_data_path, exist_ok=True)

    # Read in the functional connectivity data
    with fc_path.open("rb") as handle:
        fc_results = pickle.load(handle)

    # # Parse the functional connectivity data and save to CSV and pickle files
    # fc_data_parser(fc_results, pls_data_path)

    # Match task embeddings and all FC metrics by worker_id.
    # embedding_path = Path(f'./Results/Final_Models/{selected_config}')
    embedding_path = Path(f'./Results/HyperNN_Grid_Search_Evaluation/Cross_task_Convergence/Consensus_Embeddings')
    filtered_fc_path = pls_data_path.parent / f"fc_filtered.pkl"
    with filtered_fc_path.open("rb") as handle:
        fc_data = pickle.load(handle)

    # Match embeddings with FC metrics for a specific task
    for task in ['CCTHot', 'discountFix', 'motorSelectiveStop', 'stopSignal', 'twoByTwo']:
        match_embeddings_with_fc(embedding_path, fc_data, task, pls_data_path, embedding='cv')