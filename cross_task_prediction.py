from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import hypernn_grid_search as training
from HyperNetwork import HyperNN, BehavioralDataset, behavioral_collate_fn, test_model

# Settings
root = Path('./Results/HyperNN_Grid_Search')
selected_config = 'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8'
output = Path('./Results/Cross_Task_Prediction') / selected_config / 'ridge_transfer'
batch_size = 16
n_splits = 10  # Existing trial-fold rotations.
n_participant_folds = 10  # Separate folds for fitting the embedding mapping.
participant_seed = 42
ridge_alphas = np.logspace(-6, 6, 13)  # Selected using inner leave-one-participant-out CV.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trial_keys = ['worker_id', 'participant_id', 'trial']

# Align source -> target embeddings on other participants only.
# Ridge penalty is chosen by embedding MSE using only outer-training participants.
# RidgeCV uses efficient leave-one-out CV; no target test outcomes tune alpha.
# Models stay frozen; this tests mapping generalization within the existing cohort,
# not participants excluded from the original neural-network training.
tasks = {task['name']: task for task in training.tasks}
participant_map = pd.read_csv(root / 'participant_map.csv').sort_values('participant_id').reset_index(drop=True)
participant_folds = list(KFold(n_splits=n_participant_folds, shuffle=True, random_state=participant_seed).split(participant_map))
participant_assignments = participant_map.copy()
participant_assignments['mapping_fold'] = -1
for fold, (_, heldout_ids) in enumerate(participant_folds):
    participant_assignments.loc[heldout_ids, 'mapping_fold'] = fold
loaders = {}

# Reuse the target task's original test trials for every source task.
for name, task in tasks.items():
    data = task['data']
    current_map = data[['participant_id', 'worker_id']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    if not current_map.astype({"participant_id": "int64"}).equals(participant_map) or not np.array_equal(current_map.participant_id, np.arange(len(current_map))):
        raise ValueError(f'Participant mapping changed for {name}')
    saved = pd.read_csv(root / 'random_splits' / f'{name}_random_splits.csv')
    aligned = data[trial_keys].merge(saved, on=trial_keys, how='left', sort=False, validate='one_to_one', indicator=True)
    if len(saved) != len(data) or not aligned['_merge'].eq('both').all():
        raise ValueError(f'Saved trials do not match current data for {name}')
    if not aligned.fold.isin(range(n_splits)).all():
        raise ValueError(f'Invalid folds for {name}')
    for split in range(n_splits):
        expected = np.select([aligned.fold.eq((split + 1) % n_splits), aligned.fold.eq(split)], ['validation', 'test'], default='train')
        if not np.array_equal(aligned[f'random_split_{split}'], expected):
            raise ValueError(f'Invalid split labels for {name}/{split}')
        selected = aligned[f'random_split_{split}'].eq('test').to_numpy()
        dataset = BehavioralDataset(data.iloc[selected].copy(), task['x_col'], task['y_col'])
        loaders[name, split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)

transfer_records = []
baseline_records = []
mapping_records = []
for split in range(n_splits):
    split_dir = root / selected_config / f'random_split_{split}'
    embeddings = {}
    for name in tasks:
        values = np.load(split_dir / f'{name}_embeddings.npy', allow_pickle=False)
        if values.shape != (len(participant_map), 4) or not np.isfinite(values).all():
            raise ValueError(f'Invalid embeddings for {name}/{split}')
        embeddings[name] = values.astype(np.float64)

    for target, task in tasks.items():
        checkpoint = torch.load(split_dir / f'{target}_model.pt', map_location='cpu', weights_only=True)
        config = checkpoint['extra_config']
        if (config['random_split'] != split or config['participant_emb_dim'] != 4
                or list(config['hidden_dims']) != [8] or config['rank'] is not None
                or config['mapping'] != 'nonlinear' or config['hyper_hidden_dim'] != 8
                or config['num_hidden_layers'] != 1):
            raise ValueError(f'Unexpected model configuration for {target}/{split}')
        model = HyperNN(n_participants=len(participant_map), input_dim=len(task['x_col']),
                        hidden_dim=tuple(config['hidden_dims']), output_dim=2,
                        participant_emb_dim=config['participant_emb_dim'], hyper_hidden_dim=config['hyper_hidden_dim'],
                        rank=config['rank'], shared_right=config['shared_right'], mapping=config['mapping'],
                        num_hidden_layers=config['num_hidden_layers']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if not np.array_equal(model.participant_embedding.weight.detach().cpu().numpy(), embeddings[target]):
            raise ValueError(f'Saved embeddings do not match checkpoint for {target}/{split}')
        baseline = test_model(model, loaders[target, split], device)
        if baseline['n_trials'] <= 0 or not np.isfinite([baseline['raw_loss'], baseline['acc']]).all():
            raise ValueError(f'Invalid baseline results for {target}/{split}')
        # Each participant receives a mean computed without their participant fold.
        mean_embeddings = np.empty_like(embeddings[target])
        for train_ids, heldout_ids in participant_folds:
            mean_embeddings[heldout_ids] = embeddings[target][train_ids].mean(axis=0)
        with torch.no_grad():
            model.participant_embedding.weight.copy_(torch.as_tensor(mean_embeddings, dtype=torch.float32, device=device))
        mean_baseline = test_model(model, loaders[target, split], device)
        if mean_baseline['n_trials'] != baseline['n_trials'] or not np.isfinite([mean_baseline['raw_loss'], mean_baseline['acc']]).all():
            raise ValueError(f'Invalid mean-embedding results for {target}/{split}')
        baseline_records.append({'target_task': target, 'random_split': split,
                                 'original_test_nll': baseline['raw_loss'], 'original_test_accuracy': baseline['acc'],
                                 'mean_test_nll': mean_baseline['raw_loss'], 'mean_test_accuracy': mean_baseline['acc'],
                                 'n_test_trials': baseline['n_trials']})

        for source in tasks:
            if source == target:
                continue
            predicted_embeddings = np.empty_like(embeddings[target])
            for fold, (train_ids, heldout_ids) in enumerate(participant_folds):
                mapping = RidgeCV(alphas=ridge_alphas, scoring='neg_mean_squared_error').fit(
                    embeddings[source][train_ids], embeddings[target][train_ids])
                predicted_embeddings[heldout_ids] = mapping.predict(embeddings[source][heldout_ids])
                mapping_records.append({'source_task': source, 'target_task': target,
                                        'random_split': split, 'mapping_fold': fold,
                                        'alpha': float(mapping.alpha_)})
            if not np.isfinite(predicted_embeddings).all():
                raise ValueError(f'Invalid mapped embeddings for {source} -> {target}/{split}')
            # Every row is an out-of-fold prediction. Evaluating once pools test
            # trials across participant folds rather than averaging unequal folds.
            with torch.no_grad():
                model.participant_embedding.weight.copy_(torch.as_tensor(predicted_embeddings, dtype=torch.float32, device=device))
            metrics = test_model(model, loaders[target, split], device)
            if metrics['n_trials'] != baseline['n_trials'] or not np.isfinite([metrics['raw_loss'], metrics['acc']]).all():
                raise ValueError(f'Invalid transfer results for {source} -> {target}/{split}')
            transfer_records.append({
                'source_task': source, 'target_task': target, 'random_split': split,
                'test_nll': metrics['raw_loss'], 'test_accuracy': metrics['acc'], 'n_test_trials': metrics['n_trials'],
                'original_test_nll': baseline['raw_loss'], 'original_test_accuracy': baseline['acc'],
                'mean_test_nll': mean_baseline['raw_loss'], 'mean_test_accuracy': mean_baseline['acc'],
                'transfer_gain_nll': mean_baseline['raw_loss'] - metrics['raw_loss'],
                'transfer_gain_accuracy': metrics['acc'] - mean_baseline['acc'],
                'nll_change': metrics['raw_loss'] - baseline['raw_loss'],
                'accuracy_change': metrics['acc'] - baseline['acc'],
            })
        print(f'Split {split + 1}/{n_splits}: evaluated all sources on {target}', flush=True)

# Average rotations equally, matching hypernn_grid_search_evaluation.py.
# Positive transfer gain means improvement over the mean-embedding baseline.
# NLL change compares transfer against the original target embedding (lower is better).
transfer_by_split = pd.DataFrame(transfer_records)
baseline_by_split = pd.DataFrame(baseline_records)
transfer_by_pair = transfer_by_split.groupby(['source_task', 'target_task'], as_index=False).agg(
    test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
    test_nll_sd=('test_nll', 'std'), test_accuracy_sd=('test_accuracy', 'std'),
    original_test_nll=('original_test_nll', 'mean'), original_test_accuracy=('original_test_accuracy', 'mean'),
    mean_test_nll=('mean_test_nll', 'mean'), mean_test_accuracy=('mean_test_accuracy', 'mean'),
    transfer_gain_nll=('transfer_gain_nll', 'mean'), transfer_gain_accuracy=('transfer_gain_accuracy', 'mean'),
    transfer_gain_nll_sd=('transfer_gain_nll', 'std'),
    nll_change=('nll_change', 'mean'), accuracy_change=('accuracy_change', 'mean'),
    nll_change_sd=('nll_change', 'std'), accuracy_change_sd=('accuracy_change', 'std'),
    n_splits=('random_split', 'nunique'), n_test_trials=('n_test_trials', 'sum'))
assert len(transfer_by_split) == n_splits * len(tasks) * (len(tasks) - 1)
assert transfer_by_pair.n_splits.eq(n_splits).all()
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(mapping_records).to_csv(output / 'ridge_selected_alphas.csv', index=False)
participant_assignments.to_csv(output / 'participant_mapping_folds.csv', index=False)
baseline_by_split.to_csv(output / 'original_test_results_by_split.csv', index=False)
transfer_by_split.to_csv(output / 'transfer_results_by_split.csv', index=False)
transfer_by_pair.to_csv(output / 'transfer_results_by_pair.csv', index=False)
print(transfer_by_pair.sort_values('transfer_gain_nll', ascending=False).to_string(index=False))
print(f'Ridge transfer results saved to {output}')
