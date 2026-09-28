import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import CCA
from HyperNetwork import HyperNN, BehavioralDataset, behavioral_collate_fn, train_one_epoch, evaluate

# ======================================================================================================================
# Settings
# ======================================================================================================================
BASE = Path(__file__).resolve().parent
config_id = 'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8'
root = BASE / 'Results/HyperNN_Grid_Search_Evaluation'
output = BASE / 'Final_Models' / config_id / 'full_data'
batch_size = 16
learning_rate = 1e-3
seed = 42
model_config = dict(hidden_dim=8, num_hidden_layers=1, participant_emb_dim=4,
                    rank=None, mapping='nonlinear', hyper_hidden_dim=8, output_dim=2)

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Reuse exactly the grid-search preprocessing, whose input paths are relative.
    previous_cwd = Path.cwd()
    try:
        os.chdir(BASE)
        import hypernn_grid_search as training
    finally:
        os.chdir(previous_cwd)
    output.mkdir(parents=True, exist_ok=True)

    # Use the validation-selected epoch, not the later early-stopping epoch.
    results = pd.read_csv(root / 'test_results_by_split.csv')
    results = results.loc[results.config_id.eq(config_id)]
    records = []
    component_records = []

    # ==================================================================================================================
    # Fit all trials and save the final epoch directly. No validation or testing loader.
    # ==================================================================================================================
    for task_number, task in enumerate(training.tasks):
        name = task['name']
        data = task['data']
        epochs = results.loc[results.task.eq(name), ['random_split', 'best_epoch']]
        if len(epochs) != 10 or set(epochs.random_split) != set(range(10)):
            raise ValueError(f'Expected ten unique saved splits for {name}')
        if epochs.best_epoch.isna().any() or (epochs.best_epoch <= 0).any():
            raise ValueError(f'Invalid saved epochs for {name}')
        median_epoch = float(epochs.best_epoch.median())
        n_epochs = int(np.floor(median_epoch + 0.5))  # Round half epochs up.

        participants = data[['participant_id', 'worker_id']].drop_duplicates().sort_values('participant_id')
        if (participants.participant_id.duplicated().any() or participants.worker_id.duplicated().any()
                or not np.array_equal(participants.participant_id, np.arange(len(participants)))):
            raise ValueError(f'Invalid participant mapping for {name}')
        consensus = pd.read_csv(root / 'Cross_Task_Convergence/Consensus_Embeddings' / f'{config_id}_{name}.csv')
        # Match identities explicitly, never assume CSV row order equals embedding row order.
        aligned = participants.merge(consensus, on=['participant_id', 'worker_id'], how='left', validate='one_to_one')
        score_columns = [c for c in consensus.columns if c.startswith('gcca_')]
        if len(consensus) != len(participants) or not score_columns or aligned[score_columns].isna().any().any():
            raise ValueError(f'Missing or mismatched GCCA participants for {name}')

        task_seed = seed + task_number
        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        dataset = BehavioralDataset(data.copy(), task['x_col'], task['y_col'])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(task_seed), collate_fn=behavioral_collate_fn)
        model = HyperNN(n_participants=len(participants), input_dim=len(task['x_col']), **model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        print(f'{name}: full-data training for {n_epochs} epochs on {device}', flush=True)
        for epoch in range(1, n_epochs + 1):
            stats = train_one_epoch(model, loader, optimizer, device, model_type='hyper',
                                    emb_reg=0.0, hyper_reg=0.0, max_grad_norm=1.0)
            history.append({'epoch': epoch, 'batch_mean_train_loss': stats['loss'],
                            'batch_mean_train_accuracy': stats['acc']})
            if epoch == 1 or epoch % 50 == 0 or epoch == n_epochs:
                print(f"  epoch {epoch}: batch mean loss {stats['loss']:.5f}", flush=True)

        torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'final_epoch': n_epochs, 'median_cv_best_epoch': median_epoch, 'seed': task_seed,
                    'config_id': config_id, 'task': name, 'training_data': 'full',
                    'participant_map': participants.to_dict('list'),
                    'extra_config': dict(model_config, input_dim=len(task['x_col']), n_participants=len(participants),
                                         learning_rate=learning_rate, batch_size=batch_size, emb_reg=0.0, hyper_reg=0.0)},
                   output / f'{name}_model.pt')
        embedding = model.participant_embedding.weight.detach().cpu().numpy()
        np.save(output / f'{name}_embeddings.npy', embedding)
        participants.to_csv(output / f'{name}_participant_map.csv', index=False)
        pd.DataFrame(history).to_csv(output / f'{name}_history.csv', index=False)
        # Descriptive training performance only; these values never select an epoch.
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)
        performance = evaluate(model, full_loader, device)

        # ==============================================================================================================
        # CCA: full-data embeddings versus the saved task-wise ten-fold GCCA consensus.
        # In-sample agreement, not independent predictive validation. Components are canonical axes, not original axes.
        # ==============================================================================================================
        views = [StandardScaler().fit_transform(embedding[aligned.participant_id.to_numpy(dtype=int)]),
                 StandardScaler().fit_transform(aligned[score_columns].to_numpy())]
        dimensions = min(np.linalg.matrix_rank(v) for v in views)
        if dimensions < 1:
            raise ValueError(f'Degenerate embeddings for {name}')
        cca = CCA(latent_dimensions=int(dimensions)).fit(views)
        correlations = cca.score(views)
        scores = cca.transform(views)
        for component, r in enumerate(correlations, start=1):
            component_records.append({'task': name, 'config_id': config_id, 'component': component, 'cca_r': float(r)})
        np.savez(output / f'{name}_cca_scores.npz', full_data=scores[0], cv_consensus=scores[1],
                 participant_id=aligned.participant_id.to_numpy(), worker_id=aligned.worker_id.to_numpy(dtype=str))
        records.append({'task': name, 'config_id': config_id, 'median_cv_best_epoch': median_epoch,
                        'final_epoch': n_epochs, 'n_participants': len(participants),
                        'n_valid_trials': performance['n_trials'], 'full_train_nll': performance['raw_loss'],
                        'full_train_accuracy': performance['acc'], 'cca_first_r': float(correlations[0]),
                        'cca_mean_r': float(np.mean(correlations))})
        pd.DataFrame(records).to_csv(output / 'full_data_summary.csv', index=False)
        pd.DataFrame(component_records).to_csv(output / 'cca_by_component.csv', index=False)
        print(f"  First CCA r = {correlations[0]:.4f}", flush=True)
