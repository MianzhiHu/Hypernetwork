import os
from pathlib import Path
import hypernn_grid_search as training
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import CCA, GCCA
from cca_zoo.metrics import pairwise_correlations, average_pairwise_correlations
from cca_zoo.model_selection import permutation_test_significance
from HyperNetwork import HyperNN, BehavioralDataset, behavioral_collate_fn, train_one_epoch, evaluate
from task_config import selected_config

# ======================================================================================================================
# Settings
# ======================================================================================================================
BASE = Path(__file__).resolve().parent
root = BASE / 'Results/HyperNN_Grid_Search_Evaluation'
raw = BASE / 'Results/HyperNN_Grid_Search'
output = BASE / 'Results' / 'Final_Models' / selected_config
os.makedirs(output, exist_ok=True)
batch_size = 16
learning_rate = 1e-3
seed = 42
n_permutation = 10000
n_jobs = 32
model_config = dict(hidden_dim=8, num_hidden_layers=1, participant_emb_dim=4, rank=None, mapping='nonlinear', hyper_hidden_dim=16, output_dim=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Use the validation-selected epoch, not the later early-stopping epoch.
    results = pd.read_csv(root / 'test_results_all.csv')
    results = results.loc[results.config_id.eq(selected_config)]

    # ==================================================================================================================
    # Fit all trials within the CV-derived budget; retain the lowest full-data training NLL.
    # ==================================================================================================================
    for task_number, task in enumerate(training.tasks):
        name = task['name']
        data = task['data']
        epochs = results.loc[results.task.eq(name), ['random_split', 'best_epoch']]

        # Fix the full-data training budget at the rounded-up median CV-best epoch.
        median_epoch = float(epochs.best_epoch.median())
        n_epochs = int(np.ceil(median_epoch))
        participants = data[['participant_id', 'worker_id']].drop_duplicates().sort_values('participant_id')

        task_seed = seed + task_number
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        torch.manual_seed(task_seed)
        dataset = BehavioralDataset(data.copy(), task['x_col'], task['y_col'])

        # Load all data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(task_seed), collate_fn=behavioral_collate_fn)
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=behavioral_collate_fn)

        # Define the model and optimizer
        model = HyperNN(n_participants=len(participants), input_dim=len(task['x_col']), **model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        history = []
        best_train_nll = float('inf')
        best_epoch = 0
        best_state = None
        print(f'{name}: full-data training for {n_epochs} epochs on {device}', flush=True)
        for epoch in range(1, n_epochs + 1):
            stats = train_one_epoch(model, loader, optimizer, device, model_type='hyper',
                                    emb_reg=0.0, hyper_reg=0.0)
            train_metrics = evaluate(model, full_loader, device)
            train_nll = train_metrics['raw_loss']
            if not np.isfinite(train_nll):
                raise ValueError(f'Non-finite training NLL for {name} at epoch {epoch}')
            if train_nll < best_train_nll:
                best_train_nll = train_nll
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({'epoch': epoch, 'batch_mean_train_loss': stats['loss'],
                            'batch_mean_train_accuracy': stats['acc'],
                            'full_train_nll': train_nll, 'full_train_accuracy': train_metrics['acc']})
            if epoch == 1 or epoch % 50 == 0 or epoch == n_epochs:
                print(f"  epoch {epoch}: full training NLL {train_nll:.5f}", flush=True)

        # Restore the selected checkpoint before saving or extracting embeddings.
        model.load_state_dict(best_state)
        print(f'{name}: selected epoch {best_epoch}/{n_epochs}, training NLL {best_train_nll:.5f}', flush=True)
        torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'final_epoch': best_epoch, 'max_epochs': n_epochs,
                    'best_train_nll': best_train_nll, 'checkpoint_selection': 'lowest_full_train_nll',
                    'median_cv_best_epoch': median_epoch,
                    'epoch_rule': 'ceil_median_cv_best_epoch', 'seed': task_seed,
                    'selected_config': selected_config, 'task': name, 'training_data': 'full',
                    'participant_map': participants.to_dict('list'),
                    'extra_config': dict(model_config, input_dim=len(task['x_col']), n_participants=len(participants),
                                         learning_rate=learning_rate, batch_size=batch_size, emb_reg=0.0, hyper_reg=0.0)},
                   output / f'{name}_model.pt')
        embedding = model.participant_embedding.weight.detach().cpu().numpy()
        np.save(output / f'{name}_embeddings.npy', embedding)
        participants.to_csv(output / f'{name}_participant_map.csv', index=False)
        pd.DataFrame(history).to_csv(output / f'{name}_history.csv', index=False)

        # ==============================================================================================================
        # CCA: full-data embeddings versus the saved task-wise ten-fold GCCA consensus.
        # ==============================================================================================================
        # Match identities explicitly, never assume CSV row order equals embedding row order.
        consensus = pd.read_csv(root / 'Cross_Task_Convergence/Consensus_Embeddings' / f'{selected_config}_{name}.csv')
        aligned = participants.merge(consensus, on=['participant_id', 'worker_id'], how='left', validate='one_to_one')
        score_columns = [c for c in consensus.columns if c.startswith('gcca_')]

        # Combine the full-data embedding and the ten-fold GCCA consensus into two views for CCA.
        views = [StandardScaler().fit_transform(embedding[aligned.participant_id.to_numpy(dtype=int)]),
                 StandardScaler().fit_transform(aligned[score_columns].to_numpy())]
        dimensions = min(np.linalg.matrix_rank(v) for v in views)
        if dimensions != 4:
            raise ValueError(f'Degenerate GCCA inputs for {name}')
        cca = CCA(n_components=int(dimensions)).fit(views)
        scores = cca.transform(views)
        cca_results = permutation_test_significance(cca, views, n_permutations=n_permutation, random_state=seed, n_jobs=n_jobs)
        correlations = cca_results.correlations
        p_values = cca_results.p_values

        # record the CCA results for each component, and save the full-data and CV-consensus scores to a compressed NPZ file.
        component_records = []
        for component, r in enumerate(correlations, start=1):
            component_records.append({
                'task': name, 'selected_config': selected_config, 'component': component, 'cca_r': correlations[component - 1], 'p': p_values[component - 1]})
        pd.DataFrame(component_records).to_csv(output / f'{name}_consensus_cca_by_component.csv', index=False)
        print(f'Full versus consensus for {name}: r = {correlations[0]:.4f}, p = {p_values[0]:.4f}', flush=True)

        # ==============================================================================================================
        # Pairwise CCA across the ten CV splits and the full-data embedding
        # ==============================================================================================================
        # Load the raw embeddings
        raw_embedding = [np.load(raw / selected_config / f'random_split_{split}' / f'{name}_embeddings.npy', allow_pickle=False)
                         for split in range(10)]

        # Pairwise CCA across the ten CV splits and the full-data embedding
        cca_pair_records = []
        for split, cv_score in enumerate(raw_embedding):
            views = [StandardScaler().fit_transform(embedding[aligned.participant_id.to_numpy(dtype=int)]),
                     StandardScaler().fit_transform(cv_score)]
            dimensions = min(np.linalg.matrix_rank(v) for v in views)
            if dimensions != 4:
                raise ValueError(f'Degenerate CV split {split} for {name}')
            cca = CCA(n_components=int(dimensions)).fit(views)
            scores = cca.transform(views)
            cca_results = permutation_test_significance(cca, views, n_permutations=n_permutation, random_state=seed, n_jobs=n_jobs)
            correlations = cca_results.correlations
            p_values = cca_results.p_values
            cca_pair_records.append({
                'task': name, 'selected_config': selected_config, 'random_split': split, 'cca_r': correlations[0], 'p': p_values[0]})

        cca_pair_df = pd.DataFrame(cca_pair_records)
        cca_pair_df.to_csv(output / f'{name}_pairwise_cca_by_split.csv', index=False)
        print(f'Mean pairwise CCA for {name}: r = {cca_pair_df.cca_r.mean():.4f}', flush=True)
