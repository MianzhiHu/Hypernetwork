from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from HyperNetwork import HyperNN, BaselineNN, preprocess_task, BehavioralDataset, behavioral_collate_fn
from utils.CognitiveModels import DelayedDiscounting, ColumbiaCardTask
from task_config import TASK_CONFIGS

BASE = Path(__file__).resolve().parent
selected_config = 'layers_1_dims_64_rank_full_emb_4_nonlinear_nodes_8_reg_0.0'
split_dir = BASE / 'Results/HyperNN_Grid_Search/random_splits'
cognitive_root = BASE / 'Results/Final_Models/CogModels'
basic_root = BASE / 'Results/Final_Models/BasicNN/layers_1_dims_64'
hyper_root = split_dir.parent / selected_config
output = BASE / 'Results/Balanced_Metrics_Test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    torch.set_num_threads(1)
    records, checks = [], []
    for task in ['cct', 'dd']:
        raw = pd.read_csv(BASE / f'Data/hypernetwork_data/{task}_data.csv')
        data, x_col, y_col = preprocess_task(raw.copy(), TASK_CONFIGS[task.upper()], task_name=task)
        saved = pd.read_csv(split_dir / f'{task}_random_splits.csv')
        keys = ['worker_id', 'participant_id', 'trial']
        aligned = data[keys].merge(saved, on=keys, how='outer', validate='one_to_one', indicator=True)
        assert len(aligned) == len(data) and aligned['_merge'].eq('both').all()
        identities = data[['participant_id', 'worker_id']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
        saved_map = pd.read_csv(split_dir.parent / 'participant_map.csv')
        assert identities.astype({'participant_id': 'int64'}).equals(saved_map[identities.columns].sort_values('participant_id').reset_index(drop=True))
        raw['trial'] = raw.groupby('worker_id').cumcount() + 1
        raw = raw.merge(saved[['worker_id', 'trial', 'fold']], on=['worker_id', 'trial'], how='outer', validate='one_to_one', indicator=True)
        assert raw['_merge'].eq('both').all()
        raw = raw.loc[raw.responded.eq(1)].copy()
        raw['choice'] = (raw.choice.map({'smaller_sooner': 0, 'larger_later': 1}) if task == 'dd'
                         else raw.action.map({'end_round': 0, 'draw_card': 1}))
        for split in range(10):
            selected = aligned[f'random_split_{split}'].eq('test').to_numpy()
            assert np.array_equal(selected, aligned.fold.eq(split))
            loader = DataLoader(BehavioralDataset(data.loc[selected].copy(), x_col, y_col),
                                batch_size=16, shuffle=False, collate_fn=behavioral_collate_fn)
            predictions = []
            for family, root in [('BasicNN', basic_root), ('HyperNN', hyper_root)]:
                checkpoint = torch.load(root / f'random_split_{split}/{task}_model.pt', map_location='cpu', weights_only=True)
                config = checkpoint['extra_config']
                assert config['random_split'] == split
                if family == 'BasicNN':
                    assert config['x_col'] == x_col and config['y_col'] == y_col
                    model = BaselineNN(len(x_col), config['hidden_dim'], config['output_dim'],
                                       config['num_hidden_layers'], config['nonlinearity'])
                else:
                    model = HyperNN(len(identities), len(x_col), config['hidden_dims'], 2,
                        num_hidden_layers=config['num_hidden_layers'], participant_emb_dim=config['participant_emb_dim'],
                        hyper_hidden_dim=config['hyper_hidden_dim'], rank=config['rank'],
                        shared_right=config['shared_right'], mapping=config['mapping'],
                        alpha=config.get('alpha'), nonlinearity=config.get('nonlinearity', 'tanh'))
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                model.to(device).eval()
                ys, ps, labels, losses = [], [], [], []
                with torch.no_grad():
                    for batch in loader:
                        logits, _ = model(batch['x'].to(device), batch['participant_id'].to(device))
                        y = batch['y'].to(device)
                        valid = batch['mask'].to(device).bool() & (y >= 0) & (y < 2)
                        logp = logits[valid].log_softmax(-1)
                        observed = y[valid]
                        ys.extend(observed.cpu().numpy())
                        ps.extend(logp.exp()[:, 1].cpu().numpy())
                        labels.extend(logp.argmax(-1).cpu().numpy())
                        losses.extend((-logp.gather(1, observed[:, None]).squeeze(1)).cpu().numpy())
                predictions.append((family, root.name, ys, ps, labels, losses))
            test = raw.loc[raw.fold.eq(split)]
            for path in sorted((cognitive_root / task).glob(f'*_fit_{split}.csv')):
                name = path.stem.rsplit('_fit_', 1)[0]
                params = pd.read_csv(path).set_index('participant_id').best_parameters
                model = DelayedDiscounting(name) if task == 'dd' else ColumbiaCardTask(name)
                ys, ps, labels, losses = [], [], [], []
                for worker, pdata in test.groupby('worker_id'):
                    values = np.fromstring(params.loc[worker].strip('[]').replace(',', ' '), sep=' ')
                    assert len(values) == len(model._param_map[name]) and np.isfinite(values).all()
                    for attr, idx in model._param_map[name].items():
                        setattr(model, attr, values[idx])
                    if task == 'dd':
                        value = model._function_map[name](pdata.large_amount.to_numpy(), pdata.later_delay.to_numpy())
                        utilities = np.column_stack([pdata.small_amount, value])
                    else:
                        value = model._function_map[name](*[pdata[c].to_numpy() for c in model.input_columns[:-1]])
                        utilities = np.column_stack([np.zeros(len(pdata)), value])
                    probabilities = model.softmax(utilities)
                    y = pdata.choice.to_numpy(dtype=int)
                    # Neural category coding reverses the cognitive choice labels in both tasks.
                    ys.extend(1 - y)
                    ps.extend(probabilities[:, 0])
                    labels.extend(1 - probabilities.argmax(1))
                    losses.extend(-np.log(probabilities[np.arange(len(y)), y]))
                predictions.append(('Cognitive', name, ys, ps, labels, losses))
            # Calculate on all valid trials within each fold, NOT averages of participant AUCs.
            reference_y = sorted(predictions[0][2])
            for family, name, ys, ps, labels, losses in predictions:
                assert sorted(ys) == reference_y and np.isfinite(ps).all() and np.isfinite(losses).all()
                records.append(dict(task=task, model_family=family, model=name, random_split=split,
                    test_nll=np.mean(np.asarray(losses, dtype=float)), test_accuracy=np.mean(np.asarray(ys) == labels),
                    test_balanced_accuracy=balanced_accuracy_score(ys, labels), test_auc=roc_auc_score(ys, ps),
                    n_test_trials=len(ys)))
            print(f'{task} fold {split} complete', flush=True)
    folds = pd.DataFrame(records)
    assert len(folds) == 100
    summary = folds.groupby(['task', 'model_family', 'model'], as_index=False).agg(
        test_nll=('test_nll', 'mean'), test_accuracy=('test_accuracy', 'mean'),
        test_balanced_accuracy=('test_balanced_accuracy', 'mean'), test_auc=('test_auc', 'mean'),
        test_nll_sd=('test_nll', 'std'), n_splits=('random_split', 'nunique'), n_test_trials=('n_test_trials', 'sum'))
    # Check existing fold metrics, allowing saved parameter rounding and flagging changed softmax clipping.
    previous = pd.read_csv(BASE / 'Results/Cognitive_Model_Evaluation/cct_dd_model_by_fold.csv')
    checked = folds.merge(previous, left_on=['task', 'model_family', 'model', 'random_split'],
                          right_on=['task', 'model_family', 'model', 'fold'], validate='one_to_one', suffixes=('', '_saved'))
    assert len(checked) == len(folds) and checked.n_test_trials.eq(checked.n_test_trials_saved).all()
    checked['nll_difference'] = checked.test_nll - checked.test_nll_saved
    checked['accuracy_difference'] = checked.test_accuracy - checked.test_accuracy_saved
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / 'cct_dd_model_by_fold.csv', index=False)
    summary.to_csv(output / 'cct_dd_model_comparison.csv', index=False)
    checked.to_csv(output / 'saved_results_check.csv', index=False)
    print(summary.to_string(index=False))
    print('Maximum absolute differences from saved results:')
    print(checked.groupby('model_family')[['nll_difference', 'accuracy_difference']].agg(lambda x: x.abs().max()))
