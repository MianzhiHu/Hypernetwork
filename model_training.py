from pathlib import Path
import os
import numpy as np
import pandas as pd
from utils.CognitiveModels import *
import random
from task_config import TASK_CONFIGS

# Only allow the following imports when running the neural network training section.
# Otherwise, they may trigger CUDA errors when the cognitive models are being fit in parallel.
# import torch
# from HyperNetwork import BaselineNN, BehavioralDataset, behavioral_collate_fn, preprocess_task, fit_model, test_model
# from torch.utils.data import DataLoader
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BASE = Path(__file__).resolve().parent
data_dir = BASE / 'Data/hypernetwork_data'
split_dir = BASE / 'Results/HyperNN_Grid_Search/random_splits'
cog_output = BASE / 'Results/Final_Models/CogModels'
nn_output = BASE / 'Results/Final_Models/BasicNN'
num_iterations = 10
max_workers = 8
hidden_dim = 64
num_hidden_layers = 1
batch_size = 16
n_epochs = 400
learning_rate = 1e-3
patience = 50
seed = 42

if __name__ == '__main__':
    dd_data = pd.read_csv(data_dir / 'dd_data.csv')
    cct_data = pd.read_csv(data_dir / 'cct_data.csv')
    motor_data = pd.read_csv(data_dir / 'motor_data.csv')
    stop_signal_data = pd.read_csv(data_dir / 'stop_signal_data.csv')

    # ==================================================================================================================
    # Cognitive model training and evaluation
    # ==================================================================================================================
    # DD
    for split in range(10):
        print(f"Training DD model on split {split}")
        dd_data['trial'] = dd_data.groupby('worker_id').cumcount() + 1
        dd_split = pd.read_csv(split_dir / f'dd_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold'] + [f'random_split_{split}']
        dd_data_fold = dd_data.merge(dd_split[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not dd_data_fold['_merge'].eq('both').all():
            raise ValueError(f'DD: trial identities do not match the saved folds')
        dd_data_fold = dd_data_fold.drop(columns='_merge')
        # Keep only valid trials for training and testing
        dd_data_fold = dd_data_fold[dd_data_fold['responded'] == 1]
        dd_data_fold['choice'] = dd_data_fold.choice.map({'smaller_sooner': 0, 'larger_later': 1})
        training_data = dd_data_fold[dd_data_fold['random_split_{}'.format(split)].isin(['train', 'validation'])]
        training_dict = dict_generator_cognitive(training_data, task='dd')
        test_data = dd_data_fold[dd_data_fold['random_split_{}'.format(split)] == 'test']
        test_dict = dict_generator_cognitive(test_data, task='dd')

        # Define the models
        dd_model_exponential = DelayedDiscounting('dd_exponential')
        dd_model_hyperbolic = DelayedDiscounting('dd_hyperbolic')
        dd_model_hyperboloid = DelayedDiscounting('dd_hyperboloid')

        # Fit the models
        dd_exponential_results = dd_model_exponential.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        dd_hyperbolic_results = dd_model_hyperbolic.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        dd_hyperboloid_results = dd_model_hyperboloid.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)

        # Evaluate the models on the test set
        dd_exponential_eval = dd_model_exponential.evaluate(dd_exponential_results.set_index('participant_id')['best_parameters'], test_dict)
        dd_hyperbolic_eval = dd_model_hyperbolic.evaluate(dd_hyperbolic_results.set_index('participant_id')['best_parameters'], test_dict)
        dd_hyperboloid_eval = dd_model_hyperboloid.evaluate(dd_hyperboloid_results.set_index('participant_id')['best_parameters'], test_dict)

        # Save the results
        run_dir = cog_output / 'dd'
        run_dir.mkdir(parents=True, exist_ok=True)

        dd_exponential_results.to_csv(run_dir / f'dd_exponential_fit_{split}.csv', index=False)
        dd_hyperbolic_results.to_csv(run_dir / f'dd_hyperbolic_fit_{split}.csv', index=False)
        dd_hyperboloid_results.to_csv(run_dir / f'dd_hyperboloid_fit_{split}.csv', index=False)
        dd_exponential_eval.to_csv(run_dir / f'dd_exponential_eval_{split}.csv', index=False)
        dd_hyperbolic_eval.to_csv(run_dir / f'dd_hyperbolic_eval_{split}.csv', index=False)
        dd_hyperboloid_eval.to_csv(run_dir / f'dd_hyperboloid_eval_{split}.csv', index=False)
        print(f"Completed training DD models on split {split}")

    # CCT
    for split in range(10):
        print(f"Training CCT model on split {split}")
        cct_data['trial'] = cct_data.groupby('worker_id').cumcount() + 1
        cct_split = pd.read_csv(split_dir / f'cct_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold'] + [f'random_split_{split}']
        cct_data_fold = cct_data.merge(cct_split[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not cct_data_fold['_merge'].eq('both').all():
            raise ValueError(f'CCT: trial identities do not match the saved folds')
        cct_data_fold = cct_data_fold.drop(columns='_merge')
        # Keep only valid trials for training and testing
        cct_data_fold = cct_data_fold[cct_data_fold['responded'] == 1]
        cct_data_fold['action'] = cct_data_fold.action.map({'draw_card': 1, 'end_round': 0})
        training_data = cct_data_fold[cct_data_fold['random_split_{}'.format(split)].isin(['train', 'validation'])]
        training_dict = dict_generator_cognitive(training_data, task='cct')
        test_data = cct_data_fold[cct_data_fold['random_split_{}'.format(split)] == 'test']
        test_dict = dict_generator_cognitive(test_data, task='cct')

        # Define the models
        cct_model_pt = ColumbiaCardTask('cct_pt')
        cct_model_pt_prob = ColumbiaCardTask('cct_pt_prob')
        cct_model_pt_loss_shape = ColumbiaCardTask('cct_pt_loss_shape')

        # Fit the models
        cct_pt_results = cct_model_pt.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        cct_pt_prob_results = cct_model_pt_prob.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        cct_pt_loss_shape_results = cct_model_pt_loss_shape.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)

        # Evaluate the models on the test set
        cct_pt_eval = cct_model_pt.evaluate(cct_pt_results.set_index('participant_id')['best_parameters'], test_dict)
        cct_pt_prob_eval = cct_model_pt_prob.evaluate(cct_pt_prob_results.set_index('participant_id')['best_parameters'], test_dict)
        cct_pt_loss_shape_eval = cct_model_pt_loss_shape.evaluate(cct_pt_loss_shape_results.set_index('participant_id')['best_parameters'], test_dict)

        # Save the results
        run_dir = cog_output / 'cct'
        run_dir.mkdir(parents=True, exist_ok=True)
        cct_pt_results.to_csv(run_dir / f'cct_pt_fit_{split}.csv', index=False)
        cct_pt_prob_results.to_csv(run_dir / f'cct_pt_prob_fit_{split}.csv', index=False)
        cct_pt_loss_shape_results.to_csv(run_dir / f'cct_pt_loss_shape_fit_{split}.csv', index=False)
        cct_pt_eval.to_csv(run_dir / f'cct_pt_eval_{split}.csv', index=False)
        cct_pt_prob_eval.to_csv(run_dir / f'cct_pt_prob_eval_{split}.csv', index=False)
        cct_pt_loss_shape_eval.to_csv(run_dir / f'cct_pt_loss_shape_eval_{split}.csv', index=False)

    # Stop-Signal
    for split in range(10):
        print(f"Training Stop-Signal model on split {split}")
        ss_data = stop_signal_data.copy()
        ss_data['trial'] = ss_data.groupby('worker_id').cumcount() + 1
        ss_split = pd.read_csv(split_dir / f'stopsignal_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold'] + [f'random_split_{split}']
        ss_data_fold = ss_data.merge(ss_split[columns], on=['worker_id', 'trial'], how='outer', validate='one_to_one', indicator=True)
        if not ss_data_fold['_merge'].eq('both').all():
            raise ValueError(f'Stop-Signal: trial identities do not match the saved folds')
        ss_data_fold = ss_data_fold.drop(columns='_merge')
        training_data = ss_data_fold[ss_data_fold['random_split_{}'.format(split)].isin(['train', 'validation'])]
        training_dict = dict_generator_cognitive(training_data, task='stop_signal')
        test_data = ss_data_fold[ss_data_fold['random_split_{}'.format(split)] == 'test']
        test_dict = dict_generator_cognitive(test_data, task='stop_signal')

        # Define the models
        ss_model_logistic = StopSignal('ss_logistic')
        ss_model_hr_exgau = StopSignal('ss_hr_exgau')
        ss_model_rdex = StopSignal('ss_rdex')

        # Fit the models
        ss_logistic_results = ss_model_logistic.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        ss_hr_exgau_results = ss_model_hr_exgau.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        ss_rdex_results = ss_model_rdex.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)

        # Evaluate the models on the test set
        ss_logistic_eval = ss_model_logistic.evaluate(ss_logistic_results.set_index('participant_id')['best_parameters'], test_dict)
        ss_hr_exgau_eval = ss_model_hr_exgau.evaluate(ss_hr_exgau_results.set_index('participant_id')['best_parameters'], test_dict)
        ss_rdex_eval = ss_model_rdex.evaluate(ss_rdex_results.set_index('participant_id')['best_parameters'], test_dict)

        # Save the results
        run_dir = cog_output / 'stopsignal'
        run_dir.mkdir(parents=True, exist_ok=True)
        ss_logistic_results.to_csv(run_dir / f'ss_logistic_fit_{split}.csv', index=False)
        ss_hr_exgau_results.to_csv(run_dir / f'ss_hr_exgau_fit_{split}.csv', index=False)
        ss_rdex_results.to_csv(run_dir / f'ss_rdex_fit_{split}.csv', index=False)
        ss_logistic_eval.to_csv(run_dir / f'ss_logistic_eval_{split}.csv', index=False)
        ss_hr_exgau_eval.to_csv(run_dir / f'ss_hr_exgau_eval_{split}.csv', index=False)
        ss_rdex_eval.to_csv(run_dir / f'ss_rdex_eval_{split}.csv', index=False)

    # Motor Stop-Signal
    for split in range(10):
        print(f"Training Motor Stop-Signal model on split {split}")
        motor_data['trial'] = motor_data.groupby('worker_id').cumcount() + 1
        motor_split = pd.read_csv(split_dir / f'motor_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold'] + [f'random_split_{split}']
        motor_data_fold = motor_data.merge(motor_split[columns], on=['worker_id', 'trial'], how='outer', validate='one_to_one', indicator=True)
        if not motor_data_fold['_merge'].eq('both').all():
            raise ValueError(f'Motor Stop-Signal: trial identities do not match the saved folds')
        motor_data_fold = motor_data_fold.drop(columns='_merge')
        training_data = motor_data_fold[motor_data_fold['random_split_{}'.format(split)].isin(['train', 'validation'])]
        training_dict = dict_generator_cognitive(training_data, task='motor')
        test_data = motor_data_fold[motor_data_fold['random_split_{}'.format(split)] == 'test']
        test_dict = dict_generator_cognitive(test_data, task='motor')

        # Define the models
        motor_model_logistic = StopSignal('motor_logistic')
        motor_model_hr_exgau = StopSignal('motor_hr_exgau')
        motor_model_rdex = StopSignal('motor_rdex')

        # Fit the models
        motor_logistic_results = motor_model_logistic.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        motor_hr_exgau_results = motor_model_hr_exgau.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)
        motor_rdex_results = motor_model_rdex.fit(training_dict, num_iterations=num_iterations, max_workers=max_workers)

        # Evaluate the models on the test set
        motor_logistic_eval = motor_model_logistic.evaluate(motor_logistic_results.set_index('participant_id')['best_parameters'], test_dict)
        motor_hr_exgau_eval = motor_model_hr_exgau.evaluate(motor_hr_exgau_results.set_index('participant_id')['best_parameters'], test_dict)
        motor_rdex_eval = motor_model_rdex.evaluate(motor_rdex_results.set_index('participant_id')['best_parameters'], test_dict)

        # Save the results
        run_dir = cog_output / 'motor'
        run_dir.mkdir(parents=True, exist_ok=True)
        motor_logistic_results.to_csv(run_dir / f'motor_logistic_fit_{split}.csv', index=False)
        motor_hr_exgau_results.to_csv(run_dir / f'motor_hr_exgau_fit_{split}.csv', index=False)
        motor_rdex_results.to_csv(run_dir / f'motor_rdex_fit_{split}.csv', index=False)
        motor_logistic_eval.to_csv(run_dir / f'motor_logistic_eval_{split}.csv', index=False)
        motor_hr_exgau_eval.to_csv(run_dir / f'motor_hr_exgau_eval_{split}.csv', index=False)
        motor_rdex_eval.to_csv(run_dir / f'motor_rdex_eval_{split}.csv', index=False)

    # # ==================================================================================================================
    # # Baseline neural network training
    # # ==================================================================================================================
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # tasks = [
    #     {'name': 'cct', 'data': cct_data, 'config': 'CCT'},
    #     {'name': 'dd', 'data': dd_data, 'config': 'DD'},
    #     {'name': 'motor', 'data': motor_data, 'config': 'Motor'},
    #     {'name': 'stopsignal', 'data': stop_signal_data, 'config': 'StopSignal'},
    # ]
    # for task in tasks:
    #     name = task['name']
    #     data, x_col, y_col = preprocess_task(task['data'], TASK_CONFIGS[task['config']], task_name=name)
    #     saved = pd.read_csv(split_dir / f'{name}_random_splits.csv')
    #     keys = ['worker_id', 'participant_id', 'trial']
    #     columns = keys + ['fold'] + [f'random_split_{i}' for i in range(10)]
    #     data = data.merge(saved[columns], on=keys, how='outer', validate='one_to_one', indicator=True)
    #     if not data['_merge'].eq('both').all():
    #         raise ValueError(f'{name}: trial identities do not match the saved folds')
    #     data = data.drop(columns='_merge')
    #     if not data.fold.isin(range(10)).all():
    #         raise ValueError(f'{name}: invalid saved fold numbers')
    #     for split in range(10):
    #         expected = np.select([data.fold.eq((split + 1) % 10), data.fold.eq(split)],
    #                              ['validation', 'test'], default='train')
    #         if not np.array_equal(data[f'random_split_{split}'], expected):
    #             raise ValueError(f'{name}: inconsistent saved split {split}')
    #     task.update(data=data, x_col=x_col, y_col=y_col)
    #
    # config_id = f'layers_{num_hidden_layers}_dims_{hidden_dim}'
    # run_output = nn_output / config_id
    # # Do not overwrite an earlier run accidentally.
    # if run_output.exists() and any(run_output.iterdir()):
    #     raise FileExistsError(f'{run_output} already contains results; choose a new nn_output to rerun')
    # run_output.mkdir(parents=True, exist_ok=True)
    # test_results = []
    # for split in range(10):
    #     for task_number, task in enumerate(tasks):
    #         name, data = task['name'], task['data']
    #         task_seed = seed + 10000 * split + task_number
    #         random.seed(task_seed)
    #         np.random.seed(task_seed)
    #         torch.manual_seed(task_seed)
    #         if torch.cuda.is_available():
    #             torch.cuda.manual_seed_all(task_seed)
    #         loaders = {}
    #         for subset in ['train', 'validation', 'test']:
    #             selected = data.loc[data[f'random_split_{split}'].eq(subset)].copy()
    #             valid = selected.choice_mask.eq(1) & selected[task['y_col']].isin([0, 1])
    #             if not valid.any():
    #                 raise ValueError(f'{name}/{split}/{subset}: no valid trials')
    #             loaders[subset] = DataLoader(
    #                 BehavioralDataset(selected, task['x_col'], task['y_col']),
    #                 batch_size=batch_size, shuffle=subset == 'train',
    #                 collate_fn=behavioral_collate_fn,
    #                 generator=torch.Generator().manual_seed(task_seed))
    #         model = BaselineNN(input_dim=len(task['x_col']), hidden_dim=hidden_dim,
    #                            output_dim=2, num_hidden_layers=num_hidden_layers, nonlinearity='tanh')
    #         run_dir = run_output / f'random_split_{split}'
    #         run_dir.mkdir(exist_ok=True)
    #         config = dict(task=name, random_split=split, seed=task_seed, hidden_dim=hidden_dim,
    #                       num_hidden_layers=num_hidden_layers, input_dim=len(task['x_col']), output_dim=2,
    #                       nonlinearity='tanh', x_col=task['x_col'], y_col=task['y_col'],
    #                       learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs,
    #                       split_file=str(split_dir / f'{name}_random_splits.csv'))
    #         model, history, best_val_loss = fit_model(
    #             model, loaders['train'], loaders['validation'], n_epochs=n_epochs, device=device,
    #             save_path=str(run_dir / f'{name}_model.pt'), model_type='baseline',
    #             lr=learning_rate, patience=patience, emb_reg=0.0, hyper_reg=0.0,
    #             extra_config=config, verbose=False)
    #         pd.DataFrame(history).to_csv(run_dir / f'{name}_history.csv', index=False)
    #         metrics = test_model(model, loaders['test'], device)
    #         record = dict(task=name, config_id=config_id, random_split=split, best_val_loss=best_val_loss,
    #                       **{f'test_{key}': value for key, value in metrics.items()})
    #         record['test_nll'] = record.pop('test_raw_loss')
    #         record['test_accuracy'] = record.pop('test_acc')
    #         test_results.append(record)
    #         pd.DataFrame([record]).to_csv(run_dir / f'{name}_test.csv', index=False)
    #         pd.DataFrame(test_results).to_csv(run_output / 'test_by_split.csv', index=False)
    #         print(f'{name}, split {split}: test NLL={metrics["raw_loss"]:.4f}, accuracy={metrics["acc"]:.4f}')
    #
    # # Equal weight for each of the ten held-out folds.
    # by_task = pd.DataFrame(test_results).groupby(['task', 'config_id'], as_index=False)[
    #     ['test_nll', 'test_accuracy']].mean()
    # by_task.to_csv(run_output / 'test_by_task.csv', index=False)
    # print(by_task)
