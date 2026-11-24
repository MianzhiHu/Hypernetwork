from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.CognitiveModels import *

BASE = Path(__file__).resolve().parent
data_dir = BASE / 'Data/hypernetwork_data'
split_dir = BASE / 'Results/HyperNN_Grid_Search/random_splits'
cog_output = BASE / 'Results/Final_Models/CogModels_Group'
num_iterations = 100
seed = 42

if __name__ == '__main__':
    dd_data = pd.read_csv(data_dir / 'dd_data.csv')
    cct_data = pd.read_csv(data_dir / 'cct_data.csv')
    stop_signal_data = pd.read_csv(data_dir / 'stop_signal_data.csv')
    motor_data = pd.read_csv(data_dir / 'motor_data.csv')
    twobytwo_data = pd.read_csv(data_dir / 'two_by_two_data.csv')

    # DD
    for split in range(10):
        data = dd_data.copy()
        # Trial numbering must precede valid-trial filtering.
        data['trial'] = data.groupby('worker_id').cumcount() + 1
        saved = pd.read_csv(split_dir / 'dd_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold', f'random_split_{split}']
        data = data.merge(saved[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not data['_merge'].eq('both').all():
            raise ValueError('DD: trial identities do not match the saved folds')
        data = data.drop(columns='_merge')
        data = data[data['responded'] == 1].copy()
        data['choice'] = data.choice.map({'smaller_sooner': 0, 'larger_later': 1})
        training_data = data[data[f'random_split_{split}'].isin(['train', 'validation'])].copy()
        test_data = data[data[f'random_split_{split}'] == 'test']
        # Pool all training participants; keep their identities in the test data.
        training_data['worker_id'] = 'group'
        training_dict = dict_generator_cognitive(training_data, task='dd')['group']
        training_dict = {column: np.asarray(values) for column, values in training_dict.items()}
        test_dict = dict_generator_cognitive(test_data, task='dd')

        models = [DelayedDiscounting('dd_exponential'), DelayedDiscounting('dd_hyperbolic'),
                  DelayedDiscounting('dd_hyperboloid')]
        run_dir = cog_output / 'dd'
        run_dir.mkdir(parents=True, exist_ok=True)
        for model_number, model in enumerate(models):
            fit_path = run_dir / f'{model.model_type}_fit_{split}.csv'
            eval_path = run_dir / f'{model.model_type}_eval_{split}.csv'
            if fit_path.is_file() and eval_path.is_file():
                print(f'Skipping {model.model_type}, split {split}')
                continue

            # Fit one shared parameter vector using the unchanged likelihood function.
            rng = np.random.default_rng(seed + 0 + 100 * split + model_number)
            bounds = model_bounds[model.model_type]
            inputs = tuple(training_dict[column] for column in model.input_columns)
            best_nll = np.inf
            best_parameters = None
            for iteration in range(num_iterations):
                initial_guess = [rng.uniform(low, high) for low, high in bounds]
                result = minimize(model.negative_log_likelihood, initial_guess, args=inputs,
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
                if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun < best_nll:
                    best_nll = float(result.fun)
                    best_parameters = result.x.copy()
                    converged = bool(result.success)
                    message = str(result.message)
                print(f'{model.model_type}, split {split}, iteration {iteration + 1}/{num_iterations}', flush=True)
            if best_parameters is None:
                raise RuntimeError(f'{model.model_type}/{split}: all fits failed')

            # Use the same parameters to evaluate every participant.
            params = pd.Series({participant: best_parameters for participant in test_dict})
            evaluated = model.evaluate(params, test_dict)
            k = len(best_parameters)
            n = len(training_dict['choice'])
            fitted = pd.DataFrame([{'participant_id': 'group', 'best_parameters': best_parameters,
                                    'best_nll': best_nll, 'AIC': 2 * k + 2 * best_nll,
                                    'BIC': k * np.log(n) + 2 * best_nll,
                                    'converged': converged, 'message': message}])
            fitted.to_csv(fit_path, index=False)
            evaluated.to_csv(eval_path, index=False)

    # CCT
    for split in range(10):
        data = cct_data.copy()
        # Trial numbering must precede valid-trial filtering.
        data['trial'] = data.groupby('worker_id').cumcount() + 1
        saved = pd.read_csv(split_dir / 'cct_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold', f'random_split_{split}']
        data = data.merge(saved[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not data['_merge'].eq('both').all():
            raise ValueError('CCT: trial identities do not match the saved folds')
        data = data.drop(columns='_merge')
        data = data[data['responded'] == 1].copy()
        data['action'] = data.action.map({'draw_card': 1, 'end_round': 0})
        training_data = data[data[f'random_split_{split}'].isin(['train', 'validation'])].copy()
        test_data = data[data[f'random_split_{split}'] == 'test']
        # Pool all training participants; keep their identities in the test data.
        training_data['worker_id'] = 'group'
        training_dict = dict_generator_cognitive(training_data, task='cct')['group']
        training_dict = {column: np.asarray(values) for column, values in training_dict.items()}
        test_dict = dict_generator_cognitive(test_data, task='cct')

        models = [ColumbiaCardTask('cct_pt'), ColumbiaCardTask('cct_pt_prob'),
                  ColumbiaCardTask('cct_pt_loss_shape')]
        run_dir = cog_output / 'cct'
        run_dir.mkdir(parents=True, exist_ok=True)
        for model_number, model in enumerate(models):
            fit_path = run_dir / f'{model.model_type}_fit_{split}.csv'
            eval_path = run_dir / f'{model.model_type}_eval_{split}.csv'
            if fit_path.is_file() and eval_path.is_file():
                print(f'Skipping {model.model_type}, split {split}')
                continue

            # Fit one shared parameter vector using the unchanged likelihood function.
            rng = np.random.default_rng(seed + 10000 + 100 * split + model_number)
            bounds = model_bounds[model.model_type]
            inputs = tuple(training_dict[column] for column in model.input_columns)
            best_nll = np.inf
            best_parameters = None
            for iteration in range(num_iterations):
                initial_guess = [rng.uniform(low, high) for low, high in bounds]
                result = minimize(model.negative_log_likelihood, initial_guess, args=inputs,
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
                if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun < best_nll:
                    best_nll = float(result.fun)
                    best_parameters = result.x.copy()
                    converged = bool(result.success)
                    message = str(result.message)
                print(f'{model.model_type}, split {split}, iteration {iteration + 1}/{num_iterations}', flush=True)
            if best_parameters is None:
                raise RuntimeError(f'{model.model_type}/{split}: all fits failed')

            # Use the same parameters to evaluate every participant.
            params = pd.Series({participant: best_parameters for participant in test_dict})
            evaluated = model.evaluate(params, test_dict)
            k = len(best_parameters)
            n = len(training_dict['choice'])
            fitted = pd.DataFrame([{'participant_id': 'group', 'best_parameters': best_parameters,
                                    'best_nll': best_nll, 'AIC': 2 * k + 2 * best_nll,
                                    'BIC': k * np.log(n) + 2 * best_nll,
                                    'converged': converged, 'message': message}])
            fitted.to_csv(fit_path, index=False)
            evaluated.to_csv(eval_path, index=False)

    # Stop-Signal
    for split in range(10):
        data = stop_signal_data.copy()
        # Trial numbering must precede valid-trial filtering.
        data['trial'] = data.groupby('worker_id').cumcount() + 1
        saved = pd.read_csv(split_dir / 'stopsignal_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold', f'random_split_{split}']
        data = data.merge(saved[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not data['_merge'].eq('both').all():
            raise ValueError('Stop-Signal: trial identities do not match the saved folds')
        data = data.drop(columns='_merge')
        training_data = data[data[f'random_split_{split}'].isin(['train', 'validation'])].copy()
        test_data = data[data[f'random_split_{split}'] == 'test']
        # Pool all training participants; keep their identities in the test data.
        training_data['worker_id'] = 'group'
        training_dict = dict_generator_cognitive(training_data, task='stop_signal')['group']
        training_dict = {column: np.asarray(values) for column, values in training_dict.items()}
        test_dict = dict_generator_cognitive(test_data, task='stop_signal')

        models = [StopSignal('ss_logistic'), StopSignal('ss_hr_exgau'), StopSignal('ss_rdex')]
        run_dir = cog_output / 'stopsignal'
        run_dir.mkdir(parents=True, exist_ok=True)
        for model_number, model in enumerate(models):
            fit_path = run_dir / f'{model.model_type}_fit_{split}.csv'
            eval_path = run_dir / f'{model.model_type}_eval_{split}.csv'
            if fit_path.is_file() and eval_path.is_file():
                print(f'Skipping {model.model_type}, split {split}')
                continue

            # Fit one shared parameter vector using the unchanged likelihood function.
            rng = np.random.default_rng(seed + 20000 + 100 * split + model_number)
            bounds = model_bounds[model.model_type]
            inputs = tuple(training_dict[column] for column in model.input_columns)
            best_nll = np.inf
            best_parameters = None
            for iteration in range(num_iterations):
                initial_guess = [rng.uniform(low, high) for low, high in bounds]
                result = minimize(model.negative_log_likelihood, initial_guess, args=inputs,
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
                if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun < best_nll:
                    best_nll = float(result.fun)
                    best_parameters = result.x.copy()
                    converged = bool(result.success)
                    message = str(result.message)
                print(f'{model.model_type}, split {split}, iteration {iteration + 1}/{num_iterations}', flush=True)
            if best_parameters is None:
                raise RuntimeError(f'{model.model_type}/{split}: all fits failed')

            # Use the same parameters to evaluate every participant.
            params = pd.Series({participant: best_parameters for participant in test_dict})
            evaluated = model.evaluate(params, test_dict)
            k = len(best_parameters)
            n = len(training_dict['choice'])
            fitted = pd.DataFrame([{'participant_id': 'group', 'best_parameters': best_parameters,
                                    'best_nll': best_nll, 'AIC': 2 * k + 2 * best_nll,
                                    'BIC': k * np.log(n) + 2 * best_nll,
                                    'converged': converged, 'message': message}])
            fitted.to_csv(fit_path, index=False)
            evaluated.to_csv(eval_path, index=False)

    # Motor Stop-Signal
    for split in range(10):
        data = motor_data.copy()
        # Trial numbering must precede valid-trial filtering.
        data['trial'] = data.groupby('worker_id').cumcount() + 1
        saved = pd.read_csv(split_dir / 'motor_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold', f'random_split_{split}']
        data = data.merge(saved[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not data['_merge'].eq('both').all():
            raise ValueError('Motor Stop-Signal: trial identities do not match the saved folds')
        data = data.drop(columns='_merge')
        training_data = data[data[f'random_split_{split}'].isin(['train', 'validation'])].copy()
        test_data = data[data[f'random_split_{split}'] == 'test']
        # Pool all training participants; keep their identities in the test data.
        training_data['worker_id'] = 'group'
        training_dict = dict_generator_cognitive(training_data, task='motor')['group']
        training_dict = {column: np.asarray(values) for column, values in training_dict.items()}
        test_dict = dict_generator_cognitive(test_data, task='motor')

        models = [StopSignal('motor_logistic'), StopSignal('motor_hr_exgau'), StopSignal('motor_rdex')]
        run_dir = cog_output / 'motor'
        run_dir.mkdir(parents=True, exist_ok=True)
        for model_number, model in enumerate(models):
            fit_path = run_dir / f'{model.model_type}_fit_{split}.csv'
            eval_path = run_dir / f'{model.model_type}_eval_{split}.csv'
            if fit_path.is_file() and eval_path.is_file():
                print(f'Skipping {model.model_type}, split {split}')
                continue

            # Fit one shared parameter vector using the unchanged likelihood function.
            rng = np.random.default_rng(seed + 30000 + 100 * split + model_number)
            bounds = model_bounds[model.model_type]
            inputs = tuple(training_dict[column] for column in model.input_columns)
            best_nll = np.inf
            best_parameters = None
            for iteration in range(num_iterations):
                initial_guess = [rng.uniform(low, high) for low, high in bounds]
                result = minimize(model.negative_log_likelihood, initial_guess, args=inputs,
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
                if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun < best_nll:
                    best_nll = float(result.fun)
                    best_parameters = result.x.copy()
                    converged = bool(result.success)
                    message = str(result.message)
                print(f'{model.model_type}, split {split}, iteration {iteration + 1}/{num_iterations}', flush=True)
            if best_parameters is None:
                raise RuntimeError(f'{model.model_type}/{split}: all fits failed')

            # Use the same parameters to evaluate every participant.
            params = pd.Series({participant: best_parameters for participant in test_dict})
            evaluated = model.evaluate(params, test_dict)
            k = len(best_parameters)
            n = len(training_dict['choice'])
            fitted = pd.DataFrame([{'participant_id': 'group', 'best_parameters': best_parameters,
                                    'best_nll': best_nll, 'AIC': 2 * k + 2 * best_nll,
                                    'BIC': k * np.log(n) + 2 * best_nll,
                                    'converged': converged, 'message': message}])
            fitted.to_csv(fit_path, index=False)
            evaluated.to_csv(eval_path, index=False)

    # Two-by-two
    for split in range(10):
        data = twobytwo_data.copy()
        # Trial numbering must precede valid-trial filtering.
        data['trial'] = data.groupby('worker_id').cumcount() + 1
        saved = pd.read_csv(split_dir / 'twobytwo_random_splits.csv')
        columns = ['worker_id', 'trial', 'fold', f'random_split_{split}']
        data = data.merge(saved[columns], on=['worker_id', 'trial'], how='outer',
                          validate='one_to_one', indicator=True)
        if not data['_merge'].eq('both').all():
            raise ValueError('Two-by-two: trial identities do not match the saved folds')
        data = data.drop(columns='_merge')
        training_data = data[data[f'random_split_{split}'].isin(['train', 'validation'])].copy()
        test_data = data[data[f'random_split_{split}'] == 'test']
        # Pool all training participants; keep their identities in the test data.
        training_data['worker_id'] = 'group'
        training_dict = dict_generator_cognitive(training_data, task='twobytwo')['group']
        training_dict = {column: np.asarray(values) for column, values in training_dict.items()}
        test_dict = dict_generator_cognitive(test_data, task='twobytwo')

        models = [TwoByTwo('twobytwo_ddm'), TwoByTwo('twobytwo_lba'), TwoByTwo('twobytwo_ccr')]
        run_dir = cog_output / 'twobytwo'
        run_dir.mkdir(parents=True, exist_ok=True)
        for model_number, model in enumerate(models):
            fit_path = run_dir / f'{model.model_type}_fit_{split}.csv'
            eval_path = run_dir / f'{model.model_type}_eval_{split}.csv'
            if fit_path.is_file() and eval_path.is_file():
                print(f'Skipping {model.model_type}, split {split}')
                continue

            # Fit one shared parameter vector using the unchanged likelihood function.
            rng = np.random.default_rng(seed + 40000 + 100 * split + model_number)
            bounds = model_bounds[model.model_type]
            inputs = tuple(training_dict[column] for column in model.input_columns)
            best_nll = np.inf
            best_parameters = None
            for iteration in range(num_iterations):
                initial_guess = [rng.uniform(low, high) for low, high in bounds]
                result = minimize(model.negative_log_likelihood, initial_guess, args=inputs,
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
                if np.isfinite(result.fun) and np.isfinite(result.x).all() and result.fun < best_nll:
                    best_nll = float(result.fun)
                    best_parameters = result.x.copy()
                    converged = bool(result.success)
                    message = str(result.message)
                print(f'{model.model_type}, split {split}, iteration {iteration + 1}/{num_iterations}', flush=True)
            if best_parameters is None:
                raise RuntimeError(f'{model.model_type}/{split}: all fits failed')

            # Use the same parameters to evaluate every participant.
            params = pd.Series({participant: best_parameters for participant in test_dict})
            evaluated = model.evaluate(params, test_dict)
            k = len(best_parameters)
            n = len(training_dict['choice'])
            fitted = pd.DataFrame([{'participant_id': 'group', 'best_parameters': best_parameters,
                                    'best_nll': best_nll, 'AIC': 2 * k + 2 * best_nll,
                                    'BIC': k * np.log(n) + 2 * best_nll,
                                    'converged': converged, 'message': message}])
            fitted.to_csv(fit_path, index=False)
            evaluated.to_csv(eval_path, index=False)
