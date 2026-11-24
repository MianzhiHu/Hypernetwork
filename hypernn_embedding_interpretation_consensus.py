from pathlib import Path
from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as student_t
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.multivariate_ols import MultivariateLS
from statsmodels.regression.linear_model import OLS
from task_config import selected_config, TASK_DIMENSIONS


use_task_dimensions = True  # True: use TASK_DIMENSIONS; False: use all four dimensions.
task_dimensions = {task.lower(): n for task, n in TASK_DIMENSIONS.items()}


selected_models = {
    'cct': 'cct_pt',
    'dd': 'dd_hyperboloid',
    'stopsignal': 'ss_logistic',
    'motor': 'motor_logistic',
    'twobytwo': 'twobytwo_ccr',
}

# Parameter names must follow the order in the saved fit vectors.
model_parameters = {
    'cct_pt': ['Choice Consistency', 'Value Sensitivity', 'Loss Aversion'],
    'cct_pt_prob': ['Choice Consistency', 'Value Curvature', 'Loss Aversion' , 'eta', 'delta'],
    'cct_pt_loss_shape': ['Inverse Temperature', 'Value Curvature', 'Loss Aversion', 'Loss Value Curvature'],
    'dd_exponential': ['t', 'k'],
    'dd_hyperbolic': ['t', 'k'],
    'dd_hyperboloid': ['Choice Consistency', 'Delay Discounting Rate', 'Delayed Reward Sensitivity'],
    'ss_logistic': ['50% Inhibition Delay', 'Inhibition Curve Scale', 'Go Accuracy', 'Go Failure'],
    'ss_hr_exgau': ['mu_correct', 'sigma_correct', 'tau_correct', 'mu_error', 'sigma_error',
                   'tau_error', 'mu_stop', 'sigma_stop', 'tau_stop', 'p_trigger_failure'],
    'ss_rdex': ['Drift Rate - Correct Response', 'Drift Rate - Incorrect Response', 'Response Threshold',
                'Non-Decision Time', 'Stop-Time Gaussian Mean', 'Stop-Time Gaussian SD', 'Stop-time Exponential Tail',
                'Go Failure Probability', 'Trigger Failure Probability'],
    'motor_logistic': ['50% Inhibition Delay', 'Inhibition Curve Scale', 'Go Accuracy',
                       'Go Failure - Critical', 'Go Failure - Non-Critical'],
    # Only one motor_hr_exgau fit is currently saved.
    'motor_hr_exgau': ['mu_correct', 'sigma_correct', 'tau_correct', 'mu_error', 'sigma_error',
                      'tau_error', 'mu_stop', 'sigma_stop', 'tau_stop', 'p_trigger_failure', 'critical_scale'],
    'motor_rdex': ['Drift Rate - Correct Response', 'Drift Rate - Incorrect Response', 'Response Threshold',
                'Non-Decision Time', 'Stop-Time Gaussian Mean', 'Stop-Time Gaussian SD', 'Stop-time Exponential Tail',
                'Go Failure Probability', 'Trigger Failure Probability', 'Critical-Trial Response Threshold Scale'],
    'twobytwo_ddm': ['drift', 'color_effect', 'cue_switch_effect', 'task_switch_effect', 'cti_effect',
                     'incongruent_effect', 'boundary', 'nondecision'],
    'twobytwo_lba': ['Drift Rate', 'Color vs. Magnitude Drift Rate Difference', 'Cue Switch vs. Repeat Drift Rate Difference',
                     'Task Switch vs. Repeat Drift Rate Difference', 'Long vs. Short Cue-Target Interval Drift Rate Difference',
                     'Incongruent vs. Congruent Drift Rate Difference', 'Incorrect-response drift', 'Starting-point variability',
                     'Threshold Gap', 'Non-Decision Time'],
    'twobytwo_ccr': ['Unassociated Evidence', 'Cue Association Gain', 'Target Association Gain', 'Cue Carryover',
                     'Color Cue Encoding Rate', 'Magnitude Cue Encoding Rate', 'Non-Decision Time'],
}


if __name__ == '__main__':
    results_dir = Path(__file__).resolve().parent / 'Results'
    models_dir = results_dir / 'Final_Models'
    embedding_dir = results_dir / 'HyperNN_Grid_Search_Evaluation/Cross_Task_Convergence/Consensus_Embeddings'
    aligned_dir = results_dir / 'HyperNN_Grid_Search_Evaluation/Embedding_Stability/GCCA_Aligned_Embeddings'
    output_dir = results_dir / 'Embedding_Interpretation_Consensus' / selected_config
    pls_data_path = results_dir.parent / 'Data' / 'PLS_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    pls_data_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.family': 'sans-serif', 'font.size': 7,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    regression_tables = []
    model_test_tables = []
    mv_test_tables = []
    fold_regression_tables = []
    fold_model_test_tables = []
    fold_mv_test_tables = []
    fold_correlation_tables = []
    test_columns = {'Effect': 'effect', 'Statistic': 'statistic', 'Value': 'value',
                    'Num DF': 'num_df', 'Den DF': 'den_df', 'F Value': 'f', 'Pr > F': 'p'}

    for task, model in selected_models.items():
        n_dimensions = task_dimensions[task] if use_task_dimensions else 4
        names = model_parameters[model]
        participants = pd.read_csv(embedding_dir / f'{selected_config}_{task}.csv')
        fit_paths = sorted((models_dir / 'CogModels' / task).glob(f'{model}_fit_*.csv'))
        fits = pd.concat(pd.read_csv(path, usecols=['participant_id', 'best_parameters'])
                         .assign(random_split=int(path.stem.rsplit('_', 1)[1])) for path in fit_paths)
        fits[names] = np.stack([np.fromstring(value.strip('[]'), sep=' ') for value in fits['best_parameters']])
        parameters = fits.groupby('participant_id')[names].mean().rename_axis('worker_id')
        matched = participants.merge(parameters, on='worker_id')

        embedding_columns = sorted(matched.filter(regex=r'^gcca_\d+$').columns,
                                   key=lambda name: int(name.rsplit('_', 1)[1]))[:n_dimensions]
        embedding_scores = matched[embedding_columns]
        cog_parameters = matched[names]

        # # Save for behavioral PLS.
        # embedding_scores.to_csv(pls_data_path / f'{task}_behav_gcca_scores.csv', index=False)
        # cog_parameters.to_csv(pls_data_path / f'{task}_cognitive_parameters.csv', index=False)

        # Prepare for multivariate regression of cognitive parameters onto GCCA-aligned embeddings.
        outcome_names = embedding_scores.columns.tolist()
        regression_data = matched.copy()
        regression_data[outcome_names] = StandardScaler().fit_transform(embedding_scores)
        regression_data[names] = StandardScaler().fit_transform(matched[names])
        n = len(matched)
        formula = (' + '.join(f'Q({name!r})' for name in outcome_names)
                   + ' ~ ' + ' + '.join(f'Q({name!r})' for name in names))
        term_names = {f'Q({name!r})': name for name in names}
        overall_hypothesis = [('All cognitive parameters', np.eye(len(names) + 1)[1:], None)]

        # One multivariate fit models all GCCA dimensions jointly.
        res = MultivariateLS.from_formula(formula, data=regression_data).fit()
        print(res.summary())
        (output_dir / f'{task}_{model}_multivariate_summary.txt').write_text(
            res.summary().as_text(), encoding='utf-8')

        # Overall model: all cognitive slopes are zero across all outcomes.
        overall_test = res.mv_test(overall_hypothesis)
        model_test_tables.append(overall_test.summary_frame.reset_index().rename(columns=test_columns)
                                 .assign(task=task, model=model, n=n))
        print(overall_test.summary_frame)

        # Per-parameter joint effects; skip the intercept of standardized data.
        mvt = res.mv_test(skip_intercept_test=True)
        mv_table = mvt.summary_frame.reset_index().rename(columns=test_columns)
        mv_table = mv_table.rename(columns={'effect': 'parameter'})
        mv_table['parameter'] = mv_table['parameter'].replace(term_names)
        mv_test_tables.append(mv_table.assign(task=task, model=model, n=n))
        print(mvt.summary_frame)

        # Dimension-specific coefficients and t-tests come from the joint fit.
        for j, outcome in enumerate(outcome_names):
            regression_tables.append(pd.DataFrame({
                'task': task, 'model': model, 'n': n, 'outcome': outcome, 'parameter': names,
                'beta': np.asarray(res.params)[1:, j], 'se': np.asarray(res.bse)[1:, j],
                't': np.asarray(res.tvalues)[1:, j], 'p': np.asarray(res.pvalues)[1:, j],
            }))
        print(f'{task} | {model} | {len(fit_paths)} fits | n={n} | outcomes={outcome_names}')

        # Match each cognitive fit to its own split's GCCA-aligned embeddings.
        with np.load(aligned_dir / f'{selected_config}_{task}.npz', allow_pickle=False) as aligned:
            for split, split_scores in zip(aligned['random_splits'], aligned['scores']):
                fold_parameters = fits.loc[fits['random_split'].eq(split), ['participant_id', *names]]
                fold_parameters = fold_parameters.rename(columns={'participant_id': 'worker_id'})
                fold_embeddings = pd.DataFrame(split_scores[:, :n_dimensions], columns=outcome_names)
                fold_embeddings.insert(0, 'worker_id', aligned['worker_id'])
                fold_matched = fold_embeddings.merge(fold_parameters, on='worker_id')
                fold_data = fold_matched.copy()
                fold_data[outcome_names] = StandardScaler().fit_transform(fold_matched[outcome_names])
                fold_data[names] = StandardScaler().fit_transform(fold_matched[names])
                fold_n = len(fold_matched)

                fold_fit = MultivariateLS.from_formula(formula, data=fold_data).fit()
                (output_dir / f'{task}_{model}_split_{int(split)}_multivariate_summary.txt').write_text(
                    fold_fit.summary().as_text(), encoding='utf-8')
                fold_overall = fold_fit.mv_test(overall_hypothesis)
                fold_model_test_tables.append(
                    fold_overall.summary_frame.reset_index().rename(columns=test_columns)
                    .assign(task=task, model=model, random_split=int(split), n=fold_n))
                fold_mvt = fold_fit.mv_test(skip_intercept_test=True)
                fold_mv_table = fold_mvt.summary_frame.reset_index().rename(columns=test_columns)
                fold_mv_table = fold_mv_table.rename(columns={'effect': 'parameter'})
                fold_mv_table['parameter'] = fold_mv_table['parameter'].replace(term_names)
                fold_mv_test_tables.append(
                    fold_mv_table.assign(task=task, model=model, random_split=int(split), n=fold_n))

                for j, outcome in enumerate(outcome_names):
                    correlation = pearsonr(fold_data[names].to_numpy(),
                                          fold_data[outcome].to_numpy()[:, None], axis=0)
                    fold_correlation_tables.append(pd.DataFrame({
                        'task': task, 'model': model, 'random_split': int(split), 'n': fold_n,
                        'outcome': outcome, 'parameter': names,
                        'r': correlation.statistic, 'p': correlation.pvalue,
                    }))
                    fold_regression_tables.append(pd.DataFrame({
                        'task': task, 'model': model, 'random_split': int(split), 'n': fold_n,
                        'outcome': outcome, 'parameter': names,
                        'beta': np.asarray(fold_fit.params)[1:, j], 'se': np.asarray(fold_fit.bse)[1:, j],
                        't': np.asarray(fold_fit.tvalues)[1:, j], 'p': np.asarray(fold_fit.pvalues)[1:, j],
                    }))

    results = pd.concat(regression_tables, ignore_index=True)
    results['p_fdr'] = multipletests(results['p'], method='fdr_bh')[1]
    results.to_csv(output_dir / 'regression_pvalues.csv', index=False)

    fold_correlations = pd.concat(fold_correlation_tables, ignore_index=True)
    fold_correlations['p_fdr'] = multipletests(fold_correlations['p'], method='fdr_bh')[1]
    fold_correlations.to_csv(output_dir / 'foldwise_correlations.csv', index=False)
    fold_regressions = pd.concat(fold_regression_tables, ignore_index=True)
    fold_regressions['p_fdr'] = multipletests(fold_regressions['p'], method='fdr_bh')[1]
    fold_regressions['significance'] = fold_regressions['p_fdr'] < 0.05
    # print those that are significant after FDR correction across all 10 split
    significant_folds = fold_regressions.loc[fold_regressions['significance']]
    all_sig = significant_folds.groupby(['task', 'model', 'outcome', 'parameter']).size().reset_index(name='significant_splits')
    all_sig = all_sig.loc[all_sig['significant_splits'] == 10]
    print(f"Significant parameters after FDR correction across all 10 splits:\n{all_sig}")
    fold_regressions.to_csv(output_dir / 'foldwise_regression_pvalues.csv', index=False)

    overall_results = pd.concat(model_test_tables, ignore_index=True)
    multivariate_results = pd.concat(mv_test_tables, ignore_index=True)
    fold_overall_results = pd.concat(fold_model_test_tables, ignore_index=True)
    fold_multivariate_results = pd.concat(fold_mv_test_tables, ignore_index=True)

    # Keep each of the four multivariate statistics as its own testing family.
    # Split-wise fits are repeated estimates, not independent participant samples.
    for table, filename in [
        (overall_results, 'overall_multivariate_tests.csv'),
        (multivariate_results, 'parameter_multivariate_tests.csv'),
        (fold_overall_results, 'foldwise_overall_multivariate_tests.csv'),
        (fold_multivariate_results, 'foldwise_parameter_multivariate_tests.csv'),
    ]:
        table['p'] = table['p'].astype(float)
        table['p_fdr'] = np.nan
        for _, rows in table.groupby('statistic'):
            table.loc[rows.index, 'p_fdr'] = multipletests(rows['p'], method='fdr_bh')[1]
        table['significance'] = table['p_fdr'] < 0.05
        table.to_csv(output_dir / filename, index=False)

    # Save one coefficient grid per task for independent figure assembly.
    coefficient_index = ['task', 'model', 'n', 'parameter']
    gcca_coefficients = results.pivot(index=coefficient_index, columns='outcome', values='beta')
    gcca_pvalues = results.pivot(index=coefficient_index, columns='outcome', values='p_fdr').sort_index()
    limit = 1.1
    cmap = LinearSegmentedColormap.from_list('signed_coefficients', ['#547FA6', '#FAFAF7', '#C27F6A'])
    cmap.set_bad('#D9D9D9')
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 6.5, 'axes.labelsize': 6.5,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})

    for task, model in selected_models.items():
        task_coefficients = gcca_coefficients.xs((task, model), level=['task', 'model'])
        n = int(task_coefficients.index.get_level_values('n')[0])
        task_coefficients = task_coefficients.droplevel('n')
        dimensions = sorted(task_coefficients.columns[task_coefficients.notna().any()],
                            key=lambda name: int(name.rsplit('_', 1)[1]))
        coefficients = task_coefficients.reindex(model_parameters[model])[dimensions]
        cell_pvalues = gcca_pvalues.loc[(task, model, n)].reindex(index=coefficients.index, columns=dimensions)

        # Common physical cell sizes keep the independent grids consistent when assembled.
        grid_width = 0.42 * len(dimensions)
        grid_height = 0.28 * len(coefficients)
        colorbar_space = 0.45 if task == 'twobytwo' else 0
        figure_width = 2.0 + grid_width + 0.04
        figure_height = grid_height + 0.36 + colorbar_space
        fig = plt.figure(figsize=(figure_width, figure_height))
        ax = fig.add_axes([2.0 / figure_width, (0.30 + colorbar_space) / figure_height,
                           grid_width / figure_width, grid_height / figure_height])

        color_values = np.ma.masked_where(
            ~cell_pvalues.lt(0.05).to_numpy(), np.clip(coefficients.to_numpy(), -limit, limit))
        image = ax.imshow(color_values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_ylim(len(coefficients) - 0.5, -0.5)
        ax.set_xticks(np.arange(len(dimensions)), [name.rsplit('_', 1)[1] for name in dimensions])
        ax.set_yticks(np.arange(len(coefficients)), [fill(name, width=30) for name in coefficients.index])
        ax.tick_params(axis='x', length=2, pad=2)
        ax.tick_params(axis='y', length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, values in enumerate(coefficients.to_numpy()):
            for j, value in enumerate(values):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.5, color='black')

        if task == 'twobytwo':
            color_ax = fig.add_axes([0.25, 0.23 / figure_height, 0.80, 0.10 / figure_height])
            colorbar = fig.colorbar(image, cax=color_ax, orientation='horizontal', ticks=[-limit, 0, limit])
            colorbar.set_label('Standardized regression coefficient', fontsize=6.5, labelpad=2)
            colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
            colorbar.outline.set_visible(False)
            color_explanations = [TextArea(text, textprops={'fontsize': 7, 'color': color})
                                  for text, color in [('Red: positive β; ', cmap(1.0)),
                                                      ('Blue: negative β; ', cmap(0.0)),
                                                      ('Gray: insignificant after FDR.', 'gray')]]
            fig.add_artist(AnnotationBbox(
                HPacker(children=color_explanations, align='baseline', pad=0, sep=0),
                (0.20, 0.43 / figure_height), xycoords=fig.transFigure,
                box_alignment=(0, 0), frameon=False, pad=0))

        plot_path = output_dir / f'{task}_gcca_cognitive_coefficient_map'
        fig.savefig(plot_path.with_suffix('.png'), format='png', dpi=600, bbox_inches='tight', pad_inches=0.04)
        plt.close(fig)

    print(f'Independent task coefficient grids saved to {output_dir}')

    # ==================================================================================================================
    # Cross-task GCCA: interpretation of the saved task-specific aligned scores_cv
    # ==================================================================================================================
    cross_task_output = output_dir / 'Cross_Task_GCCA_Interpretation'
    cross_task_output.mkdir(parents=True, exist_ok=True)
    cross_task_regression_tables = []
    cross_task_correlation_tables = []
    cross_task_model_test_tables = []

    for task, model in selected_models.items():
        names = model_parameters[model]
        aligned_scores = np.load(models_dir / selected_config / f'{task}_cross_task_gcca_embeddings.npy')[:, :1]
        outcome_names = ['cross_gcca_1']
        # Saved score rows follow the task's original consensus CSV participant order.
        participants = pd.read_csv(embedding_dir / f'{selected_config}_{task}.csv', usecols=['worker_id'])
        participants[outcome_names] = aligned_scores

        fit_paths = sorted((models_dir / 'CogModels' / task).glob(f'{model}_fit_*.csv'))
        fits = pd.concat(pd.read_csv(path, usecols=['participant_id', 'best_parameters']) for path in fit_paths)
        fits[names] = np.stack([np.fromstring(value.strip('[]'), sep=' ') for value in fits['best_parameters']])
        parameters = fits.groupby('participant_id')[names].mean().rename_axis('worker_id')
        matched = participants.merge(parameters, on='worker_id')
        regression_data = matched.copy()
        regression_data[outcome_names] = StandardScaler().fit_transform(matched[outcome_names])
        regression_data[names] = StandardScaler().fit_transform(matched[names])
        n = len(matched)

        # One outcome: ordinary multiple regression with all cognitive parameters entered together.
        formula = ("Q('cross_gcca_1') ~ " + ' + '.join(f'Q({name!r})' for name in names))
        cross_task_fit = OLS.from_formula(formula, data=regression_data).fit()
        (cross_task_output / f'{task}_{model}_regression_summary.txt').write_text(
            cross_task_fit.summary().as_text(), encoding='utf-8')
        cross_task_model_test_tables.append({
            'task': task, 'model': model, 'n': n,
            'f': cross_task_fit.fvalue, 'p': cross_task_fit.f_pvalue,
            'num_df': cross_task_fit.df_model, 'den_df': cross_task_fit.df_resid,
            'r_squared': cross_task_fit.rsquared, 'adjusted_r_squared': cross_task_fit.rsquared_adj,
        })
        cross_task_regression_tables.append(pd.DataFrame({
            'task': task, 'model': model, 'n': n, 'outcome': 'cross_gcca_1', 'parameter': names,
            'beta': np.asarray(cross_task_fit.params)[1:], 'se': np.asarray(cross_task_fit.bse)[1:],
            't': np.asarray(cross_task_fit.tvalues)[1:], 'p': np.asarray(cross_task_fit.pvalues)[1:],
        }))
        correlation = pearsonr(regression_data[names].to_numpy(),
                              regression_data['cross_gcca_1'].to_numpy()[:, None], axis=0)
        cross_task_correlation_tables.append(pd.DataFrame({
            'task': task, 'model': model, 'n': n, 'outcome': 'cross_gcca_1', 'parameter': names,
            'r': correlation.statistic, 'p': correlation.pvalue,
        }))
        print(f'{task} | first cross-task dimension | n={n}')

    cross_task_regressions = pd.concat(cross_task_regression_tables, ignore_index=True)
    cross_task_regressions['p_fdr'] = multipletests(cross_task_regressions['p'], method='fdr_bh')[1]
    cross_task_regressions.to_csv(cross_task_output / 'regression_pvalues.csv', index=False)
    cross_task_correlations = pd.concat(cross_task_correlation_tables, ignore_index=True)
    cross_task_correlations['p_fdr'] = multipletests(cross_task_correlations['p'], method='fdr_bh')[1]
    cross_task_correlations.to_csv(cross_task_output / 'correlations.csv', index=False)
    cross_task_overall_tests = pd.DataFrame(cross_task_model_test_tables)
    cross_task_overall_tests['p_fdr'] = multipletests(cross_task_overall_tests['p'], method='fdr_bh')[1]
    cross_task_overall_tests['significance'] = cross_task_overall_tests['p_fdr'] < 0.05
    cross_task_overall_tests.to_csv(cross_task_output / 'overall_regression_tests.csv', index=False)

    # One coefficient plot for the first shared GCCA dimension across all five tasks.
    # Intervals are nominal OLS 95% CIs; stars use BH-FDR across all 22 slopes.
    parameter_order = [(task, parameter) for task, model in selected_models.items()
                       for parameter in model_parameters[model]]
    plot_data = cross_task_regressions.set_index(['task', 'parameter']).loc[parameter_order].reset_index()
    residual_df = cross_task_overall_tests.set_index('task')['den_df']
    critical_t = student_t.ppf(0.975, plot_data.task.map(residual_df))
    plot_data['ci_lower'] = plot_data.beta - critical_t * plot_data.se
    plot_data['ci_upper'] = plot_data.beta + critical_t * plot_data.se
    task_labels = {'cct': 'Columbia Card Task', 'dd': 'Delay Discounting Task',
                   'stopsignal': 'Stop-Signal Task', 'motor': 'Motor Selective Stop-Signal Task',
                   'twobytwo': 'Cued Task Switching Task'}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Three task groups on the left, two on the right, with identical scales and row spacing.
    task_groups = [list(selected_models)[:3], list(selected_models)[3:]]
    panel_bottom = max(sum(len(model_parameters[selected_models[task]]) for task in tasks)
                       + 1.65 * (len(tasks) - 1) for tasks in task_groups) + 1
    extent = np.ceil(np.abs(plot_data[['ci_lower', 'ci_upper']].to_numpy()).max() * 5) / 5
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.7), sharex=True)
    fig.subplots_adjust(left=0.25, right=0.955, bottom=0.27, top=0.98, wspace=1.2)
    for ax, tasks in zip(axes, task_groups):
        cursor = 1.0
        row_positions = []
        row_labels = []
        for task in tasks:
            rows = plot_data.loc[plot_data.task.eq(task)]
            positions = cursor + np.arange(len(rows))
            row_positions.extend(positions)
            row_labels.extend(rows.parameter)
            ax.text(-0.95, cursor - 0.9, task_labels[task], transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=8, fontweight='bold', color='#263743')
            for y, result in zip(positions, rows.itertuples()):
                significant = result.p_fdr < 0.05
                color = ('#C27F6A' if result.beta >= 0 else '#547FA6') if significant else '#9EA8AF'
                ax.errorbar(result.beta, y,
                            xerr=[[result.beta - result.ci_lower], [result.ci_upper - result.beta]],
                            fmt='o', color=color, markerfacecolor=color if significant else 'white',
                            markeredgewidth=0.9, markersize=4.3, elinewidth=0.9,
                            capsize=2, capthick=0.8, zorder=3)
                stars = '*' * sum(result.p_fdr < threshold for threshold in [0.05, 0.01, 0.001])
                ax.text(1.035, y, stars, transform=ax.get_yaxis_transform(), ha='left', va='center',
                        fontsize=8, fontweight='bold', color='#263743')
            cursor += len(rows) + 1.65

        ax.axvline(0, color='#8B969F', linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        ax.set_yticks(row_positions, row_labels)
        ax.set_ylim(panel_bottom, -0.6)
        ax.set_xlim(-extent - 0.05, extent + 0.05)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(axis='y', length=0, pad=7, labelcolor='#263743')
        ax.tick_params(axis='x', length=3, width=0.6, labelcolor='#263743')
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#B8C1C7')
        ax.spines['bottom'].set_linewidth(0.6)
    fig.supxlabel('Standardized regression coefficient, β', x=0.60, y=0.18, fontsize=7)
    color_key = [TextArea(text, textprops={'fontsize': 10, 'color': color})
                 for text, color in [('Red: positive β; ', '#C27F6A'),
                                     ('Blue: negative β; ', '#547FA6'),
                                     ('Empty circles: not significant after FDR.', '#777777')]]
    fig.add_artist(AnnotationBbox(
        HPacker(children=color_key, align='baseline', pad=0, sep=0),
        (0.035, 0.12), xycoords=fig.transFigure,
        box_alignment=(0, 0), frameon=False, pad=0))
    fig.text(0.035, 0.075, 'Error bars: 95% confidence intervals. BH-FDR across 22 coefficients: * p < .05; ** p < .01; *** p < .001',
             fontsize=10, color='#777777')
    plot_path = cross_task_output / 'cross_task_gcca_first_dimension_interpretation_wide'
    plot_data.to_csv(cross_task_output / 'cross_task_gcca_first_dimension_interpretation_wide_source.csv', index=False)
    fig.savefig(plot_path.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print(f'Combined first-dimension GCCA interpretation saved to {cross_task_output}')
