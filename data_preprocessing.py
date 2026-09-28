import os
import numpy as np
import pandas as pd

data_dir = f'{os.getcwd()}/ds004636/'
possible_sessions = ['ses-3', 'ses-2', 'ses-1']

# ======================================================================================================================
# ANT
# ======================================================================================================================
ant_data = []
ant_col = ['correct', 'cue', 'flanker_location', 'flanker_middle_direction', 'flanker_type', 'responded', 'worker_id']
i = 0

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_ant = False

    for ses in possible_sessions:
        ant_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(ant_dir):
            continue

        for file in os.listdir(ant_dir):
            if file.endswith(".tsv") and "ANT" in file:
                file_path = os.path.join(ant_dir, file)

                df = pd.read_csv(file_path, sep="\t")
                if 'trial_id' in df.columns:
                    # Standard ANT files alternate cue and stimulus rows. Match each stimulus to its preceding cue instead of
                    # pairing rows by position: sub-s561 starts with a stimulus row, which shifts every positional pair.
                    df['cue'] = df['cue'].ffill()
                    df = df[df['trial_id'] == 'stim'].copy()

                    missing_cue = df['cue'].isna()
                    if missing_cue.any():
                        # sub-s561 has one leading stimulus whose omitted cue is known to be spatial.
                        if sub_dir == 'sub-s561' and missing_cue.sum() == 1 and missing_cue.iloc[0]:
                            df.loc[missing_cue, 'cue'] = 'spatial'
                        else:
                            raise ValueError(f"Unexpected missing ANT cue(s) in {file_path}")
                elif df['cue'].isna().any():
                    # Session-3 ANT replacements are already stored as one row per stimulus trial.
                    raise ValueError(f"Unexpected missing ANT cue(s) in collapsed file {file_path}")

                # Use RT only to verify response presence; do not transform or retain its numeric value.
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"ANT response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df = df[ant_col].reset_index(drop=True)
                ant_data.append(df)

                found_ant = True
                break

        if found_ant:
            break

    if not found_ant:
        print(f"Participant {sub_dir} does not have ANT file in ses-3, ses-2, or ses-1.")

ant_data = pd.concat(ant_data, ignore_index=True)
print(f'Total ANT data shape: {ant_data.shape}; number of unique participants: {ant_data["worker_id"].nunique()}')
print(f'ANT data contains NaN: {ant_data.isnull().values.any()}')

# ======================================================================================================================
# Delayed Discounting
# ======================================================================================================================
dd_data = []
dd_col = ['choice', 'large_amount', 'later_delay', 'small_amount', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_dd = False

    for ses in possible_sessions:
        dd_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(dd_dir):
            continue

        for file in os.listdir(dd_dir):
            if file.endswith(".tsv") and "discountFix" in file:
                file_path = os.path.join(dd_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Use RT only to verify response presence; do not transform or retain its numeric value.
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"DD response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df = df[dd_col]
                dd_data.append(df)

                found_dd = True
                break

        if found_dd:
            break

    if not found_dd:
        print(f"Participant {sub_dir} does not have discountFix file in ses-3, ses-2, or ses-1.")

dd_data = pd.concat(dd_data, ignore_index=True)
print(f'Total Delayed Discounting data shape: {dd_data.shape}; number of unique participants: {dd_data["worker_id"].nunique()}')
print(f'Delayed Discounting data contains NaN: {dd_data.isnull().values.any()}')

# ======================================================================================================================
# Columbia card task
# ======================================================================================================================
cct_data = []
cct_col = ['action', 'clicked_on_loss_card', 'gain_amount', 'gain_probability', 'loss_amount', 'loss_probability',
           'num_cards', 'num_click_in_round', 'num_loss_cards', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_cct = False

    for ses in possible_sessions:
        cct_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(cct_dir):
            continue

        for file in os.listdir(cct_dir):
            if file.endswith(".tsv") and "CCTHot" in file:
                file_path = os.path.join(cct_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Remove all feedback and ITI
                df = df[df['trial_id'] == 'stim'].copy()
                # Use RT only to verify response presence; do not transform or retain its numeric value.
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"CCT response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df = df[cct_col]
                cct_data.append(df)

                found_cct = True
                break

        if found_cct:
            break

    if not found_cct:
        print(f"Participant {sub_dir} does not have discountFix file in ses-3, ses-2, or ses-1.")

cct_data = pd.concat(cct_data, ignore_index=True)
print(f'Total Columbia card task shape: {cct_data.shape}; number of unique participants: {cct_data["worker_id"].nunique()}')
print(f'Columbia card task contains NaN: {cct_data.isnull().values.any()}')

# ======================================================================================================================
# Stroop Task
# ======================================================================================================================
stroop_data = []
stroop_col = ['correct', 'stim_color', 'stim_word', 'trial_type', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_stroop = False

    for ses in possible_sessions:
        stroop_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(stroop_dir):
            continue

        for file in os.listdir(stroop_dir):
            if file.endswith(".tsv") and "stroop" in file:
                file_path = os.path.join(stroop_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Use RT only to verify response presence; do not transform or retain its numeric value.
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"Stroop response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df = df[stroop_col]
                stroop_data.append(df)

                found_stroop = True
                break

        if found_stroop:
            break

    if not found_stroop:
        print(f"Participant {sub_dir} does not have stroop file in ses-3, ses-2, or ses-1.")

stroop_data = pd.concat(stroop_data, ignore_index=True)
print(f'Total Columbia card task shape: {stroop_data.shape}; number of unique participants: {stroop_data["worker_id"].nunique()}')
print(f'Columbia card task contains NaN: {stroop_data.isnull().values.any()}')


# ======================================================================================================================
# Motor selective stop-signal task
# ======================================================================================================================
motor_data = []
motor_col = ['block_duration', 'response_time', 'SS_delay', 'correct', 'trial_condition', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_motor = False

    for ses in possible_sessions:
        motor_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(motor_dir):
            continue

        for file in os.listdir(motor_dir):
            if file.endswith(".tsv") and "motorSelectiveStop" in file:
                file_path = os.path.join(motor_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Match cleaned timing records to original BIDS trials; never join by row position.
                duration_df = pd.read_csv(f'./Data/data_with_duration/{sub_dir[4:]}_motorSelectiveStop_cleaned.csv')
                duration_df = duration_df.loc[duration_df.exp_stage.eq('test') & duration_df.trial_id.eq('stim')].copy()
                duration_df['onset'] = (duration_df.time_elapsed - duration_df.block_duration) / 1000
                duration_df['response_time'] = duration_df.rt.replace(-1, np.nan) / 1000
                duration_df['SS_delay'] = duration_df.SS_delay / 1000
                duration_df['block_duration'] = duration_df.block_duration / 1000  # seconds, like response_time
                match_columns = ['worker_id', 'onset', 'key_press', 'correct_response',
                                 'response_time', 'SS_delay', 'stopped']
                match_keys = df[match_columns].copy()
                for column in ['onset', 'response_time', 'SS_delay']:
                    match_keys[column] = match_keys[column].round(6)
                    duration_df[column] = duration_df[column].round(6)
                matched = match_keys.merge(duration_df[match_columns + ['block_duration']],
                                           on=match_columns, how='left', sort=False,
                                           validate='one_to_one', indicator=True)
                if not matched['_merge'].eq('both').all():
                    raise ValueError(f'Duration records do not match original trials: {file_path}')
                if not (np.isfinite(matched.block_duration) & matched.block_duration.gt(0)).all():
                    raise ValueError(f'Invalid block duration: {file_path}')
                df['block_duration'] = matched.block_duration.to_numpy()

                # No response is successful inhibition on critical stop trials but an omission on go trials.
                # Use RT only to verify response presence; do not transform or retain its numeric value.
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"Motor response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df['trial_condition'] = df['trial_type'].replace({'crit_stop_success': 'crit_stop',
                                                                  'crit_stop_failure': 'crit_stop'})
                df['expected_stopped'] = df['trial_condition'].eq('crit_stop')
                df['correct'] = np.where(
                    df['expected_stopped'],
                    # On crit_stop trials, correct means successfully withholding
                    df['stopped'] == True,
                    # On all other trials, correct means responding with the correct key
                    (df['stopped'] == False) & (df['key_press'] == df['correct_response'])
                ).astype(int)
                # There was only one trial with a negative RT. It seems like the clock went backwards for some reason.
                # By checking the raw data, we think the RT should be approximately 0.625 seconds.
                df['response_time'] = df['response_time'].apply(lambda x: 0.625 if x < 0 else x)
                df = df[motor_col]
                motor_data.append(df)

                found_motor = True
                break

        if found_motor:
            break

    if not found_motor:
        print(f"Participant {sub_dir} does not have motor file in ses-3, ses-2, or ses-1.")

motor_data = pd.concat(motor_data, ignore_index=True)
print(f'Total Motor selective stop-signal task shape: {motor_data.shape}; number of unique participants: {motor_data["worker_id"].nunique()}')
print(f'Motor selective stop-signal task contains NaN: {motor_data.isnull().values.any()}')


# ======================================================================================================================
# Dot Pattern Expectancy task
# ======================================================================================================================
dpx_data = []
dpx_col = ['condition', 'correct', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_dpx = False

    for ses in possible_sessions:
        dpx_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(dpx_dir):
            continue

        for file in os.listdir(dpx_dir):
            if file.endswith(".tsv") and "DPX" in file:
                file_path = os.path.join(dpx_dir, file)
                df = pd.read_csv(file_path, sep='\t')

                # Cue rows do not expect responses; each probe row contains the behavioral outcome for one DPX trial.
                df = df[df['trial_id'] == 'probe'].copy()
                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"DPX response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df['correct'] = df['correct'].astype(int)
                df = df[dpx_col].reset_index(drop=True)
                dpx_data.append(df)

                found_dpx = True
                break

        if found_dpx:
            break

    if not found_dpx:
        print(f"Participant {sub_dir} does not have DPX file in ses-3, ses-2, or ses-1.")

dpx_data = pd.concat(dpx_data, ignore_index=True)
print(f'Total DPX data shape: {dpx_data.shape}; number of unique participants: {dpx_data["worker_id"].nunique()}')
print(f'DPX data contains NaN: {dpx_data.isnull().values.any()}')

# ======================================================================================================================
# Stop-signal task
# ======================================================================================================================
stop_signal_data = []
stop_signal_col = ['block_duration', 'response_time', 'SS_delay', 'correct', 'trial_condition', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_stop_signal = False

    for ses in possible_sessions:
        stop_signal_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(stop_signal_dir):
            continue

        for file in os.listdir(stop_signal_dir):
            if file.endswith(".tsv") and "stopSignal" in file:
                file_path = os.path.join(stop_signal_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                # Match cleaned timing records to original BIDS trials; never join by row position.
                duration_df = pd.read_csv(f'./Data/data_with_duration/{sub_dir[4:]}_stopSignal_cleaned.csv')
                duration_df = duration_df.loc[duration_df.exp_stage.eq('test') & duration_df.trial_id.eq('stim')].copy()
                duration_df['onset'] = (duration_df.time_elapsed - duration_df.block_duration) / 1000
                duration_df['response_time'] = duration_df.rt.replace(-1, np.nan) / 1000
                duration_df['SS_delay'] = duration_df.SS_delay / 1000
                duration_df['block_duration'] = duration_df.block_duration / 1000  # seconds, like response_time
                match_columns = ['worker_id', 'onset', 'key_press', 'correct_response',
                                 'response_time', 'SS_delay', 'stopped']
                match_keys = df[match_columns].copy()
                for column in ['onset', 'response_time', 'SS_delay']:
                    match_keys[column] = match_keys[column].round(6)
                    duration_df[column] = duration_df[column].round(6)
                matched = match_keys.merge(duration_df[match_columns + ['block_duration']],
                                           on=match_columns, how='left', sort=False,
                                           validate='one_to_one', indicator=True)
                if not matched['_merge'].eq('both').all():
                    raise ValueError(f'Duration records do not match original trials: {file_path}')
                if not (np.isfinite(matched.block_duration) & matched.block_duration.gt(0)).all():
                    raise ValueError(f'Invalid block duration: {file_path}')
                df['block_duration'] = matched.block_duration.to_numpy()

                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"Stop-signal response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)

                # Do not retain success/failure in the condition because it would directly reveal the outcome.
                df['trial_condition'] = df['trial_type'].replace({'stop_success': 'stop',
                                                                  'stop_failure': 'stop'})
                df['expected_stopped'] = df['trial_condition'].eq('stop')
                df['correct'] = np.where(
                    df['expected_stopped'],
                    # On stop trials, correct means successfully withholding.
                    df['stopped'] == True,
                    # On go trials, correct means responding with the correct key.
                    (df['stopped'] == False) & (df['key_press'] == df['correct_response'])
                ).astype(int)
                df = df[stop_signal_col]
                stop_signal_data.append(df)

                found_stop_signal = True
                break

        if found_stop_signal:
            break

    if not found_stop_signal:
        print(f"Participant {sub_dir} does not have stopSignal file in ses-3, ses-2, or ses-1.")

stop_signal_data = pd.concat(stop_signal_data, ignore_index=True)
print(f'Total stop-signal data shape: {stop_signal_data.shape}; number of unique participants: {stop_signal_data["worker_id"].nunique()}')
print(f'Stop-signal data contains NaN: {stop_signal_data.isnull().values.any()}')

# ======================================================================================================================
# Two-by-two task (N = 106)
# ======================================================================================================================
two_by_two_data = []
two_by_two_col = ['CTI', 'correct', 'cue', 'stim_color', 'stim_number',
                  'switch_type', 'task_type', 'responded', 'worker_id']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith("sub-"):
        continue

    found_two_by_two = False

    for ses in possible_sessions:
        two_by_two_dir = os.path.join(data_dir, sub_dir, ses, "func")

        if not os.path.isdir(two_by_two_dir):
            continue

        for file in os.listdir(two_by_two_dir):
            if file.endswith(".tsv") and "twoByTwo" in file:
                file_path = os.path.join(two_by_two_dir, file)
                df = pd.read_csv(file_path, sep='\t')

                rt_indicates_response = df['response_time'].notna()
                key_indicates_response = df['key_press'].ne(-1)
                if not rt_indicates_response.equals(key_indicates_response):
                    raise ValueError(f"Two-by-two response-time/key-press mismatch in {file_path}")
                df['responded'] = key_indicates_response.astype(int)
                df['correct'] = df['correct'].astype(int)
                df = df[two_by_two_col]
                two_by_two_data.append(df)

                found_two_by_two = True
                break

        if found_two_by_two:
            break

    if not found_two_by_two:
        print(f"Participant {sub_dir} does not have twoByTwo file in ses-3, ses-2, or ses-1.")

two_by_two_data = pd.concat(two_by_two_data, ignore_index=True)
print(f'Total two-by-two data shape: {two_by_two_data.shape}; number of unique participants: {two_by_two_data["worker_id"].nunique()}')
print(f'Two-by-two data contains NaN: {two_by_two_data.isnull().values.any()}')

# ======================================================================================================================
# Ward and Allport Tower Task (WATT3; Unused)
# ======================================================================================================================
# watt3_data = []
# watt3_col = ['planning', 'condition', 'pre_move_position', 'goal_state', 'start_state', 'problem_id', 'trial_id',
#              'key_press', 'responded', 'worker_id']
#
# for sub_dir in os.listdir(data_dir):
#     if not sub_dir.startswith("sub-"):
#         continue
#
#     found_watt3 = False
#
#     for ses in possible_sessions:
#         watt3_dir = os.path.join(data_dir, sub_dir, ses, "func")
#
#         if not os.path.isdir(watt3_dir):
#             continue
#
#         for file in os.listdir(watt3_dir):
#             if file.endswith(".tsv") and "WATT3" in file:
#                 file_path = os.path.join(watt3_dir, file)
#                 df = pd.read_csv(file_path, sep='\t')
#
#                 # Keep movements only. Instructions, feedback, and ITIs do not represent choices.
#                 df = df[df['trial_id'].isin(['to_hand', 'to_board'])].copy()
#
#                 # current_position is the state after the recorded action and would leak the outcome. Reconstruct the
#                 # state available before each move, using start_state for the first move of each problem.
#                 df['pre_move_position'] = df.groupby(['exp_stage', 'problem_id'], sort=False)['current_position'].shift(1)
#                 df['pre_move_position'] = df['pre_move_position'].fillna(df['start_state'])
#
#                 # Practice problems are not part of the experimental test data.
#                 df = df[df['exp_stage'] == 'test'].copy()
#                 rt_indicates_response = df['response_time'].notna()
#                 key_indicates_response = df['key_press'].ne(-1)
#                 if not rt_indicates_response.equals(key_indicates_response):
#                     raise ValueError(f"WATT3 response-time/key-press mismatch in {file_path}")
#                 df['responded'] = key_indicates_response.astype(int)
#                 df = df[watt3_col].reset_index(drop=True)
#                 watt3_data.append(df)
#
#                 found_watt3 = True
#                 break
#
#         if found_watt3:
#             break
#
#     if not found_watt3:
#         print(f"Participant {sub_dir} does not have WATT3 file in ses-3, ses-2, or ses-1.")
#
# watt3_data = pd.concat(watt3_data, ignore_index=True)
# print(f'Total WATT3 data shape: {watt3_data.shape}; number of unique participants: {watt3_data["worker_id"].nunique()}')
# print(f'WATT3 data contains NaN: {watt3_data.isnull().values.any()}')

# ======================================================================================================================
# Survey medley (item-level questionnaire responses)
# ======================================================================================================================
survey_data = []
survey_col = ['worker_id', 'survey', 'trial_type', 'item_text', 'item_coding', 'coded_response', 'responded']

for sub_dir in os.listdir(data_dir):
    if not sub_dir.startswith('sub-'):
        continue

    found_survey = False
    for ses in possible_sessions:
        survey_dir = os.path.join(data_dir, sub_dir, ses, 'func')
        if not os.path.isdir(survey_dir):
            continue

        for file in sorted(os.listdir(survey_dir)):
            if file.endswith(".tsv") and "surveyMedley_" in file:
                file_path = os.path.join(survey_dir, file)
                df = pd.read_csv(file_path, sep='\t')
                df['coded_response'] = pd.to_numeric(df['coded_response'], errors='raise')
                df['responded'] = (df['key_press'].notna() & df['key_press'].ne(-1) & df['response_time'].notna() &
                                   df['coded_response'].notna() & ~df['coded_response'].isin([0.0, 6.0])).astype(int)
                df = df[survey_col].reset_index(drop=True)
                survey_data.append(df)
                found_survey = True
                break

        if found_survey:
            break

    if not found_survey:
        print(f'Participant {sub_dir} does not have surveyMedley file in ses-3, ses-2, or ses-1.')

survey_data = pd.concat(survey_data, ignore_index=True)
print(f'Saved survey data shape: {survey_data.shape}; number of unique participants: {survey_data["worker_id"].nunique()}')


# ======================================================================================================================
# Restrict all saved task files to the common nine-task cohort
# ======================================================================================================================
common_participants_4task = (
    set(dd_data['worker_id'])
    & set(cct_data['worker_id'])
    & set(motor_data['worker_id'])
    & set(stop_signal_data['worker_id'])
    & set(survey_data['worker_id'])
)

common_participants_8task = (
    set(ant_data['worker_id'])
    & set(dd_data['worker_id'])
    & set(cct_data['worker_id'])
    & set(stroop_data['worker_id'])
    & set(motor_data['worker_id'])
    & set(dpx_data['worker_id'])
    & set(stop_signal_data['worker_id'])
    & set(two_by_two_data['worker_id'])
    & set(survey_data['worker_id'])
)

# 4 task cohort: delayed discounting, Columbia card task, motor selective stop-signal, stop-signal, survey medley
ant_data = ant_data[ant_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
dd_data = dd_data[dd_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
cct_data = cct_data[cct_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
stroop_data = stroop_data[stroop_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
motor_data = motor_data[motor_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
dpx_data = dpx_data[dpx_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
stop_signal_data = stop_signal_data[stop_signal_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
two_by_two_data = two_by_two_data[two_by_two_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)
survey_data = survey_data[survey_data['worker_id'].isin(common_participants_4task)].reset_index(drop=True)

# 8 task cohort
ant_data_8task = ant_data[ant_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
dd_data_8task = dd_data[dd_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
cct_data_8task = cct_data[cct_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
stroop_data_8task = stroop_data[stroop_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
motor_data_8task = motor_data[motor_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
dpx_data_8task = dpx_data[dpx_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
stop_signal_data_8task = stop_signal_data[stop_signal_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
two_by_two_data_8task = two_by_two_data[two_by_two_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)
survey_data_8task = survey_data[survey_data['worker_id'].isin(common_participants_8task)].reset_index(drop=True)

# Save all common-cohort task files together only after every preprocessing pipeline has completed.
ant_data.to_csv('./Data/hypernetwork_data/ant_data.csv', index=False)
dd_data.to_csv('./Data/hypernetwork_data/dd_data.csv', index=False)
cct_data.to_csv('./Data/hypernetwork_data/cct_data.csv', index=False)
stroop_data.to_csv('./Data/hypernetwork_data/stroop_data.csv', index=False)
motor_data.to_csv('./Data/hypernetwork_data/motor_data.csv', index=False)
dpx_data.to_csv('./Data/hypernetwork_data/dpx_data.csv', index=False)
stop_signal_data.to_csv('./Data/hypernetwork_data/stop_signal_data.csv', index=False)
two_by_two_data.to_csv('./Data/hypernetwork_data/two_by_two_data.csv', index=False)
survey_data.to_csv('./Data/hypernetwork_data/survey_data.csv', index=False)

ant_data_8task.to_csv('./Data/hypernetwork_data_8_task/ant_data_8task.csv', index=False)
dd_data_8task.to_csv('./Data/hypernetwork_data_8_task/dd_data_8task.csv', index=False)
cct_data_8task.to_csv('./Data/hypernetwork_data_8_task/cct_data_8task.csv', index=False)
motor_data_8task.to_csv('./Data/hypernetwork_data_8_task/motor_data_8task.csv', index=False)
dpx_data_8task.to_csv('./Data/hypernetwork_data_8_task/dpx_data_8task.csv', index=False)
stop_signal_data_8task.to_csv('./Data/hypernetwork_data_8_task/stop_signal_data_8task.csv', index=False)
two_by_two_data_8task.to_csv('./Data/hypernetwork_data_8_task/two_by_two_data_8task.csv', index=False)
survey_data_8task.to_csv('./Data/hypernetwork_data_8_task/survey_data_8task.csv', index=False)

print(f'Common four-task cohort: {len(common_participants_4task)} participants')
print(f'Common eight-task cohort: {len(common_participants_8task)} participants')

