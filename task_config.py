TASK_CONFIGS = {
    'ANT': {
        'categorical_x': ['cue', 'flanker_location', 'flanker_middle_direction', 'flanker_type'],
        'continuous_x': [],
        "y_col": 'correct',
        'y_transform': 'int',
        'mask_col': 'true'
    },

    'CCT': {
        'categorical_x': [],
        'continuous_x': ['gain_amount', 'loss_amount', 'num_cards', 'num_click_in_round', 'num_loss_cards'],
        'y_col': 'action',
        'y_transform': 'category',
        'mask_col': 'true'
    },

    'DD': {
        'categorical_x': [],
        'continuous_x': ['large_amount', 'later_delay', 'small_amount'],
        'y_col': 'choice',
        'y_transform': 'category_minus_1',
        'mask_col': 'true'
    },

    'Motor': {
        'categorical_x': ['trial_condition'],
        'continuous_x': ['SS_delay', 'block_duration'],
        'y_col': 'correct',
        'y_transform': 'int',
        'mask_col': 'false'
    },

    'Stroop': {
        'categorical_x': ['stim_color', 'stim_word'],
        'continuous_x': [],
        'y_col': 'correct',
        'y_transform': 'int',
        'mask_col': 'true'
    },

    'Dpx': {
        'categorical_x': ['condition'],
        'continuous_x': [],
        'y_col': 'correct',
        'y_transform': 'category',
        'mask_col': 'true'
    },

    'TwoByTwo': {
        'categorical_x': ['cue', 'stim_color', 'stim_number', 'switch_type', 'task_type'],
        'continuous_x': ['CTI', 'block_duration'],
        'y_col': 'correct',
        'y_transform': 'int',
        'mask_col': 'false'
    },

    'StopSignal': {
        'categorical_x': ['trial_condition'],
        'continuous_x': ['SS_delay', 'block_duration'],
        'y_col': 'correct',
        'y_transform': 'int',
        'mask_col': 'false'
    },

}