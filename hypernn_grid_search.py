import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from HyperNetwork import *
from task_config import TASK_CONFIGS

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ======================================================================================================================
# Read in all data
# ======================================================================================================================
cct_data = pd.read_csv('./Data/hypernetwork_data/cct_data.csv')
dd_data = pd.read_csv('./Data/hypernetwork_data/dd_data.csv')
motor_data = pd.read_csv('./Data/hypernetwork_data/motor_data.csv')
stopsignal_data = pd.read_csv('./Data/hypernetwork_data/stop_signal_data.csv')
twobytwo_data = pd.read_csv('./Data/hypernetwork_data/two_by_two_data.csv')
# ant_data = pd.read_csv('./Data/hypernetwork_data/ant_data.csv')
# stroop_data = pd.read_csv('./Data/hypernetwork_data/stroop_data.csv')
# dpx_data = pd.read_csv('./Data/hypernetwork_data/dpx_data.csv')


dis = (twobytwo_data.groupby('worker_id')['correct'].mean().reset_index().rename(columns={'correct': 'discrimination'})
       .sort_values('discrimination', ascending=False).reset_index(drop=True))
# dpx: 3 perfect; 10 > 99%
# ant: 30 perfect; 47 > 99%
# stroop: 12 perfect; 12 > 99%

# ======================================================================================================================
# Preprocess each task's data
# ======================================================================================================================
cct_data, cct_x_col, cct_y_col = preprocess_task(cct_data, TASK_CONFIGS["CCT"], task_name='CCT')
dd_data, dd_x_col, dd_y_col = preprocess_task(dd_data, TASK_CONFIGS["DD"], task_name='DD')
motor_data, motor_x_col, motor_y_col = preprocess_task(motor_data, TASK_CONFIGS["Motor"], task_name='Motor')
stopsignal_data, stopsignal_x_col, stopsignal_y_col = preprocess_task(stopsignal_data, TASK_CONFIGS["StopSignal"], task_name='StopSignal')
twobytwo_data, twobytwo_x_col, twobytwo_y_col = preprocess_task(twobytwo_data, TASK_CONFIGS["TwoByTwo"], task_name='TwoByTwo')
# ant_data, ant_x_col, ant_y_col = preprocess_task(ant_data, TASK_CONFIGS["ANT"], task_name='ANT')
# stroop_data, stroop_x_col, stroop_y_col = preprocess_task(stroop_data, TASK_CONFIGS["Stroop"], task_name='Stroop')
# dpx_data, dpx_x_col, dpx_y_col = preprocess_task(dpx_data, TASK_CONFIGS["Dpx"], task_name='Dpx')

# Each preprocessing pipeline provides the same pieces needed by the shared training loop.
# tasks = [
#     {"name": "ant", "data": ant_data, "x_col": ant_x_col, "y_col": ant_y_col},
#     {"name": "cct", "data": cct_data, "x_col": cct_x_col, "y_col": cct_y_col},
#     {"name": "dd", "data": dd_data, "x_col": dd_x_col, "y_col": dd_y_col},
#     {"name": "motor", "data": motor_data, "x_col": motor_x_col, "y_col": motor_y_col},
#     {"name": "stroop", "data": stroop_data, "x_col": stroop_x_col, "y_col": stroop_y_col},
#     {"name": "dpx", "data": dpx_data, "x_col": dpx_x_col, "y_col": dpx_y_col},
#     {"name": "stopsignal", "data": stopsignal_data, "x_col": stopsignal_x_col, "y_col": stopsignal_y_col},
#     {"name": "twobytwo", "data": twobytwo_data, "x_col": twobytwo_x_col, "y_col": twobytwo_y_col},
# ]

tasks = [
    {"name": "cct", "data": cct_data, "x_col": cct_x_col, "y_col": cct_y_col},
    {"name": "dd", "data": dd_data, "x_col": dd_x_col, "y_col": dd_y_col},
    {"name": "motor", "data": motor_data, "x_col": motor_x_col, "y_col": motor_y_col},
    {"name": "stopsignal", "data": stopsignal_data, "x_col": stopsignal_x_col, "y_col": stopsignal_y_col},
    {"name": "twobytwo", "data": twobytwo_data, "x_col": twobytwo_x_col, "y_col": twobytwo_y_col},
]



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_folds = 10

    # Grid search
    num_hidden_dim = [8, 16, 32, 64]
    num_hidden_layers = [1]
    num_embedding_dim = [2, 4, 8, 16]
    num_hyper_hidden_dim = [4, 8, 16, 32]
    # hyper_reg_values = [0.0]
    hyper_reg_values = [0.0, 1e-4, 1e-3, 1e-2]

    # # Regularization search
    # num_hidden_dim = [8]
    # num_hidden_layers = [1]
    # num_embedding_dim = [4]
    # num_hyper_hidden_dim = [8]
    # hyper_reg_values = [0.0]  # Edit the values to test.

    batch_size = 16
    n_epochs = 400
    learning_rate = 1e-3
    patience = 50
    emb_reg = 0.0
    seed = 42

    # Create a grid of hyperparameter configurations to explore
    grid = []
    for hidden_layers in num_hidden_layers:
        for hidden_dim in num_hidden_dim:
            hidden_dims = (hidden_dim,) * hidden_layers
            # for rank in [2, None]:
            for rank in [None]:
                for participant_emb_dim in num_embedding_dim:
                # # Comment out the linear mapping for regularization search, since it doesn't have hyper_hidden_dim.
                #     grid.append({
                #         "num_hidden_layers": hidden_layers,
                #         "hidden_dims": hidden_dims,
                #         "rank": rank,
                #         "participant_emb_dim": participant_emb_dim,
                #         "mapping": "linear",
                #         "hyper_hidden_dim": None,
                #     })
                    for hyper_hidden_dim in num_hyper_hidden_dim:
                        grid.append({
                            "num_hidden_layers": hidden_layers,
                            "hidden_dims": hidden_dims,
                            "rank": rank,
                            "participant_emb_dim": participant_emb_dim,
                            "mapping": "nonlinear",
                            "hyper_hidden_dim": hyper_hidden_dim,
                        })

    grid = [{**config, "hyper_reg": value} for config in grid for value in hyper_reg_values]

    save_dir = "./Results/HyperNN_Grid_Search"
    # save_dir = "./Results/HyperNN_Reg_Search/" # Change to this after the config search is done
    os.makedirs(save_dir, exist_ok=True)

    # Save a mapping of participant IDs to worker IDs for reference
    # Note that we have to transform worker_id into participant_id because the hypernetwork model needs interger
    # participant IDs as indices into the embedding layer, whereas the original worker_id is a string.
    participant_map = (tasks[0]["data"][["participant_id", "worker_id"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True))
    for task in tasks[1:]:
        task_map = (task["data"][["participant_id", "worker_id"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True))
        if not participant_map.equals(task_map):
            raise ValueError(f"Participant indexing differs for task {task['name']}.")
    participant_map_path = os.path.join(save_dir, "participant_map.csv")
    if os.path.exists(participant_map_path):
        # Category codes may be int8, while CSV reloads them as int64.
        # Compare the actual assignments without requiring the same storage dtype.
        try:
            pd.testing.assert_frame_equal(participant_map, pd.read_csv(participant_map_path), check_dtype=False)
        except AssertionError as error:
            raise ValueError("Existing participant map differs; refusing to overwrite it.") from error
    else:
        participant_map.to_csv(participant_map_path, index=False)

    # Save the exact trial assignments once so other models can reuse them.
    random_split_dir = os.path.join(save_dir, "random_splits")
    os.makedirs(random_split_dir, exist_ok=True)
    split_indices = {}
    split_assignment_paths = {}
    split_seed_rows = []

    # Now we can generate the random splits for each task, ensuring that they are consistent across runs and that the assignments are valid.
    for task_number, task in enumerate(tasks):
        key_columns = ["worker_id", "participant_id", "trial"]
        current_keys = task["data"][key_columns].copy()
        if current_keys.duplicated(key_columns).any():
            raise ValueError(f"Trial identifiers are not unique for task {task['name']}.")

        assignment_path = os.path.abspath(os.path.join(random_split_dir, f"{task['name']}_random_splits.csv"))
        required_split_columns = ["fold"] + [f"random_split_{i}" for i in range(split_folds)]

        # Check if the assignment file already exists. If it does, validate that it has the correct columns and
        # that the trial identifiers match the current preprocessed data. If not, generate new random splits and save them to a CSV file.
        if os.path.exists(assignment_path):
            saved_assignments = pd.read_csv(assignment_path)
            missing_columns = [column for column in key_columns + required_split_columns if column not in saved_assignments.columns]
            if missing_columns:
                raise ValueError(f"Saved splits for task {task['name']} are missing columns: {missing_columns}. "
                                 f"Remove that split file if you want it regenerated.")
            if saved_assignments.duplicated(key_columns).any():
                raise ValueError(f"Saved trial identifiers are not unique for task {task['name']}.")

            assignments = current_keys.merge(saved_assignments, on=key_columns, how="left", validate="one_to_one",
                                             sort=False, indicator=True)
            if len(saved_assignments) != len(current_keys) or not assignments["_merge"].eq("both").all():
                raise ValueError(f"Saved splits do not match the current preprocessed rows for task {task['name']}. "
                                 f"Remove that split file if you intentionally changed preprocessing.")
            assignments = assignments.drop(columns="_merge")
            print(f"Loaded existing random splits for {task['name']}: {assignment_path}")
        else:
            assignments = current_keys.copy()

        split_seed = seed + task_number

        # If the assignment file does not exist, generate new random splits and save them to a CSV file.
        # The splits are generated using the `add_interspersed_folds` function, which ensures that the folds are
        # interspersed across the data. Each fold is assigned a label of "train", "validation", or "test" based on
        # the current random split. The assignments are then saved to a CSV file for future reference.
        if not os.path.exists(assignment_path):
            split_data = add_interspersed_folds(task["data"], n_folds=split_folds, seed=split_seed)
            assignments["fold"] = split_data["fold"].to_numpy()

            for random_split in range(split_folds):
                test_fold = random_split
                validation_fold = (random_split + 1) % split_folds
                is_validation = assignments["fold"].eq(validation_fold)
                is_test = assignments["fold"].eq(test_fold)
                assignments[f"random_split_{random_split}"] = np.select(
                    [is_validation, is_test],
                    ["validation", "test"],
                    default="train",
                )
            assignments.to_csv(assignment_path, index=False)

        if not assignments["fold"].isin(range(split_folds)).all():
            raise ValueError(f"Saved folds for task {task['name']} must be integers from 0 to {split_folds - 1}.")

        fold_assignments = pd.Series(assignments["fold"].to_numpy(), index=task["data"].index)

        # For each random split, validate that the saved assignments are consistent with the fold assignments.
        for random_split in range(split_folds):
            split_column = f"random_split_{random_split}"
            test_fold = random_split
            validation_fold = (random_split + 1) % split_folds
            is_validation = fold_assignments.eq(validation_fold)
            is_test = fold_assignments.eq(test_fold)
            is_training = ~(is_validation | is_test)

            expected_labels = np.select([assignments["fold"].eq(validation_fold), assignments["fold"].eq(test_fold)],
                                        ["validation", "test"],  default="train")
            if not np.array_equal(assignments[split_column].to_numpy(), expected_labels):
                raise ValueError(f"Saved column {split_column} is inconsistent with fold assignments for task {task['name']}.")
            split_indices[(random_split, task["name"])] = {
                "train": task["data"].index[is_training].to_numpy(),
                "validation": task["data"].index[is_validation].to_numpy(),
                "test": task["data"].index[is_test].to_numpy(),
            }
            split_seed_rows.append({
                "task": task["name"],
                "random_split": random_split,
                "split_seed": split_seed,
                "split_column": split_column,
                "validation_fold": validation_fold,
                "testing_fold": test_fold,
                "training_trials": int(is_training.sum()),
                "validation_trials": int(is_validation.sum()),
                "testing_trials": int(is_test.sum()),
            })

        split_assignment_paths[task["name"]] = assignment_path

    split_seed_manifest = pd.DataFrame(split_seed_rows)
    split_seed_manifest["split_assignment_path"] = split_seed_manifest["task"].map(split_assignment_paths)
    split_seed_manifest.to_csv(os.path.join(random_split_dir, "random_split_manifest.csv"), index=False)

    manifest_path = os.path.join(save_dir, "training_manifest.csv")
    manifest_rows = (pd.read_csv(manifest_path).to_dict("records")
                     if os.path.exists(manifest_path) else [])
    total_runs = split_folds * len(grid)

    for random_split in range(split_folds):
        print(f"Fold rotation {random_split + 1}/{split_folds}")

        for config_number, config in enumerate(grid, start=1):
            hyper_reg = config["hyper_reg"]
            rank_label = "full" if config["rank"] is None else str(config["rank"])
            node_label = (
                "na" if config["hyper_hidden_dim"] is None
                else str(config["hyper_hidden_dim"])
            )
            width_label = "x".join(
                str(width) for width in config["hidden_dims"]
            )
            config_id = (
                f"layers_{config['num_hidden_layers']}_dims_{width_label}_"
                f"rank_{rank_label}_"
                f"emb_{config['participant_emb_dim']}_{config['mapping']}_"
                f"nodes_{node_label}"
                f"_reg_{hyper_reg}"
            )
            run_number = random_split * len(grid) + config_number
            print(f"[{run_number:03d}/{total_runs}] {config_id}")

            split_dir = os.path.join(save_dir, config_id, f"random_split_{random_split}")
            os.makedirs(split_dir, exist_ok=True)

            for task_number, task in enumerate(tasks):
                split_seed = seed + task_number
                test_fold = random_split
                validation_fold = (random_split + 1) % split_folds
                task_seed = seed + 10000 * random_split + task_number
                random.seed(task_seed)
                np.random.seed(task_seed)
                torch.manual_seed(task_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(task_seed)

                task_split_indices = split_indices[(random_split, task["name"])]
                is_training = task["data"].index.isin(task_split_indices["train"])
                is_validation = task["data"].index.isin(task_split_indices["validation"])
                is_test = task["data"].index.isin(task_split_indices["test"])
                if not np.all(is_training.astype(int) + is_validation.astype(int) + is_test.astype(int) == 1):
                    raise ValueError(f"Train/validation/test assignments overlap or omit rows for task {task['name']}.")

                train_df = task["data"].loc[is_training].copy()
                val_df = task["data"].loc[is_validation].copy()
                split_column = f"random_split_{random_split}"
                split_assignment_path = split_assignment_paths[task["name"]]

                checkpoint_path = os.path.abspath(os.path.join(split_dir, f"{task['name']}_model.pt"))
                embedding_path = os.path.abspath(os.path.join(split_dir, f"{task['name']}_embeddings.npy"))
                history_path = os.path.abspath(os.path.join(split_dir, f"{task['name']}_history.csv"))
                expected_checkpoint_config = {
                    **config,
                    "model": "HyperNN",
                    "emb_reg": emb_reg,
                    "hyper_reg": hyper_reg,
                    "hidden_dims": list(config["hidden_dims"]),
                    "hidden_dim": config["hidden_dims"][0],
                    "random_split": random_split,
                    "split_seed": split_seed,
                    "split_assignment_path": split_assignment_path,
                    "split_column": split_column,
                    "validation_fold": validation_fold,
                    "testing_fold": test_fold,
                    "training_fraction": 0.8,
                    "validation_fraction": 0.1,
                    "testing_fraction": 0.1,
                    "shared_right": False,
                }

                result_paths = (checkpoint_path, embedding_path, history_path)
                completed_files_exist = all(os.path.isfile(path) and os.path.getsize(path) > 0
                                            for path in result_paths)
                checkpoint_matches = False
                if any(os.path.exists(path) for path in result_paths):
                    # Existing output is never permission to overwrite, even if incomplete.
                    if not completed_files_exist:
                        raise RuntimeError(f"Incomplete existing results: {checkpoint_path}. "
                                           "Inspect them or use a new output folder; nothing was overwritten.")
                    try:
                        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    except Exception as error:
                        raise RuntimeError(f"Cannot read existing checkpoint: {checkpoint_path}. "
                                           "Refusing to overwrite it.") from error
                    saved_config = saved_checkpoint.get("extra_config") or {}
                    # Compare the results folder/task split location, not its machine-specific absolute prefix.
                    # The actual saved split assignments are validated above.
                    saved_split = str(saved_config.get("split_assignment_path", "")).replace("\\", "/")
                    current_split = split_assignment_path.replace("\\", "/")
                    mismatches = [key for key, value in expected_checkpoint_config.items()
                                  if key != "split_assignment_path" and saved_config.get(key) != value]
                    if saved_split.split("/")[-3:] != current_split.split("/")[-3:]:
                        mismatches.append("split_assignment_path")
                    if saved_checkpoint.get("model_state_dict") is None or saved_checkpoint.get("best_epoch", 0) <= 0:
                        mismatches.append("completed_checkpoint")
                    if mismatches:
                        raise RuntimeError(f"Existing checkpoint differs in {mismatches}: {checkpoint_path}. "
                                           "Refusing to overwrite it; use a different output folder for a new fit.")
                    checkpoint_matches = True

                manifest_row = {
                    "config_id": config_id,
                    "random_split": random_split,
                    "split_seed": split_seed,
                    "split_assignment_path": split_assignment_path,
                    "split_column": split_column,
                    "validation_fold": validation_fold,
                    "testing_fold": test_fold,
                    "task": task["name"],
                    "num_hidden_layers": config["num_hidden_layers"],
                    "rank": config["rank"],
                    "participant_emb_dim": config["participant_emb_dim"],
                    "mapping": config["mapping"],
                    "hyper_hidden_dim": config["hyper_hidden_dim"],
                    "hidden_dims": "-".join(str(width) for width in config["hidden_dims"]),
                    "hidden_dim": config["hidden_dims"][0],
                    "regularization": hyper_reg,
                    "training_fraction": 0.8,
                    "validation_fraction": 0.1,
                    "testing_fraction": 0.1,
                    "checkpoint_path": checkpoint_path,
                    "embedding_path": embedding_path,
                    "history_path": history_path,
                }
                manifest_rows.append(manifest_row)

                if checkpoint_matches:
                    pd.DataFrame(manifest_rows).drop_duplicates(["config_id", "random_split", "task"], keep="last").to_csv(manifest_path, index=False)
                    print(f"    fold rotation {random_split}: skipped completed {task['name']}")
                    continue

                loader_generator = torch.Generator().manual_seed(task_seed)
                train_loader = DataLoader(
                    BehavioralDataset(train_df, task["x_col"], task["y_col"]),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=behavioral_collate_fn,
                    generator=loader_generator,
                )
                val_loader = DataLoader(
                    BehavioralDataset(val_df, task["x_col"], task["y_col"]),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=behavioral_collate_fn,
                )

                model = HyperNN(
                    n_participants=task["data"]["participant_id"].nunique(),
                    input_dim=len(task["x_col"]),
                    hidden_dim=config["hidden_dims"],
                    output_dim=2,
                    participant_emb_dim=config["participant_emb_dim"],
                    hyper_hidden_dim=config["hyper_hidden_dim"] or 4,
                    rank=config["rank"],
                    shared_right=False,
                    nonlinearity="tanh",
                    mapping=config["mapping"],
                    num_hidden_layers=config["num_hidden_layers"],
                )

                model, history, _ = fit_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=n_epochs,
                    device=device,
                    save_path=checkpoint_path,
                    model_type="hyper",
                    lr=learning_rate,
                    patience=patience,
                    emb_reg=emb_reg,
                    hyper_reg=hyper_reg,
                    extra_config=expected_checkpoint_config,
                    verbose=False,
                )

                np.save(embedding_path, model.participant_embedding.weight.detach().cpu().numpy())
                pd.DataFrame(history).to_csv(history_path, index=False)
                pd.DataFrame(manifest_rows).drop_duplicates(["config_id", "random_split", "task"], keep="last").to_csv(manifest_path, index=False)
                print(f"    fold rotation {random_split}: saved {task['name']}")

    print(f"Training complete. Results saved to: {save_dir}")
