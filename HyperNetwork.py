import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from cca_zoo.linear import CCA, GCCA
from threadpoolctl import threadpool_limits
from CKA.CKA import CKA
from itertools import combinations


class HyperNN(nn.Module):
    """
    Participant-conditioned feedforward neural network in which a hypernetwork generates participant-specific updates
    to the input-to-hidden weight matrix.
    """

    def __init__(
        self,
        n_participants: int,
        input_dim: int,
        hidden_dim: int | list[int] | tuple[int, ...],
        output_dim: int,
        num_hidden_layers: int = 1,
        participant_emb_dim: int = 8,
        hyper_hidden_dim: int = 8,
        rank: int | None = None,
        alpha: float | None = None,
        shared_right: bool = False,
        nonlinearity: str = "tanh",
        mapping: str = "nonlinear",
    ):
        super().__init__()

        if not isinstance(num_hidden_layers, int) or num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * num_hidden_layers
        else:
            hidden_dims = list(hidden_dim)
            if len(hidden_dims) != num_hidden_layers:
                raise ValueError(
                    "hidden_dim must contain one width per hidden layer"
                )
        if not hidden_dims or any(
            not isinstance(width, int) or width < 1 for width in hidden_dims
        ):
            raise ValueError("all hidden dimensions must be positive integers")
        first_hidden_dim = hidden_dims[0]
        last_hidden_dim = hidden_dims[-1]

        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        if mapping not in {"nonlinear", "linear"}:
            raise ValueError("mapping must be 'nonlinear' or 'linear'")

        max_rank = min(first_hidden_dim, input_dim)
        if rank is not None and not 1 <= rank <= max_rank:
            raise ValueError(f"rank must be between 1 and {max_rank}, got {rank}")
        if rank is None and alpha is not None:
            raise ValueError("alpha is only used when rank is specified")
        if shared_right and rank is None:
            raise ValueError("shared_right=True requires rank to be specified")

        self.n_participants = n_participants
        self.input_dim = input_dim
        self.hidden_dim = first_hidden_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.participant_emb_dim = participant_emb_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.rank = rank
        self.alpha = rank if rank is not None and alpha is None else alpha
        self.shared_right = shared_right
        self.nonlinearity = nonlinearity
        self.mapping = mapping
        self.num_hidden_layers = num_hidden_layers

        self.participant_embedding = nn.Embedding(n_participants, participant_emb_dim)
        self.W_xh_base = nn.Parameter(torch.empty(first_hidden_dim, input_dim))
        self.b_h = nn.Parameter(torch.zeros(first_hidden_dim))
        self.W_hy = nn.Parameter(torch.empty(output_dim, last_hidden_dim))
        self.b_y = nn.Parameter(torch.zeros(output_dim))

        if rank is None:
            hyper_out_dim = first_hidden_dim * input_dim
        elif shared_right:
            hyper_out_dim = first_hidden_dim * rank
            self.B_shared = nn.Parameter(torch.empty(input_dim, rank))
        else:
            hyper_out_dim = (first_hidden_dim + input_dim) * rank

        if mapping == "linear":
            self.hypernet = nn.Linear(participant_emb_dim, hyper_out_dim)
        else:
            self.hypernet = nn.Sequential(
                nn.Linear(participant_emb_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, hyper_out_dim),
            )

        self.extra_hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[layer_index - 1], hidden_dims[layer_index])
            for layer_index in range(1, num_hidden_layers)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_xh_base)
        nn.init.zeros_(self.b_h)
        nn.init.xavier_uniform_(self.W_hy)
        nn.init.zeros_(self.b_y)
        for layer in self.extra_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.participant_embedding.weight, mean=0.0, std=0.02)

        if self.mapping == "nonlinear":
            nn.init.kaiming_uniform_(self.hypernet[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.hypernet[0].bias)
            output_layer = self.hypernet[2]
        else:
            output_layer = self.hypernet

        if self.rank is None:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        elif self.shared_right:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            nn.init.orthogonal_(self.B_shared)
        else:
            left_size = self.hidden_dim * self.rank
            nn.init.normal_(output_layer.weight[:left_size], mean=0.0, std=0.01)
            nn.init.zeros_(output_layer.weight[left_size:])
            nn.init.zeros_(output_layer.bias)

    def generate_delta_W_xh(self, participant_ids: torch.Tensor) -> torch.Tensor:
        batch_size = participant_ids.size(0)
        hyper_out = self.hypernet(self.participant_embedding(participant_ids))

        if self.rank is None:
            return hyper_out.view(batch_size, self.hidden_dim, self.input_dim)
        if self.shared_right:
            left_factor = hyper_out.view(batch_size, self.hidden_dim, self.rank)
            delta_w = torch.matmul(left_factor, self.B_shared.T)
            return (self.alpha / self.rank) * delta_w

        left_size = self.hidden_dim * self.rank
        left_factor = hyper_out[:, :left_size].view(batch_size, self.hidden_dim, self.rank)
        right_factor = hyper_out[:, left_size:].view(batch_size, self.input_dim, self.rank)
        delta_w = torch.bmm(left_factor, right_factor.transpose(1, 2))
        return (self.alpha / self.rank) * delta_w

    def forward(self, x: torch.Tensor, participant_ids: torch.Tensor):
        delta_W_xh = self.generate_delta_W_xh(participant_ids)
        W_xh = self.W_xh_base.unsqueeze(0) + delta_W_xh
        hidden = torch.bmm(x, W_xh.transpose(1, 2)) + self.b_h
        if self.nonlinearity == "tanh":
            hidden = torch.tanh(hidden)
        else:
            hidden = F.relu(hidden)
        for layer in self.extra_hidden_layers:
            hidden = layer(hidden)
            if self.nonlinearity == "tanh":
                hidden = torch.tanh(hidden)
            else:
                hidden = F.relu(hidden)
        logits = F.linear(hidden, self.W_hy, self.b_y)
        return logits, hidden

class BaselineNN(nn.Module):
    """HyperNN's feedforward backbone with shared weights and no participant embeddings."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1, nonlinearity="tanh"):
        super().__init__()

        if not isinstance(num_hidden_layers, int) or num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * num_hidden_layers
        else:
            hidden_dims = list(hidden_dim)
            if len(hidden_dims) != num_hidden_layers:
                raise ValueError(
                    "hidden_dim must contain one width per hidden layer"
                )
        if not hidden_dims or any(
            not isinstance(width, int) or width < 1 for width in hidden_dims
        ):
            raise ValueError("all hidden dimensions must be positive integers")
        first_hidden_dim = hidden_dims[0]
        last_hidden_dim = hidden_dims[-1]

        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        self.input_dim = input_dim
        self.hidden_dim = first_hidden_dim
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.W_xh_base = nn.Parameter(torch.empty(first_hidden_dim, input_dim))
        self.b_h = nn.Parameter(torch.zeros(first_hidden_dim))
        self.W_hy = nn.Parameter(torch.empty(output_dim, last_hidden_dim))
        self.b_y = nn.Parameter(torch.zeros(output_dim))
        self.extra_hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[layer_index - 1], hidden_dims[layer_index])
            for layer_index in range(1, num_hidden_layers)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_xh_base)
        nn.init.zeros_(self.b_h)
        nn.init.xavier_uniform_(self.W_hy)
        nn.init.zeros_(self.b_y)
        for layer in self.extra_hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, participant_ids=None):
        # Accept IDs for compatibility with the existing loaders; they are not used.
        hidden = F.linear(x, self.W_xh_base, self.b_h)
        hidden = torch.tanh(hidden) if self.nonlinearity == "tanh" else F.relu(hidden)
        for layer in self.extra_hidden_layers:
            hidden = layer(hidden)
            hidden = torch.tanh(hidden) if self.nonlinearity == "tanh" else F.relu(hidden)
        logits = F.linear(hidden, self.W_hy, self.b_y)
        return logits, hidden


def sequence_nll_loss(logits, targets, mask=None):
    """
    logits:  [B, T, C]
    targets: [B, T]
    mask:    [B, T] with 1 for valid choice trials, 0 for no-response/padded trials

    Returns mean cross-entropy over valid sequence positions only.
    """
    B, T, C = logits.shape

    targets = targets.long()

    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = mask.bool()

    # Valid class labels must be 0, 1, ..., C-1.
    valid_targets = (targets >= 0) & (targets < C)

    # Final mask ignores: no-response trials, padded trials, validation and test trials
    final_mask = mask & valid_targets

    # Replace invalid targets before loss.
    targets_safe = targets.clone()
    targets_safe[~final_mask] = -100

    logits_flat = logits.reshape(B * T, C)
    targets_flat = targets_safe.reshape(B * T)

    loss_per_item = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction="none")
    final_mask_flat = final_mask.reshape(B * T).float()

    return (loss_per_item * final_mask_flat).sum() / final_mask_flat.sum().clamp_min(1.0)


def sequence_accuracy(logits, targets, mask=None):
    """
    logits:  [B, T, C]
    targets: [B, T]
    mask:    [B, T]
    """
    B, T, C = logits.shape

    targets = targets.long()

    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = mask.bool()

    valid_targets = (targets >= 0) & (targets < C)
    final_mask = mask & valid_targets

    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & final_mask

    return (correct.sum().float() / final_mask.sum().clamp_min(1).float()).item()


def train_one_epoch(model, dataloader, optimizer, device, model_type="baseline",
                    emb_reg=1e-4, hyper_reg=1e-5, max_grad_norm=1.0):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch['x'].to(device)                         # [B, T, D]
        y = batch['y'].to(device)                         # [B, T]
        participant_ids = batch['participant_id'].to(device)  # [B]
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        optimizer.zero_grad()

        logits, _ = model(x, participant_ids)
        loss = sequence_nll_loss(logits, y, mask)

        # extra regularization only for hypernetwork model
        if model_type == "hyper":
            emb = model.participant_embedding(participant_ids)
            loss = loss + emb_reg * emb.pow(2).mean()

            if hasattr(model, "generate_delta_W_xh"):
                delta_W = model.generate_delta_W_xh(participant_ids)
            else:
                delta_W = model.generate_delta_W_hh(participant_ids)
            loss = loss + hyper_reg * delta_W.pow(2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        acc = sequence_accuracy(logits, y, mask)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, prevalence_rate=None):
    """Return trial-weighted metrics over valid trials.

    prevalence_rate: optional scalar P(y=1), between 0 and 1. When supplied,
    prevalence_distance is the mean absolute probability deviation from it.
    """
    if prevalence_rate is not None:
        prevalence_rate = float(prevalence_rate)
        if not 0 <= prevalence_rate <= 1:
            raise ValueError("prevalence_rate must be a finite probability between 0 and 1")
    model.eval()

    total_nll = 0.0
    total_correct = 0
    total_valid = 0
    total_predicted_1 = 0
    probabilities = []

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        participant_ids = batch["participant_id"].to(device)
        mask = batch.get("mask", None)
        if mask is None:
            mask = torch.ones_like(y, dtype=torch.bool)
        else:
            mask = mask.to(device).bool()

        logits, _ = model(x, participant_ids)
        valid = mask & (y >= 0) & (y < logits.size(-1))

        if valid.any():
            total_nll += F.cross_entropy(logits[valid], y[valid], reduction="sum").item()
            predictions = logits.argmax(dim=-1)[valid]
            total_correct += (predictions == y[valid]).sum().item()
            total_valid += valid.sum().item()
            total_predicted_1 += (predictions == 1).sum().item()
            probabilities.append(logits[valid].softmax(dim=-1)[:, 1].detach().cpu())

    pred_fraction_1 = (total_predicted_1 / total_valid if total_valid > 0 else float("nan"))
    probabilities = torch.cat(probabilities).double() if probabilities else torch.empty(0)
    pred_probability_1_mean = probabilities.mean().item() if total_valid else float("nan")
    pred_probability_1_std = probabilities.std(unbiased=False).item() if total_valid else float("nan")

    metrics = {
        "raw_loss": total_nll / max(total_valid, 1),
        "acc": total_correct / max(total_valid, 1),
        "n_trials": total_valid,
        "pred_fraction_1": pred_fraction_1,
        "pred_probability_1_mean": pred_probability_1_mean,
        "pred_probability_1_std": pred_probability_1_std,
        "pred_imbalance": max(pred_fraction_1, 1 - pred_fraction_1),
    }
    if prevalence_rate is not None:
        metrics["prevalence_distance"] = ((probabilities - prevalence_rate).abs().mean().item()
                                          if total_valid else float("nan"))
    return metrics


@torch.no_grad()
def evaluate_participants(model, dataloader, device):
    model.eval()
    records = []

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        participant_ids = batch['participant_id'].to(device)
        mask = batch['mask'].to(device).bool()

        logits, _ = model(x, participant_ids)
        valid = mask & (y >= 0) & (y < logits.size(-1))
        log_probs = logits.log_softmax(dim=-1)

        for i, participant_id in enumerate(participant_ids):
            selected = valid[i]
            n_trials = int(selected.sum().item())
            if n_trials == 0:
                continue

            observed = y[i, selected]
            participant_log_probs = log_probs[i, selected]
            trial_nll = -participant_log_probs.gather(
                1, observed[:, None]
            ).squeeze(1)

            predicted = participant_log_probs.argmax(dim=-1)
            n_correct = int((predicted == observed).sum().item())

            records.append({
                'participant_id': int(participant_id.item()),
                'total_nll': trial_nll.sum().item(),
                'n_trials': n_trials,
                'n_correct': n_correct,
            })

    results = pd.DataFrame(
        records,
        columns=['participant_id', 'total_nll', 'n_trials', 'n_correct']
    )
    results = results.groupby('participant_id', as_index=False).sum()
    results['test_nll'] = results.total_nll / results.n_trials
    results['test_accuracy'] = results.n_correct / results.n_trials
    return results


def fit_model(
    model,
    train_loader,
    val_loader,
    n_epochs,
    device,
    save_path,
    model_type="baseline",
    lr=1e-3,
    emb_reg=1e-4,
    hyper_reg=1e-5,
    max_grad_norm=1.0,
    patience=10,
    min_delta=1e-4,
    extra_config=None,
    verbose=True,
):
    """
    Train one model, keep the best validation checkpoint, apply early stopping,
    and save the best checkpoint.

    Early stopping rule:
        stop if validation loss does not improve by at least `min_delta`
        for `patience` consecutive epochs.

    Returns:
        model, history, best_val_loss
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            emb_reg=emb_reg,
            hyper_reg=hyper_reg,
            max_grad_norm=max_grad_norm,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["raw_loss"],
            "val_acc": val_metrics["acc"],
        })

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f} | "
                f"val loss={val_metrics['raw_loss']:.4f}, acc={val_metrics['acc']:.4f}"
            )

        # Improvement means val loss decreased by at least min_delta
        if val_metrics["raw_loss"] < (best_val_loss - min_delta):
            best_val_loss = val_metrics["raw_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f'Early stopping triggered at epoch {epoch}. '
                      f'Best epoch was {best_epoch} with val loss {best_val_loss:.4f}.')
            break

    # restore best checkpoint into model
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": best_state,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "model_type": model_type,
        "patience": patience,
        "min_delta": min_delta,
        "extra_config": extra_config,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    if verbose:
        print(f"Saved best checkpoint to: {save_path}")

    return model, history, best_val_loss


def load_checkpoint(model, checkpoint_path, device):
    """
    Load a saved checkpoint into an already-instantiated model.
    Returns:
        model, checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return model, checkpoint


@torch.no_grad()
def test_model(model, test_loader, device, verbose=False, prevalence_rate=None):
    """
    Evaluate a trained model on the test set.
    """
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        prevalence_rate=prevalence_rate
    )

    if verbose:
        print(
            f"Test loss={test_metrics['raw_loss']:.4f}, "
            f"test acc={test_metrics['acc']:.4f}"
        )
    return test_metrics


class BehavioralDataset(Dataset):
    def __init__(self, dataframe, x_var, y_var):
        self.data = []

        for p_id, group in dataframe.groupby("participant_id"):
            group = group.sort_values("trial").reset_index(drop=True)

            x_vals = group[x_var].values.astype("float32")
            y_vals = group[y_var].values.astype("int64")

            # Set up the mask
            if "loss_mask" in group.columns:
                mask_vals = group["loss_mask"].values.astype("bool")
            elif "choice_mask" in group.columns:
                mask_vals = group["choice_mask"].values.astype("bool")
            else:
                mask_vals = np.ones(len(group), dtype=bool)

            self.data.append({
                "x": torch.tensor(x_vals, dtype=torch.float32),
                "y": torch.tensor(y_vals, dtype=torch.long),
                "mask": torch.tensor(mask_vals, dtype=torch.bool),
                "participant_id": torch.tensor(p_id, dtype=torch.long),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def split_by_participant_trials(df, train_prop=0.70, val_prop=0.15):
    train_list, val_list, test_list = [], [], []

    for pid, group in df.groupby("participant_id"):
        group = group.sort_values("trial").reset_index(drop=True)

        n = len(group)
        train_end = int(n * train_prop)
        val_end = int(n * (train_prop + val_prop))

        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])

    return (
        pd.concat(train_list, ignore_index=True),
        pd.concat(val_list, ignore_index=True),
        pd.concat(test_list, ignore_index=True),
    )


def make_full_sequence_split_df(df, test_fold, val_fold, split):
    """
    Keep every participant's trials together while selecting which trials
    contribute to the loss. This works for both recurrent and feedforward models.
    """
    df = df.copy()
    df = df.sort_values(["participant_id", "trial"]).reset_index(drop=True)

    if split == "train":
        split_mask = df["fold"] != val_fold
        if test_fold is not None:
            split_mask &= df["fold"] != test_fold
    elif split == "val":
        split_mask = df["fold"] == val_fold
    elif test_fold is not None and split == "test":
        split_mask = df["fold"] == test_fold
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

    # Only valid-choice trials in the requested fold split contribute to loss.
    if "choice_mask" in df.columns:
        df["loss_mask"] = split_mask & df["choice_mask"].astype(bool)
    else:
        df["loss_mask"] = split_mask

    return df

def behavioral_collate_fn(batch):
    x_list = [item["x"] for item in batch]
    y_list = [item["y"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    participant_ids = torch.stack([item["participant_id"] for item in batch])

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=-1)

    # Padded trials should not count in loss
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0)

    return {
        "x": x_padded,                      # [B, T_max, D]
        "y": y_padded,                      # [B, T_max]
        "participant_id": participant_ids,  # [B]
        "mask": mask_padded.bool()          # [B, T_max]
    }


def collect_predictions(model, loader, device, threshold=0.5):
    model.eval()
    ys, probs = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pid = batch["participant_id"].to(device)
            mask = batch["mask"].to(device)

            logits, _ = model(x, pid)
            prob_correct = torch.softmax(logits, dim=-1)[..., 1]

            ys.append(y[mask].cpu())
            probs.append(prob_correct[mask].cpu())

    y_true = torch.cat(ys).numpy()
    p_correct = torch.cat(probs).numpy()
    y_pred = (p_correct >= threshold).astype(int)

    return y_true, y_pred, p_correct


def add_interspersed_folds(df, n_folds=5, seed=123):
    df = df.copy()
    df["fold"] = -1

    rng = np.random.default_rng(seed)

    for pid, sub_df in df.groupby("participant_id"):
        idx = sub_df.index.to_numpy().copy()
        rng.shuffle(idx)

        fold_ids = np.arange(len(idx)) % n_folds
        rng.shuffle(fold_ids)

        df.loc[idx, "fold"] = fold_ids

    return df



def add_temporal_folds(df, n_folds=5):
    df = (df.sort_values(['participant_id', 'trial']).reset_index(drop=True).copy())
    df['fold'] = -1

    for participant_id, participant_df in df.groupby('participant_id', sort=False):
        participant_indices = participant_df.index.to_numpy()
        fold_indices = np.array_split(participant_indices, n_folds,)
        for fold, indices in enumerate(fold_indices):
            df.loc[indices, "fold"] = fold

    df["fold"] = df["fold"].astype(int)

    return df


def preprocess_task(df, config, participant_col="worker_id", task_name=None):

    df = df.copy()
    # ------------------------------------------------------------------
    # Participant and trial indexing
    # ------------------------------------------------------------------
    df["participant_id"] = (df[participant_col].astype("category").cat.codes)
    df["trial"] = df.groupby("participant_id").cumcount() + 1
    df = (df.sort_values(["participant_id", "trial"]).reset_index(drop=True))

    # ------------------------------------------------------------------
    # Predictor definitions
    # ------------------------------------------------------------------
    categorical_cols = config.get("categorical_x", [])
    continuous_cols = config.get("continuous_x", [])
    y_col = config["y_col"]

    # ------------------------------------------------------------------
    # Scale continuous predictors
    # ------------------------------------------------------------------
    if continuous_cols:
        scaler = MinMaxScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols]).astype(np.float32)

    # ------------------------------------------------------------------
    # One-hot encode all categorical predictors
    # ------------------------------------------------------------------
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=np.float32)
        categorical_encoded_cols = [col for col in df.columns if any(col.startswith(f"{cat}_") for cat in categorical_cols)]
    else:
        categorical_encoded_cols = []

    # ------------------------------------------------------------------
    # Final model input columns
    # ------------------------------------------------------------------
    x_cols = continuous_cols + categorical_encoded_cols
    df[x_cols] = df[x_cols].astype(np.float32)

    # ------------------------------------------------------------------
    # Outcome transformation
    # ------------------------------------------------------------------
    y_transform = config.get("y_transform", "int")

    if y_transform == "int":
        df[y_col] = df[y_col].astype(int)

    elif y_transform == "category":
        df[y_col] = (df[y_col].astype("category").cat.codes)

    elif y_transform == "category_minus_1":
        df[y_col] = (df[y_col].astype("category").cat.codes - 1)

    else:
        raise ValueError(f"Unknown y_transform: {y_transform}")

    # ------------------------------------------------------------------
    # Response mask
    # ------------------------------------------------------------------
    response_mask = config.get('mask_col', 'true')

    if response_mask == 'true':
        df['choice_mask'] = (df['responded'].fillna(0).astype(int))
    elif response_mask == 'false':
        df['choice_mask'] = 1
    else:
        raise ValueError(f"Unknown mask_col: {response_mask}")

    # ------------------------------------------------------------------
    # Baseline statistics
    # Only use responded trials with valid binary outcomes
    # ------------------------------------------------------------------
    valid = ((df['choice_mask'] == 1) & (df[y_col] >= 0))
    y_valid = df.loc[valid, y_col]

    # This baseline calculation assumes binary outcomes
    unique_y = sorted(y_valid.unique())

    if set(unique_y).issubset({0, 1}):
        baseline_acc = y_valid.mean()

        eps = 1e-8
        p = np.clip(baseline_acc, eps, 1 - eps)
        baseline_nll = -(y_valid * np.log(p) + (1 - y_valid) * np.log(1 - p)).mean()

        if task_name is not None:
            print(f"Processing task: {task_name}")
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print(f"Baseline NLL: {baseline_nll:.4f}")
        print(f'={"="*40}')

    else:
        print(
            "Baseline accuracy/NLL skipped because "
            f"{y_col} is not binary. Classes: {unique_y}"
        )

    return df, x_cols, y_col


def evaluate_permutation(model_type, params, views, orders, cka_views=None):
    # One numerical-library thread per worker, including when n_jobs=1.
    with threadpool_limits(limits=1):
        shuffled = [v[order] for v, order in zip(views, orders)]
        model = {'CCA': CCA, 'GCCA': GCCA}[model_type](**params).fit(shuffled)
        mean_pairwise_r = model.score(shuffled)
        cka = None
        if cka_views is not None:
            raw = [v[order] for v, order in zip(cka_views, orders)]
            metric = CKA()
            cka = np.array([metric.linear_CKA(raw[a], raw[b]) for a, b in combinations(range(len(raw)), 2)])
        if not np.isfinite(mean_pairwise_r).all() or (cka is not None and not np.isfinite(cka).all()):
            raise ValueError('Non-finite permutation statistic')
        return mean_pairwise_r, cka

# ======================================================================================================================
# Unused
# ======================================================================================================================
class HyperRNN(nn.Module):
    """
    Participant-conditioned RNN in which a hypernetwork generates participant-specific
    updates to the recurrent hidden-to-hidden weight matrix.

    For participant i, the recurrent matrix is

        W_hh(i) = W_hh_base + delta_W_hh(i)

    where W_hh_base is shared across participants and delta_W_hh(i) is generated from
    a learned participant embedding.

    Three update parameterization are supported:

    1. Full-rank update (rank=None):
           delta_W_hh(i) = HyperNet(z_i)

    2. Participant-specific low-rank update (rank=r, shared_right=False):
           delta_W_hh(i) = (alpha / r) * A(z_i) @ B(z_i).T

       where both low-rank factors are generated from the participant embedding.

    3. Shared-basis low-rank update (rank=r, shared_right=True):
           delta_W_hh(i) = (alpha / r) * A(z_i) @ B_shared.T

       where A is participant-specific and B_shared is a trainable factor shared
       across all participants.

    The input-to-hidden weights, base recurrent matrix, recurrent biases, and output
    layer are shared across participants. Participant-specific variation is therefore
    restricted to the recurrent dynamics induced by delta_W_hh.

    Low-rank models satisfy rank(delta_W_hh) <= r. The default alpha is r, yielding
    a scaling factor alpha / r = 1.

    All variants are initialized so that delta_W_hh = 0 at initialization, causing
    the model to begin as a shared RNN before learning participant-specific recurrent
    modifications.

    Parameters
    ----------
    n_participants : int
        Number of unique participants.
    input_dim : int
        Number of input features at each time step.
    hidden_dim : int
        Size of the recurrent hidden state.
    output_dim : int
        Number of output logits at each time step.
    participant_emb_dim : int, default=8
        Dimensionality of the learned participant embeddings.
    hyper_hidden_dim : int, default=8
        Hidden-layer size of the hypernetwork when mapping="nonlinear". It is
        unused when mapping="linear".
    rank : int or None, default=None
        Rank of the participant-specific recurrent update. If None, a full-rank
        update is generated.
    alpha : float or None, default=None
        Scaling parameter for low-rank updates. If None and rank is specified,
        alpha defaults to rank.
    shared_right : bool, default=False
        If True, use a trainable right factor shared across participants and generate
        only the left factor from the participant embedding. Requires rank to be set.
    nonlinearity : {"tanh", "relu"}, default="tanh"
        Recurrent activation function.
    mapping : {"nonlinear", "linear"}, default="nonlinear"
        Mapping from participant embeddings to the generated recurrent update.
        "nonlinear" uses Linear -> ReLU -> Linear. "linear" uses one Linear
        layer from the participant embedding directly to the update parameters.
    """
    def __init__(
        self,
        n_participants: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        participant_emb_dim: int = 8,
        hyper_hidden_dim: int = 8,
        rank: int | None = None,
        alpha: float | None = None,
        shared_right: bool = False,
        nonlinearity: str = "tanh",
        mapping: str = "nonlinear",
    ):
        super().__init__()

        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        if mapping not in {"nonlinear", "linear"}:
            raise ValueError("mapping must be 'nonlinear' or 'linear'")

        self.n_participants = n_participants
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.participant_emb_dim = participant_emb_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.rank = rank
        self.nonlinearity = nonlinearity
        self.mapping = mapping
        self.alpha = alpha
        self.shared_right = shared_right

        if rank is not None and not 1 <= rank <= hidden_dim:
            raise ValueError(f"rank must be between 1 and hidden_dim={hidden_dim}, got {rank}")

        if rank is not None:
            self.alpha = rank if alpha is None else alpha
        else:
            self.alpha = None

        if rank is None and alpha is not None:
            raise ValueError("alpha is only used when rank is specified")

        if shared_right and rank is None:
            raise ValueError("shared_right=True requires rank to be specified")

        self.participant_embedding = nn.Embedding(n_participants, participant_emb_dim)

        # shared parameters
        self.b_x = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xh = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.W_hh_base = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        self.W_hy = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.b_y = nn.Parameter(torch.zeros(output_dim))

        if rank is None:
            hyper_out_dim = hidden_dim * hidden_dim

        elif shared_right:
            hyper_out_dim = hidden_dim * rank
            self.B_shared = nn.Parameter(
                torch.empty(hidden_dim, rank)
            )

        else:
            hyper_out_dim = 2 * hidden_dim * rank

        if mapping == "linear":
            self.hypernet = nn.Linear(participant_emb_dim, hyper_out_dim)
        else:
            self.hypernet = nn.Sequential(
                nn.Linear(participant_emb_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, hyper_out_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.b_x)
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh_base)
        nn.init.zeros_(self.b_h)

        nn.init.xavier_uniform_(self.W_hy)
        nn.init.zeros_(self.b_y)

        nn.init.normal_(self.participant_embedding.weight, mean=0.0, std=0.02)
        if self.mapping == "nonlinear":
            nn.init.kaiming_uniform_(self.hypernet[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.hypernet[0].bias)
            output_layer = self.hypernet[2]
        else:
            output_layer = self.hypernet

        if self.rank is None:
            # Full-rank generator starts with no participant-specific perturbation.
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

        elif self.shared_right:
            # A starts at zero; B_shared provides nonzero gradient to A.
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            nn.init.orthogonal_(self.B_shared)

        else:
            # A starts nonzero and B starts at zero.
            # Therefore delta_W = A @ B.T = 0 initially,
            # while B still receives gradients immediately.
            factor_size = self.hidden_dim * self.rank

            nn.init.normal_(output_layer.weight[:factor_size], mean=0.0, std=0.01)
            nn.init.zeros_(output_layer.weight[factor_size:])
            nn.init.zeros_(output_layer.bias)

    def generate_delta_W_hh(self, participant_ids: torch.Tensor) -> torch.Tensor:
        batch_size = participant_ids.size(0)
        z = self.participant_embedding(participant_ids)              # [B, E]
        hyper_out = self.hypernet(z)

        # Low-rank participant update.
        # If shared_right=False:
        #     delta_W(i) = A(z_i) @ B(z_i).T
        # If shared_right=True:
        #     delta_W(i) = A(z_i) @ B_shared.T

        if self.rank is None:
            return hyper_out.view(batch_size, self.hidden_dim, self.hidden_dim)
        elif self.shared_right:
            left_factor = hyper_out.view(batch_size, self.hidden_dim, self.rank)
            delta_w = torch.matmul(left_factor, self.B_shared.T)
            return (self.alpha / self.rank) * delta_w
        else:
            factor_size = self.hidden_dim * self.rank
            left_factor = hyper_out[:, :factor_size].view(
                batch_size, self.hidden_dim, self.rank
            )
            right_factor = hyper_out[:, factor_size:].view(
                batch_size, self.hidden_dim, self.rank
            )
            delta_w = torch.bmm(left_factor, right_factor.transpose(1, 2))  # [B, H, H]
            return (self.alpha / self.rank) * delta_w

    def recurrent_nonlinearity(self, x):
        if self.nonlinearity == "tanh":
            return torch.tanh(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor, participant_ids: torch.Tensor, h0=None):
        """
        x: [B, T, D]
        participant_ids: [B]
        h0: [B, H] or None
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h_t = h0

        delta_W_hh = self.generate_delta_W_hh(participant_ids)      # [B, H, H]
        W_hh = self.W_hh_base.unsqueeze(0) + delta_W_hh             # [B, H, H]

        logits_all = []

        for t in range(seq_len):
            x_t = x[:, t, :]                                        # [B, D]
            x_proj = F.linear(x_t, self.W_xh, self.b_x)                       # [B, H]
            h_proj = torch.bmm(W_hh, h_t.unsqueeze(-1)).squeeze(-1) # [B, H]

            h_t = self.recurrent_nonlinearity(x_proj + h_proj + self.b_h)

            logits_t = F.linear(h_t, self.W_hy, self.b_y)           # [B, C]
            logits_all.append(logits_t.unsqueeze(1))

        logits_all = torch.cat(logits_all, dim=1)                   # [B, T, C]
        return logits_all, h_t

class BaselineRNN(nn.Module):
    """
    Standard shared RNN for all participants.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=1, nonlinearity="tanh"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            nonlinearity=nonlinearity,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, participant_ids=None, h0=None):
        """
        x: [B, T, D]
        participant_ids: ignored, kept only for interface compatibility
        h0: optional initial hidden state [1, B, H]
        """
        h_all, h_n = self.rnn(x, h0)          # h_all: [B, T, H]
        logits = self.output(h_all)           # [B, T, C]
        return logits, h_n


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, participant_ids=None):
        """
        x: [B, T, D]
        participant_ids: ignored, included only for interface compatibility
        """
        h, _ = self.lstm(x)                  # [B, T, H]
        logits = self.output(h)             # [B, T, C]
        return logits, None
