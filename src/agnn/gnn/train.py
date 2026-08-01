"""
This is the training loop for the 2-branch GNN.

It trains a single fold to convergence with:
    - Class-weighted cross-entropy loss (to handle the 1.5:1 imbalance)
    - Adam optimiser with weight decay
    - Early stopping on validation AUC (the primary metric)
    - Model checkpointing (saves the state that gave the best validation AUC)
    - Deterministic seeding for reproducibility

Entry point is `train_fold`, other functions are helpers
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed torch, numpy, and Python's random for reproducibility.

    Also disables cuDNN autotuning so results are consistent across runs at
    the cost of a small speed penalty. Recommended for CV comparison and
    hyperparameter analysis.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns the average training loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for delta_batch, alpha2_batch, labels in train_loader:
        # Move both PyG batches and the labels to the target device.
        delta_batch = delta_batch.to(device)
        alpha2_batch = alpha2_batch.to(device)
        labels = labels.to(device)

        # Standard forward + loss + backward + step.
        optimizer.zero_grad()
        logits = model(delta_batch, alpha2_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Track loss weighted by batch size for accurate averaging.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Compute validation metrics on a loader.

    Returns a dict with average loss, ROC-AUC, and per-sample predictions
    for downstream analysis (e.g. confusion matrix, per-fold ROC curves).
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for delta_batch, alpha2_batch, labels in loader:
            delta_batch = delta_batch.to(device)
            alpha2_batch = alpha2_batch.to(device)
            labels_dev = labels.to(device)

            logits = model(delta_batch, alpha2_batch)
            loss = criterion(logits, labels_dev)

            # Softmax over class dim; take positive-class probability for AUC.
            probs = F.softmax(logits, dim=-1)[:, 1]

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # roc_auc_score requires both classes present. StratifiedGroupKFold
    # guarantees this in normal use, but defensive against edge cases.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "loss": total_loss / n_samples,
        "auc": auc,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_path: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a single fold to convergence with early stopping on val AUC.

    Parameters
    ----------
    model : nn.Module
        Freshly-initialised model, already moved to ``device``.
    train_loader, val_loader : DataLoader
        Training and validation loaders for this fold.
    config : dict
        Training config with keys:
            learning_rate, weight_decay, epochs, patience, class_weights.
        ``class_weights`` is [w_non_carrier, w_carrier]. Given the 1.5:1
        carrier:non-carrier imbalance, [1.5, 1.0] gives more weight to the
        minority (non-carrier) class.
    device : torch.device
        cuda or cpu.
    checkpoint_path : Path or None
        If provided, saves the best model state to this path whenever
        validation AUC improves.
    verbose : bool
        If True, prints per-epoch metrics.

    Returns
    -------
    dict with keys:
        history: {train_loss: list, val_loss: list, val_auc: list}
        best_auc: float
        best_epoch: int (1-indexed)
        best_state: state_dict of the best-AUC model
        best_metrics: full evaluation dict at the best epoch
    """
    # Adam is the standard choice for GNN training. Weight decay provides
    # L2 regularisation, important given N=77.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Class-weighted cross-entropy handles the 1.5:1 imbalance without
    # requiring data-level resampling.
    class_weights = torch.tensor(
        config["class_weights"], dtype=torch.float, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }
    best_auc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    best_metrics: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        if verbose:
            print(
                f"Epoch {epoch:3d}/{config['epochs']}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_auc={val_metrics['auc']:.4f}"
            )

        # Treat NaN AUC as no improvement (defensive against single-class folds).
        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            # Clone to CPU so future model updates don't overwrite the saved state.
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_metrics = val_metrics
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Early stopping if validation AUC hasn't improved for ``patience`` epochs.
        if epochs_without_improvement >= config["patience"]:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}: no improvement "
                    f"for {config['patience']} epochs (best AUC={best_auc:.4f} "
                    f"at epoch {best_epoch})."
                )
            break

    return {
        "history": history,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "best_metrics": best_metrics,
    }