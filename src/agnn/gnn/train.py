"""
This is the training loop for the 2-branch GNN.

It trains a single fold to convergence with:
    - Class-weighted cross-entropy loss (this handles the 1.5:1 imbalance)
    - Adam optimiser with weight decay
    - Early stopping on validation AUC (which is the primary metric)
    - Model checkpointing (this saves the state that gave the best validation AUC)
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
    """
    Seed torch, numpy, and Python's random. This is for reporudcability purposes.

    This also disables cuDNN autotuning. This means that results are consistent across runs at the cost of a small dip in speed.
    This helps for CV comparison and hyperparameter analysis.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """
    This function runs one training epoch and then returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    # Move both PyG batches and the labels to the target device
    for delta_batch, alpha2_batch, labels in train_loader:
        delta_batch = delta_batch.to(device)
        alpha2_batch = alpha2_batch.to(device)
        labels = labels.to(device)

        # Standard forward, loss, backward, step
        optimizer.zero_grad()
        logits = model(delta_batch, alpha2_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Track the loss weighted by batch size. This helps with accurate averaging.
        batch_size = labels.size(0)
        total_loss += loss.item()*batch_size
        n_samples += batch_size

    return total_loss/n_samples


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    """
    This function computes validation metrics on a loader.

    It returns a dictionary with average loss, ROC-AUC, and per-sample predictions. These are used for downstream analysis 
    (like a confusion matrix or per-fold ROC curves).
    """
    # Set the model to eval mode and prepare the accumulators for the batch loop.
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    # No gradient trackingduring evaluation. This saves memory and speeds things up.
    with torch.no_grad():
        for delta_batch, alpha2_batch, labels in loader:
            delta_batch = delta_batch.to(device)
            alpha2_batch = alpha2_batch.to(device)
            labels_dev = labels.to(device)

            # Forward pass and loss on this batch
            logits = model(delta_batch, alpha2_batch)
            loss = criterion(logits, labels_dev)

            # Softmax over class dim.  Take positive-class probability for AUC
            probs = F.softmax(logits, dim=-1)[:, 1]

            # Accumulate lossweighted by batch size. Stash predictions for AUC
            batch_size = labels.size(0)
            total_loss += loss.item()*batch_size
            n_samples += batch_size
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    # Flatten all the batches into single arrays for the metrics
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # roc_auc_score needs both classes present. This is defense against edge cases
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)

    return {"loss": total_loss/n_samples, "auc": auc, "y_true": y_true, "y_prob": y_prob}


def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict, device: torch.device,
    checkpoint_path: Path | str | None = None, verbose: bool = True) -> dict[str, Any]:
    """
    This function trains a single fold to convergence and uses early stopping on validation AUC.

    Parameters
    ----------
    - model: nn.Module
          Freshly initialised model. Already moved to `device`
    - train_loader, val_loader: DataLoader
          Training and validation loaders for the fold
    - config: dict
          Training config with following keys:learning_rate, weight_decay, epochs, patience, class_weights. `class_weights` is 
          [w_non_carrier, w_carrier]. Since the 1.5:1 carrier to non-carrier imbalance, [1.5, 1.0] gives more weight to the 
          minority (non-carrier) class.
    - device: torch.device
          cuda or cpu
    - checkpoint_path: Path or None
          If provided, this saves the best model state to this path whenever validation AUC improves
    - verbose: bool
        If set to True it will print per-epoch metrics

    Returns
    -------
    - A dictionary with the folloing keys:
          history: {train_loss: list, val_loss: list, val_auc: list}
          best_auc: float
          best_epoch: integer (1-indexed)
          best_state: state_dict of the best-AUC model
          best_metrics: full evaluation dict at the best epoch
    """
    # Adam is the standard choice for GNN training so go with this. Weight decay provides L2 regularisation. This is important 
    # given n=77
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Class-weighted cross entropy handles the 1.5:1 imbalance without needing data level resampling so use this
    class_weights = torch.tensor(config["class_weights"], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Store per-epoch metrics. This is for plotting the training curves later
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    # These track the states that are the best so far, as well as how long since improvement for early stopping.
    best_auc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    best_metrics: dict | None = None
    epochs_without_improvement = 0

    # This is the main training loop. The epoch has been 1-indexed because it creates readable logs
    for epoch in range(1, config["epochs"]+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Record this epoch's metrics for training curves.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        # Primt progress (if requested)
        if verbose:
            print(f"Epoch {epoch:3d}/{config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f}")

        # Treat NaN AUC as no improvement. This is defense against single-class folds
        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch

            # Clone to CPU. This way future model updates don't overwrite saved state
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            epochs_without_improvement = 0

            # Perssist best model to disk if a path was provided
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Early stopping if validation AUC hasn't improved for `patience` number of epochs
        if epochs_without_improvement >= config["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch}: no improvement for {config['patience']} epochs (best AUC={best_auc:.4f} at epoch {best_epoch}).")
            break

    # return the dictionary of information about training the fold
    return {"history": history, "best_auc": best_auc, "best_epoch": best_epoch, "best_state": best_state, "best_metrics": best_metrics}