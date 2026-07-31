"""
This file handles data loading and cross-validation splitting for the 2-branch GNN.

Each training sample is a pair (delta_graph, alpha2_graph, label), where both graphs are from the same (subject, epoch). 
The pair is what the model will consume as a single training example.

Cross-validation uses StratifiedGroupKFold with:
- Groups as the subject IDs (stops subject leakage per Brookshire et al. 2024)
- Stratified on labels (this is to makie sure there is balanced carrier/non-carrier ratios per fold)

Needs scikit-learn.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

# Type alias for readability. A training sample is a delta graph, its matching alpha-2 graph from the same subject-epoch, and 
# the shared APOE label.
GraphPair = tuple[Data, Data, int]


def load_subject_graphs(subject_ids: Iterable[str], graphs_root: Path) -> list[Data]:
    """
    This function loads all graph Data objects for the given subjects from disk.

    Parameters
    ----------
    - subject_ids: list of strings
          BIDS subject IDs to load
    - graphs_root: Path
          Root directory containing per subject graph files

    Returns
    -------
    - graphs: list of Data
          Flat list of all Data objects across all subjects, both bands
    """
    all_graphs: list[Data] = []
    for sub in subject_ids:
        path = Path(graphs_root)/sub/f"{sub}_task-rest_graphs.pt"
        subject_graphs = torch.load(path, weights_only=False)
        all_graphs.extend(subject_graphs)
    return all_graphs


def pair_by_epoch(graphs: Iterable[Data]) -> list[GraphPair]:
    """
    This function groups delta and alpha-2 graphs by (subject, epoch) into training pairs.

    Each output tuple is (delta_graph, alpha2_graph, label). Epochs missing either band are skipped (this shouldn't happen with a
    healthy pipeline).

    Parameters
    ----------
    - graphs: list of Data
          Flat list of Data objects with metadata subject_id, epoch_idx, band, y

    Returns
    -------
    - pairs: list of (Data, Data, int)
    """
    # Group graphs by (subject, epoch) so each key holds both bands graphs
    by_key: dict[tuple[str, int], dict[str, Data]] = defaultdict(dict)
    for g in graphs:
        key = (g.subject_id, g.epoch_idx)
        by_key[key][g.band] = g

    pairs: list[GraphPair] = []
    for band_graphs in by_key.values():
        # Only keep epochs where both bands are present
        if "delta" not in band_graphs or "alpha2" not in band_graphs:
            continue

        delta = band_graphs["delta"]
        alpha2 = band_graphs["alpha2"]

        # Both graphs of a pair should share the same subject level label
        if not hasattr(delta, "y"):
            raise ValueError(f"Graph {delta.subject_id}/{delta.epoch_idx} has no y (APOE label).")
        label = int(delta.y.item())
        pairs.append((delta, alpha2, label))

    return pairs


class GraphPairDataset(Dataset):
    """
    PyTorch dataset wrapper. List of (delta, alpha2, label) tuples
    """

    def __init__(self, pairs: list[GraphPair]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> GraphPair:
        return self.pairs[idx]


def collate_pairs(batch: list[GraphPair]) -> tuple[Batch, Batch, torch.Tensor]:
    """
    This function collates a list of pairs into two PyG Batches plus a label tensor.

    The two returned Batches have 1:1 correspondence. The i'th delta graph in delta_batch matches the i'th alpha-2 graph in 
    alpha2_batch.

    Parameters
    ----------
    - batch: list of (Data, Data, int)

    Returns
    -------
    - delta_batch: Batch
    - alpha2_batch: Batch
    - labels: LongTensor, shape (batch_size,)
    """
    delta_graphs = [item[0] for item in batch]
    alpha2_graphs = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return (Batch.from_data_list(delta_graphs), Batch.from_data_list(alpha2_graphs), labels)


def make_cv_splits(pairs: list[GraphPair], n_folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    This function creates subject level stratified cross-validation splits.

    Uses StratifiedGroupKFold. This ensures that:
        - No subject appears in both train and test 
        - Each fold has similar carrier/non-carrier ratios 

    Parameters
    ----------
    - pairs: list of (Data, Data, int)
    - n_folds: int
          Number of CV folds. Default 5
    - seed: int
          Random seed for reproducibility

    Returns
    -------
    - splits: list of (train_idx, test_idx) arrays
          One tuple per fold. Indices are positions in the pairs list
    """
    subjects = np.array([p[0].subject_id for p in pairs])
    labels = np.array([p[2] for p in pairs])

    # X isn't used by StratifiedGroupKFold apart from determining n_samples.
    dummy_X = np.zeros(len(pairs))

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(dummy_X, labels, groups=subjects))


def verify_no_subject_leakage(pairs: list[GraphPair], train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """
    This function asserts that no subject appears in both train and test.

    This is a check for the Brookshire et al. (2024) leakage risk. Call this from the training loop before each fold. This way
    splitter misuse is caught early

    Raises
    ------
    - AssertionError
          If any subject appears in both train and test indices
    """
    train_subjects = {pairs[i][0].subject_id for i in train_idx}
    test_subjects = {pairs[i][0].subject_id for i in test_idx}
    overlap = train_subjects & test_subjects
    if overlap:
        raise AssertionError(f"Subject leakage detected: {overlap}")


def make_dataloader(pairs: list[GraphPair], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Thid function creates a DataLoader for a subset of pairs.

    Parameters
    ----------
    - pairs: list of (Data, Data, int)
          Full pairs list
    - indices: array of int
          Indices into pairs to include in this loader
    - batch_size: int
          Number of pairs per batch
    - shuffle: bool
          Generally True for train, False for validation

    Returns
    -------
    - loader: DataLoader
    """
    subset = [pairs[i] for i in indices]
    dataset = GraphPairDataset(subset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pairs)