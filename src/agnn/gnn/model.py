"""
Two-branch Graph Convolutional Network for APOE e4 classification.

Architecture:

- Delta graph -> GCN branch (2 layers) -> 32-dim embedding 
                                                             | -> Concat -> MLP -> APOE
- Alpha-2 graph -> GCN branch (2 layers) -> 32-dim embedding 

Each branch processes one frequency band's graph independently. Node features are sliced per branch so each branch only sees 
its own band's power values. There should be no cross-band info at the node feature level. Cross-band interactions are learned 
by the MLP classifier from the concatenated branch embeddings.

Layer choices are made in line with Klepl et al.'s 2022 and 2023 paper:
    - GCNConv with edge weights
    - ReLU activation
    - Global mean pooling
    - Dropout between GCN layers and in the classifier head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch


class BandBranch(nn.Module):
    """
    This is the single band GCN branch. There are two GCN layers followed by graph-level pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Two Kipf-Welling GCN layers. 
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # First GCN layer with ReLU and dropout for regularisation.
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer with ReLU.
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Collapse per-node embeddings to a single graph-level embedding uisng mean pooling over each graph in the batch.
        return global_mean_pool(x, batch)


class MultiBandGCN(nn.Module):
    """
    Two-branch GCN with MLP classifier for APOE e4 carrier prediction.

    Parameters
    ----------
    - hidden_channels: int
          GCN hidden layer dimension
    - embedding_dim: int
          Branch output dimension (per-branch graph embedding size)
    - mlp_hidden: int
          MLP classifier hidden layer size
    - dropout: float
          Dropout probability, applied between GCN layers and in the classifier
    - n_classes: int
          Number of output classes (2 for binary APOE e4 classification)
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        mlp_hidden: int = 32,
        dropout: float = 0.5,
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        # One branch per frequency band. Each takes 1-dim node features (that band's relative power) and produces a fixed-size 
        # graph embedding.
        self.delta_branch = BandBranch(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            dropout=dropout,
        )
        self.alpha2_branch = BandBranch(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            dropout=dropout,
        )

        # Classifier head: concatenated branch embeddings -> hidden -> classes.
        concat_dim = 2*embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, delta_batch: Batch, alpha2_batch: Batch) -> torch.Tensor:
        """
        Performs a forward pass on paired delta and alpha-2 graph batches.

        Parameters
        ----------
        - delta_batch: torch_geometric.data.Batch
              Batched delta-band graphs.
        - alpha2_batch: torch_geometric.data.Batch
              Batched alpha-2 band graphs, paired 1:1 with delta_batch.

        Returns
        -------
        - logits: Tensor, shape (batch_size, n_classes)
              Unnormalised class scores. Apply softmax for probabilities.
        """
        # Slice node features per band. Each graph's x has shape (n_nodes, 2) where column 0 is delta relative power and column 1 is alpha-2.
        # Each branch sees only its own band's column
        delta_x = delta_batch.x[:, 0:1]
        alpha2_x = alpha2_batch.x[:, 1:2]

        # edge_attr has shape (n_edges, 1); squeeze to (n_edges,) for GCNConv
        delta_edge_weight = delta_batch.edge_attr.squeeze(-1)
        alpha2_edge_weight = alpha2_batch.edge_attr.squeeze(-1)

        # Run each band through its own branch
        delta_emb = self.delta_branch(
            delta_x,
            delta_batch.edge_index,
            delta_edge_weight,
            delta_batch.batch,
        )
        alpha2_emb = self.alpha2_branch(
            alpha2_x,
            alpha2_batch.edge_index,
            alpha2_edge_weight,
            alpha2_batch.batch,
        )

        # Concatenate branch embeddings and run through classifier
        combined = torch.cat([delta_emb, alpha2_emb], dim=-1)
        return self.classifier(combined)