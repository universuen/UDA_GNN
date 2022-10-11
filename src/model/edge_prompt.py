import torch
from torch import nn

from src.original.gnn_model import (
    num_bond_type,
    num_bond_direction,
)


class EdgePrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            in_features=num_bond_type + num_bond_direction,
            out_features=1,
            bias=False,
        )

    def forward(
            self,
            embeddings: torch.Tensor,
            weights: torch.Tensor,
    ):
        # the operation can be modified later
        return embeddings + self.linear(weights.T).T
