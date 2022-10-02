import torch
from torch import nn
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.data import Data, Batch

import src
from src import config
from src import types


class BarlowTwins(types.PretrainingModel):
    def __init__(
            self,
            model: types.GNNModel | None = None,
            lambda_: float = config.BarlowTwins.lambda_,
            sizes: tuple[int] = config.BarlowTwins.sizes,
    ):
        if model is None:
            model = src.model.gnn.EqvGNN()
        super().__init__()
        self.logger.debug(f'model: {model}')
        self.logger.debug(f'lambda_: {lambda_}')
        self.logger.debug(f'sizes: {sizes}')

        self.lambda_ = lambda_
        self.gnn = model
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.pool = global_mean_pool

    def _gnn_forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        return x

    def forward(self, b_1: Data | Batch, b_2: Data | Batch) -> torch.Tensor:
        # batch_size = data1.x.shape[0]
        batch_size = b_1.num_graphs
        z1 = self.projector(self._gnn_forward(b_1))
        z2 = self.projector(self._gnn_forward(b_2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = _off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_ * off_diag
        return loss


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
