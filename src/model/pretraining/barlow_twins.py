from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.glob import global_mean_pool

from torch_geometric.data import Data, Batch

import src
import config
from src import types
from src.original.pretrain_model import BarlowTwins as _BarlowTwins


class BarlowTwins(_BarlowTwins, types.PretrainingModel):
    def __init__(
            self,
            model: types.GNNModel | None = None,
            lambda_: float = config.BarlowTwins.lambda_,
            sizes: tuple[int] = config.BarlowTwins.sizes,
    ):
        if model is None:
            model = src.model.gnn.GNN()
        super().__init__(
            model=model,
            lambd=lambda_,
            sizes=sizes,
        )
        self.logger.debug(f'model: {model}')
        self.logger.debug(f'lambda_: {lambda_}')
        self.logger.debug(f'sizes: {sizes}')


