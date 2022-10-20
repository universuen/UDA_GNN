from __future__ import annotations

from src import types
from src.original.pretrain_model import BarlowTwins as _BarlowTwins
from src.model.pretraining import PretrainingModel


class BarlowTwins(_BarlowTwins, PretrainingModel):
    def __init__(
            self,
            model: types.GNNModel,
            lambda_: float,
            sizes: tuple[int],
            use_graph_trans: bool
    ):
        super().__init__(
            model=model,
            lambd=lambda_,
            sizes=sizes,
            use_graph_trans=use_graph_trans,
        )
