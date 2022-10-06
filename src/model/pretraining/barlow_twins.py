from __future__ import annotations

import src
# import config
from src import types
from src.original.pretrain_model import BarlowTwins as _BarlowTwins


class BarlowTwins(_BarlowTwins, types.PretrainingModel):
    def __init__(
            self,
            model: types.GNNModel,
            lambda_: float,
            sizes: tuple[int],
    ):
        super().__init__(
            model=model,
            lambd=lambda_,
            sizes=sizes,
        )
        self.logger.debug(f'model: {model}')
        self.logger.debug(f'lambda_: {lambda_}')
        self.logger.debug(f'sizes: {sizes}')


