from src import types
from src.original.gnn_model import GNN as _GNN


class GNN(_GNN, types.GNNModel):
    def __init__(
            self,
            num_layer: int,
            emb_dim: int,
            jk: str,
            drop_ratio: float,
    ):
        super().__init__(
            num_layer=num_layer,
            emb_dim=emb_dim,
            JK=jk,
            drop_ratio=drop_ratio,
        )
        self.logger.debug(f'num_layer: {num_layer}')
        self.logger.debug(f'emb_dim: {emb_dim}')
        self.logger.debug(f'jk: {jk}')
        self.logger.debug(f'drop_ratio: {drop_ratio}')
