from src import types
from src import config
from src.original.gnn_model import EqvGNN as _EqvGNN


class EqvGNN(_EqvGNN, types.GNNModel):
    def __init__(
            self,
            num_layer=config.EqvGNN.num_layer,
            emb_dim=config.EqvGNN.emb_dim,
            use_edge_dropout=config.EqvGNN.use_edge_dropout,
            jk=config.EqvGNN.jk,
            drop_ratio=config.EqvGNN.drop_ratio,
    ):
        super().__init__(
            num_layer=config.EqvGNN.num_layer,
            emb_dim=config.EqvGNN.emb_dim,
            use_edge_dropout=config.EqvGNN.use_edge_dropout,
            JK=config.EqvGNN.jk,
            drop_ratio=config.EqvGNN.drop_ratio,
        )
        self.logger.debug(f'num_layer: {num_layer}')
        self.logger.debug(f'emb_dim: {emb_dim}')
        self.logger.debug(f'use_edge_dropout: {use_edge_dropout}')
        self.logger.debug(f'jk: {jk}')
        self.logger.debug(f'drop_ratio: {drop_ratio}')
