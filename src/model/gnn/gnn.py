from src import types
import config
from src.original.gnn_model import GNN as _GNN


class GNN(_GNN, types.GNNModel):
    def __init__(
            self,
            num_layer=config.GNN.num_layer,
            emb_dim=config.GNN.emb_dim,
            jk=config.GNN.jk,
            drop_ratio=config.GNN.drop_ratio,
    ):
        super().__init__(
            num_layer=config.GNN.num_layer,
            emb_dim=config.GNN.emb_dim,
            JK=config.GNN.jk,
            drop_ratio=config.GNN.drop_ratio,
        )
        self.logger.debug(f'num_layer: {num_layer}')
        self.logger.debug(f'emb_dim: {emb_dim}')
        self.logger.debug(f'jk: {jk}')
        self.logger.debug(f'drop_ratio: {drop_ratio}')
