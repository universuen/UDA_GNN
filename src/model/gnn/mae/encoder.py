from src.model.gnn._gnn_model import GNNModel
from src.original.mae.model import GNN


class Encoder(GNN, GNNModel):
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
        # remove self loop adding process
        for conv in self.gnns:
            conv.add_selfloop = False
