from src.model.gnn._gnn_model import GNNModel
from src.original.mae.model import GNNDecoder


class Decoder(GNNDecoder, GNNModel):
    def __init__(
            self,
            hidden_dim: int,
            out_dim: int,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
        # remove self loop adding process
        self.conv.add_selfloop = False
