import argparse

from src.original.trans_bt.graph_trans_model import GraphTrans as _GraphTrans
from src.model.gnn import GNNModel


class GraphTrans(_GraphTrans, GNNModel):
    def __init__(
            self,
            gnn_dropout: float,
            d_model: int,
            transformer_dropout: float = 0.3,
    ):
        parser = argparse.ArgumentParser()
        GraphTrans.add_args(parser)
        parser.add_argument('--gnn_type', type=str, default='gin')
        args = parser.parse_args()
        args.gnn_dropout = gnn_dropout
        args.d_model = d_model
        args.transformer_dropout = transformer_dropout
        super().__init__(args)
