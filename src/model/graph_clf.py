from torch import nn
from torch_geometric.nn.glob import global_mean_pool

from src.model.gnn._gnn_model import GNNModel


class GraphClf(nn.Module):
    def __init__(
            self,
            dataset: str,
            gnn: GNNModel,
            use_graph_trans: bool,
    ):
        num_task_dict = {
            'tox21': 12,
            'hiv': 1,
            'muv': 17,
            'bace': 1,
            'bbbp': 1,
            'toxcast': 617,
            'sider': 27,
            'clintox': 2,
        }

        super(GraphClf, self).__init__()
        self.gnn = gnn
        self.use_graph_trans = use_graph_trans
        self.pool = global_mean_pool
        self.linear = nn.LazyLinear(num_task_dict[dataset])

    def forward(self, data):
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(data)
        graph_rep = node_representation if self.use_graph_trans else self.pool(node_representation, data.batch)
        return self.linear(graph_rep)
