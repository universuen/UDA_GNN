from torch import nn
from torch_geometric.nn.glob import global_mean_pool

from src import types


class GraphClf(nn.Module):
    def __init__(
            self,
            dataset: str,
            gnn: types.GNNModel,
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
        self.pool = global_mean_pool
        self.linear = nn.LazyLinear(num_task_dict[dataset])

    def forward(self, data):
        node_representation = self.gnn(data)
        graph_rep = self.pool(node_representation, data.batch)
        return self.linear(graph_rep)
