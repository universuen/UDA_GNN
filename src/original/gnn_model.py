import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # edge_index shape = [2, E]; x shape = [N ,D]; edge_embedding shape = [E, D]
        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        # return self.my_propagate(x, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        '''
        x_j shape = [E, D]
        edge_attr shape = [E, D]
        '''
        return x_j + edge_attr

    def update(self, aggr_out):
        '''
        aggr_out shape = [N, D]
        '''
        return self.mlp(aggr_out)


class DropoutRow(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, input):
        weight = torch.ones((input.shape[0], 1), device=input.device)
        weight = self.dropout(weight)
        return weight * input


class MixGINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, dropout_ratio, aggr="add"):
        super(MixGINConv, self).__init__()
        # multi-layer perceptron
        self.dropout = DropoutRow(dropout_ratio)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding = nn.Linear(num_bond_type + num_bond_direction, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.edge_embedding.weight[:, :num_bond_type].T)
        nn.init.xavier_uniform_(self.edge_embedding.weight[:, num_bond_type:].T)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, edge_weight=None, self_loop_weight=None):
        '''
        x, shape = [N, D]
        edge_index, shape = [E, 2]
        edge_attr, shape = [E, D]
        edge_weight, shape = [E, 1]
        '''
        device = edge_attr.device
        dtype = edge_attr.dtype
        N = x.shape[0]

        if edge_attr.shape[1] == 2:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            # add features corresponding to self-loop edges.
            # add self loops in the edge space
            assert edge_weight is None and self_loop_weight is None
            # regular input
            self_loop_attr = torch.zeros((N, 2), device=device, dtype=dtype)
            self_loop_attr[:, 0] = 4
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
            embedding = self.edge_embedding.weight.T
            edge_embeddings = F.embedding(edge_attr[:, 0], embedding[:num_bond_type]) + F.embedding(edge_attr[:, 1],
                                                                                                    embedding[
                                                                                                    num_bond_type:])
        else:
            # mix input. I have added the self loops and the corresponding edge attributes when mixing graphs
            assert edge_weight is not None
            edge_embeddings = self.edge_embedding(edge_attr)

        # edge_index shape = [2, E]; x shape = [N ,D]; edge_embedding shape = [E, D]
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, weight=edge_weight)

    def forward2(self, x, edge_index, edge_attr, edge_weight=None, self_loop_weight=None):
        '''
        x, shape = [N, D]
        edge_index, shape = [E, 2]
        edge_attr, shape = [E, D]
        edge_weight, shape = [E, 1]
        '''
        device = edge_attr.device
        dtype = edge_attr.dtype
        N = x.shape[0]

        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        # add features corresponding to self-loop edges.
        if edge_attr.shape[1] == 2:
            # add self loops in the edge space
            assert edge_weight is None and self_loop_weight is None
            # regular input
            self_loop_attr = torch.zeros((N, 2), device=device, dtype=dtype)
            self_loop_attr[:, 0] = 4
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
            embedding = self.edge_embedding.weight.T
            edge_embeddings = F.embedding(edge_attr[:, 0], embedding[:num_bond_type]) + F.embedding(edge_attr[:, 1],
                                                                                                    embedding[
                                                                                                    num_bond_type:])
        else:
            # mix input
            assert edge_weight is not None and self_loop_weight is not None
            edge_weight = torch.cat((edge_weight, self_loop_weight), dim=0)
            self_loop_attr = torch.zeros((N, num_bond_type + num_bond_direction), device=device, dtype=dtype)
            self_loop_attr[:, 4] = self_loop_weight  # bond type for self-loop edge
            self_loop_attr[:, num_bond_type] = self_loop_weight  # the default direction of self-loop edge
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
            edge_embeddings = self.edge_embedding(edge_attr)

        # edge_index shape = [2, E]; x shape = [N ,D]; edge_embedding shape = [E, D]
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, weight=edge_weight)

    def message(self, x_j, edge_attr, weight=None):
        '''
        x_j shape = [E, D]
        edge_attr shape = [E, D]
        weights shape = [E, 1]
        '''
        if weight is not None:
            x_j = x_j * weight.view(-1, 1)
        ## edge_attr's weight has been multiplied during mixup
        message = x_j + edge_attr
        return self.dropout(message)

    def update(self, aggr_out):
        '''
        aggr_out shape = [N, D]
        '''
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_representation


class EqvGNN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, use_edge_dropout=False, JK="last", drop_ratio=0, gnn_type="gin"):
        if gnn_type != 'gin': raise NotImplementedError
        super(EqvGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = nn.Linear(num_atom_type + num_chirality_tag, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.x_embedding.weight[:, :num_atom_type].T)
        nn.init.xavier_uniform_(self.x_embedding.weight[:, num_atom_type:].T)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if use_edge_dropout:
                self.gnns.append(MixGINConv(emb_dim, drop_ratio, aggr="add"))
            else:
                self.gnns.append(MixGINConv(emb_dim, 0, aggr="add"))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, *args):
        if len(args) == 1:
            data = args[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        else:
            x, edge_index, edge_attr = args
            edge_weight = None

        # x shape = [N, num_atom_type+num_chirality_tag]
        if x.shape[1] == 2:
            # regular input
            embedding = self.x_embedding.weight.T
            x = F.embedding(x[:, 0], embedding[: num_atom_type]) + F.embedding(x[:, 1], embedding[num_atom_type:])
        else:
            # mixed input
            x = self.x_embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, edge_weight)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_representation

# class EqvGNNBN(torch.nn.Module):
#     """

#     Args:
#         num_layer (int): the number of GNN layers
#         emb_dim (int): dimensionality of embeddings
#         JK (str): last, concat, max or sum.
#         max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
#         drop_ratio (float): dropout rate
#         gnn_type: gin, gcn, graphsage, gat

#     Output:
#         node representations

#     """
#     def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
#         if gnn_type != 'gin': raise NotImplementedError
#         super(EqvGNNBN, self).__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = drop_ratio
#         self.JK = JK

#         if self.num_layer < 2:
#             raise ValueError("Number of GNN layers must be greater than 1.")

#         self.x_embedding = nn.Linear(num_atom_type + num_chirality_tag, emb_dim, bias=False)
#         nn.init.xavier_uniform_(self.x_embedding.weight[:, :num_atom_type].T)
#         nn.init.xavier_uniform_(self.x_embedding.weight[:, num_atom_type:].T)

#         ###List of MLPs
#         self.gnns = nn.ModuleList()
#         for layer in range(num_layer):
#             self.gnns.append(MixGINConv(emb_dim, aggr = "add"))

#         ###List of batchnorms
#         self.batch_norms0 = torch.nn.ModuleList()
#         for layer in range(num_layer):
#             self.batch_norms0.append(nn.BatchNorm1d(emb_dim))

#         self.batch_norms1 = torch.nn.ModuleList()
#         for layer in range(num_layer):
#             self.batch_norms1.append(nn.BatchNorm1d(emb_dim))
#         self.bn_mode = 'left'

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

#         # x shape = [N, num_atom_type+num_chirality_tag]
#         if x.shape[1] == 2:
#             # regular input
#             embedding = self.x_embedding.weight.T
#             x = F.embedding(x[:, 0], embedding[: num_atom_type]) + F.embedding(x[:, 1], embedding[num_atom_type: ])
#         else:
#             # mixed input
#             x = self.x_embedding(x)

#         if self.bn_mode == 'left':
#             batch_norms = self.batch_norms0
#         elif self.bn_mode == 'right':
#             batch_norms = self.batch_norms1
#         else:
#             raise NotImplementedError

#         h_list = [x]
#         for layer in range(self.num_layer):
#             h = self.gnns[layer](h_list[layer], edge_index, edge_attr, edge_weight)
#             h = batch_norms[layer](h)
#             if layer == self.num_layer - 1:
#                 #remove relu for the last layer
#                 h = F.dropout(h, self.drop_ratio, training = self.training)
#             else:
#                 h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
#             h_list.append(h)

#         ### Different implementations of Jk-concat
#         if self.JK == "concat":
#             node_representation = torch.cat(h_list, dim = 1)
#         elif self.JK == "last":
#             node_representation = h_list[-1]
#         elif self.JK == "max":
#             h_list = [h.unsqueeze_(0) for h in h_list]
#             node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
#         elif self.JK == "sum":
#             h_list = [h.unsqueeze_(0) for h in h_list]
#             node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
#         return node_representation
