import torch
import torch.nn.functional as functional
import torch.nn as nn
import random

import src.model
from src.original.trans_bt.gnn_model import GNN as _GNN
from src.model.gnn import GNNModel


class GNN(_GNN, GNNModel):
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
        self.node_prompts = None
        self.edge_prompt = None
        self.mask_ratio = 0
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        if self.mask_ratio > 0 and self.training:
            '''generate random mask'''
            N = x.shape[0]
            mask = x.new_empty((N, 1), dtype=torch.float).uniform_() > self.mask_ratio
            x = x * mask.detach()

        h_list = [x]
        for layer in range(self.num_layer):
            h = h_list[layer]
            if self.node_prompts is not None:
                try:
                    if type(self.node_prompts[0]) == src.model.NodePromptPtb:
                        h = self.node_prompts[layer](h, argv[0].batch)
                    else:
                        h = self.node_prompts[layer](h)
                except IndexError:
                    pass
            h = self.gnns[layer](h, edge_index, edge_attr, edge_prompt=self.edge_prompt)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = self.dropout(h)
            else:
                h = self.dropout(functional.relu(h))
            h_list.append(h)

        node_representation = None
        # Different implementations of Jk-concat
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
