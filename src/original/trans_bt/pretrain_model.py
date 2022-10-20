import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool, global_add_pool
# from chem.graph_trans_model import GraphTrans


def weighted_global_mean_pool(x, batch, weights=None):
    if weights is None:
        return global_mean_pool(x, batch)
    weights = weights.view(-1, 1)
    g = global_add_pool(weights * x, batch)  # shape = [N, D]
    g_weight = global_add_pool(weights, batch)  # shape = [N, 1]
    g = g / g_weight
    return g


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class EqvGraphCL(nn.Module):
    def __init__(self, gnn, emb_dim, training_mode, temperature=0.1, eqv_coef=0.5):
        super(EqvGraphCL, self).__init__()
        self.temperature = temperature
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        self.eqv_coef = eqv_coef
        self.training_mode = training_mode
        self.use_mixed_negatives = False
        self.after_projector = False
        self.imix = False

    def gnn_forward(self, data):
        x = self.gnn(data)
        x = self.pool(x, data.batch)
        return x

    def loss_cl(self, x1, x2):
        T = self.temperature
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = torch.diag(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_cl_v2(self, x1, x2, semi_pos):
        '''
        Contrastive loss that excludes semi-positives from the negatives
        x1: shape = [B1, D]
        x2: shape = [B2, D]; B2 >= B1
        semi_pos: [B1, k]. Saves the position of semi-positive samples, which should be excluded from the negatives.
        '''
        B = x1.shape[0]
        device = x1.device
        T = self.temperature
        x1_abs = x1.norm(dim=1).view(-1, 1)  # shape = [B, 1]
        x2_abs = x2.norm(dim=1).view(1, -1)  # shape = [1, 2B]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / x1_abs / x2_abs

        sim_matrix = torch.exp(sim_matrix / T)  # shape = [B, 2B]

        mask = torch.zeros((B, x2.shape[0]), device=device, dtype=torch.bool)  # shape = [B1, B2]
        poses = torch.cat((torch.arange(B).view(-1, 1), semi_pos), dim=1).to(device)  # shape = [B,k+1]
        mask = mask.scatter_(1, poses, True)

        pos_sim = torch.diag(sim_matrix)
        loss = pos_sim / sim_matrix[~mask].view(B, -1).sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

    def forward(self, *args):
        if self.imix:
            return self.imix_forward(*args)

        if self.use_mixed_negatives:
            return self.cl_with_mixed_negatives(*args)
        else:
            return self.cl_without_mixed_negatives(*args)

    def imix_forward(self, data1, data2, mix_data, permutation, lamb):
        '''
        mix_data = lambda * data1 + (1-lambda) * data2
        '''
        N = data1.__num_graphs__
        device = data1.x.device
        p1 = self.projector(self.gnn_forward(data1))
        p2 = self.projector(self.gnn_forward(data2))
        mix_p = self.projector(self.gnn_forward(mix_data))
        inv_loss = self.loss_cl(p1, p2)

        p1 = F.normalize(p1, dim=-1, p=2)  # shape = [N, D]
        p2 = F.normalize(p2, dim=-1, p=2)  # shape = [N, D]
        mix_p = F.normalize(mix_p, dim=-1, p=2)  # shape = [N, D]

        eqv_loss = F.cross_entropy(mix_p @ p1.T, target=torch.arange(N, device=device),
                                   reduction='none') * lamb + F.cross_entropy(mix_p @ p2.T,
                                                                              target=permutation.to(device),
                                                                              reduction='none') * (1 - lamb)
        eqv_loss = eqv_loss.mean()
        loss = (1 - self.eqv_coef) * inv_loss + self.eqv_coef * eqv_loss
        return loss, inv_loss, eqv_loss

    def cl_with_mixed_negatives(self, data1, data2, mix_data, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        '''
        B = data1.__num_graphs__
        B2 = B // 2

        z1 = self.gnn_forward(data1)
        z2 = self.gnn_forward(data2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        raw_mix_p = self.projector(self.gnn_forward(mix_data))
        ## mixup: f(lamb * T1 * g1 + (1-lamb) * T2 * g2) = lamb * f(T1 * g1) + (1-lamb) * f(T2 * g2)

        if self.after_projector:
            man_mix_p = lamb * p1 + (1 - lamb) * p2[permutation]
        else:
            man_mix_p = self.projector(lamb * z1 + (1 - lamb) * z2[permutation])

        mix_mix_p1 = torch.cat((raw_mix_p[:B2], man_mix_p[B2:]), dim=0)  # shape = [B, D]
        mix_mix_p2 = torch.cat((man_mix_p[:B2], raw_mix_p[B2:]), dim=0)  # sahpe = [B, D]

        inv_semi_pos = torch.stack((torch.arange(B, 2 * B), permutation + B), dim=1)  # shape = [B, 2]
        inv_loss = self.loss_cl_v2(p1, torch.cat((p2, mix_mix_p2), dim=0), inv_semi_pos)

        eqv_semi_pos = torch.stack((permutation, torch.arange(B, 2 * B), permutation + B), dim=1)  # shape = [B, 3]
        eqv_loss = self.loss_cl_v2(mix_mix_p1, torch.cat((mix_mix_p2, p2), dim=0), eqv_semi_pos)

        loss = (1 - self.eqv_coef) * inv_loss + self.eqv_coef * eqv_loss
        return loss, inv_loss, eqv_loss

    def cl_without_mixed_negatives(self, data1, data2, mix_data, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        '''
        z1 = self.gnn_forward(data1)
        z2 = self.gnn_forward(data2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        raw_mix_p = self.projector(self.gnn_forward(mix_data))
        ## mixup: f(lamb * T1 * g1 + (1-lamb) * T2 * g2) = lamb * f(T1 * g1) + (1-lamb) * f(T2 * g2)
        man_mix_p1 = self.projector(lamb * z1 + (1 - lamb) * z2[permutation])

        B2 = raw_mix_p.shape[0] // 2
        mix_mix_p1 = torch.cat((raw_mix_p[:B2], man_mix_p1[B2:]), dim=0)
        mix_mix_p2 = torch.cat((man_mix_p1[:B2], raw_mix_p[B2:]), dim=0)

        inv_loss = self.loss_cl(p1, p2)
        eqv_loss = self.loss_cl_v2(mix_mix_p1, mix_mix_p2, permutation.view(-1, 1))

        loss = (1 - self.eqv_coef) * inv_loss + self.eqv_coef * eqv_loss
        return loss, inv_loss, eqv_loss


class EqvBarlowTwins(nn.Module):
    def __init__(self, model, lambd, training_mode, use_graph_trans, sizes=None):
        super().__init__()
        self.lambd = lambd
        self.gnn = model
        self.use_graph_trans = use_graph_trans

        # projector
        if sizes is None:
            sizes = [128, 1200, 1200, 1200] if use_graph_trans else [300, 1200, 1200, 1200]
        # sizes = [300, 1200, 1200, 1200]
        self.projector_aug = Projector(sizes)
        self.projector_mix = Projector(sizes)

        # normalization layer for the representations z1 and z2
        self.bn_aug = nn.BatchNorm1d(sizes[-1], affine=False)
        self.bn_mix = nn.BatchNorm1d(sizes[-1], affine=False)
        self.bn_man = nn.BatchNorm1d(sizes[-1], affine=False)
        self.pool = global_mean_pool
        self.training_mode = training_mode
        self.after_projector = False
        self.eqv_coeff = 0.5
        self.use_unified_forward = False

    def bt_loss(self, b1, b2):
        '''
        b1: shape = [B, D]
        b2: shape = [B, D]
        For original samples
            p1: T_1 \circ x
            p2: T_2 \circ x
        For mixed samples
            p1: f(alpha * x_1 + (1-alpha) * x_2)
            p2: alpha * f(x_1) + (1-alpha) * f(x_2)
        '''
        batch_size = b1.shape[0]
        # empirical cross-correlation matrix
        c = b1.T @ b2

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def gnn_forward(self, data):
        x = self.gnn(data)
        if not self.use_graph_trans:
            x = self.pool(x, data.batch)
        return x

    def before_projector_forward(self, batch_list, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        lamb: shape = [B, 1]
        '''
        if not self.use_graph_trans:
            integrated_batch = batch_list[0]
            batch_size = integrated_batch.num_graphs // 3
            # calculate z of integrated batch
            integrated_z = self.gnn_forward(integrated_batch)
            z1, z2, _ = torch.split(integrated_z, batch_size)
            integrated_z = torch.cat(
                [
                    integrated_z,
                    lamb * z1 + (1 - lamb) * z2[permutation],
                ]
            )
            # project
            integrated_p = self.projector(integrated_z)
            # apply batch norm
            p1, p2, mix_p, man_p = torch.split(integrated_p, batch_size)
            inv_loss = self.bt_loss(self.bn_aug(p1), self.bn_aug(p2))
            eqv_loss = self.bt_loss(self.bn_mix(mix_p), self.bn_man(man_p))
        else:
            # z_1, z_2, z_mix = [self.gnn_forward(b) for b in batch_list]
            # z_list = [z_1, z_2, z_mix, lamb * z_1 + (1 - lamb) * z_2[permutation]]
            # p1, p2, mix_p, man_p = [self.projector(z) for z in z_list]
            # inv_loss = self.bt_loss(self.bn_aug(p1), self.bn_aug(p2))
            # eqv_loss = self.bt_loss(self.bn_aug(mix_p), self.bn_aug(man_p))
            z1, z2, z_mix = [self.gnn_forward(b) for b in batch_list]
            z_man = lamb * z1 + (1 - lamb) * z2[permutation]
            p1, p2 = map(self.projector_aug, [z1, z2])
            mix_p, man_p = map(self.projector_mix, [z_mix, z_man])
            inv_loss = self.bt_loss(self.bn_aug(p1), self.bn_aug(p2))
            eqv_loss = self.bt_loss(self.bn_aug(mix_p), self.bn_aug(man_p))

        loss = (1 - self.eqv_coeff) * inv_loss + self.eqv_coeff * eqv_loss
        return loss, inv_loss, eqv_loss

    def after_projector_forward(self, data1, data2, mix_data, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs x
        data2: augmented graphs [N, D]
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        '''
        p1 = self.projector(self.gnn_forward(data1))  # [N, D]
        p2 = self.projector(self.gnn_forward(data2))  # [N, D]
        raw_mix_p = self.projector(self.gnn_forward(mix_data))  # [N, D] lamb * data1 + (1-lamb) * data2

        loss = 0
        ## regular
        inv_loss = self.bt_loss(p1, p2)
        loss += inv_loss

        count = 1
        eqv_loss = 0
        if self.training_mode.find('equivariance') >= 0:
            ## mixup: f(lamb * T1 * g1 + (1-lamb) * T2 * g2) = lamb * f(T1 * g1) + (1-lamb) * f(T2 * g2)
            man_mix_p1 = lamb * p1 + (1 - lamb) * p2[permutation]
            eqv_loss = self.bt_loss(raw_mix_p, man_mix_p1)
            loss += eqv_loss
            count += 1

        loss /= count
        return loss, inv_loss, eqv_loss

    def forward(self, batch_list, permutation, lamb):
        if self.use_unified_forward:
            return self.unified_forward(batch_list, permutation, lamb)
        else:
            if self.after_projector:
                return self.after_projector_forward(batch_list, permutation, lamb)
            else:
                return self.before_projector_forward(batch_list, permutation, lamb)

    def unified_forward(self, data1, data2, mix_data, permutation, lamb):
        z1 = self.gnn_forward(data1)
        z2 = self.gnn_forward(data2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        man_mix_p = self.projector(lamb * z1 + (1 - lamb) * z2[permutation])
        raw_mix_p = self.projector(self.gnn_forward(mix_data))

        inv_loss = self.__bt_loss(self.bn(p1), self.bn(p2))
        eqv_loss = self.__bt_loss(self.bn_mix(raw_mix_p), self.bn_man(man_mix_p))
        loss = (1 - self.eqv_coeff) * inv_loss + self.eqv_coeff * eqv_loss
        return loss, inv_loss, eqv_loss

    def __bt_loss(self, p1, p2):
        batch_size = p1.shape[0]
        c = p1.T @ p2
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class BarlowTwins(nn.Module):
    def __init__(self, model, lambd, sizes, use_graph_trans=False):
        super().__init__()
        self.lambd = lambd
        self.gnn = model
        self.use_graph_trans = use_graph_trans
        # projector
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.pool = global_mean_pool

    def gnn_forward(self, data):
        if self.use_graph_trans:
            x = self.gnn(data)
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = self.gnn(x, edge_index, edge_attr)
            x = self.pool(x, batch)
        return x

    def forward(self, data1, data2):
        # batch_size = data1.x.shape[0]
        batch_size = data1.num_graphs
        z1 = self.projector(self.gnn_forward(data1))
        z2 = self.projector(self.gnn_forward(data2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class BarlowTwinsCSE(nn.Module):
    def __init__(self, model, emb_dim, lambd):
        super().__init__()
        self.lambd = lambd
        self.gnn = model
        assert self.gnn.drop_ratio > 0

        # projector
        sizes = [emb_dim, emb_dim * 4, emb_dim * 4, emb_dim * 4]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.pool = global_mean_pool

    def gnn_forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        return x

    def forward(self, data):
        batch_size = data.__num_graphs__
        z1 = self.projector(self.gnn_forward(data))
        z2 = self.projector(self.gnn_forward(data))
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def Projector(sizes):
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    return nn.Sequential(*layers)


# class graphcl(nn.Module):
#
#     def __init__(self, gnn, emb_dim):
#         super(graphcl, self).__init__()
#         self.gnn = gnn
#         self.pool = global_mean_pool
#         self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
#
#     def gnn_forward(self, data):
#         if isinstance(self.gnn, GraphTrans):
#             z = self.gnn(data)
#         else:
#             x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#             x = self.gnn(x, edge_index, edge_attr)
#             z = self.pool(x, batch)
#         p = self.projector(z)
#         p = F.normalize(p, p=2, dim=-1)
#         return p
#
#     def forward(self, data1, data2):
#         p1 = self.gnn_forward(data1)
#         p2 = self.gnn_forward(data2)
#         return self.loss_cl(p1, p2)
#
#     def loss_cl(self, x1, x2):
#         '''
#         x1: shape = [B, D]
#         x2: shape = [B, D]
#         '''
#         T = 0.1
#         batch_size, _ = x1.size()
#         x1_abs = x1.norm(dim=1)
#         x2_abs = x2.norm(dim=1)
#
#         sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
#         sim_matrix = torch.exp(sim_matrix / T)
#         pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#         loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#         loss = - torch.log(loss).mean()
#         return loss


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, dropout=0.0, num_layers=2, batch_norm=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        if num_layers == 1:
            hid_dim = out_dim
        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hid_dim))
        self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                self.lins.append(nn.Linear(hid_dim, out_dim))
                self.batch_norms.append(nn.BatchNorm1d(out_dim))
            else:
                self.lins.append(nn.Linear(hid_dim, hid_dim))
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for lin, batch_norm in zip(self.lins, self.batch_norms):
            lin.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, x):
        for i, (lin, bn) in enumerate(zip(self.lins, self.batch_norms)):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = bn(x) if self.batch_norm else x
        return x


class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True, lin=True, dropout=0.0):
        super(SplineCNN, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SplineConv(in_channels, out_channels, dim, kernel_size=5)
            self.convs.append(conv)
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = nn.Linear(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(in_channels))

        self.input_node_embeddings = nn.Embedding(2, in_channels)
        nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        # print(x.shape)
        xs = [self.input_node_embeddings(x.to(torch.long).reshape(-1))]
        # print(xs[0].shape)
        for i, conv in enumerate(self.convs):
            h = conv(xs[-1], edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            xs.append(h)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.dim, self.num_layers, self.cat,
                                      self.lin, self.dropout)


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class EqvSimSiam(nn.Module):
    def __init__(self, model, emb_dim, eqv_coef):
        super().__init__()
        self.gnn = model
        self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim, bias=False),
                                       nn.BatchNorm1d(emb_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(emb_dim, emb_dim, bias=False),
                                       nn.BatchNorm1d(emb_dim, affine=False))

        h_dim = emb_dim
        self.predictor = nn.Sequential(nn.Linear(emb_dim, h_dim, bias=False),
                                       nn.BatchNorm1d(h_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(h_dim, emb_dim))
        self.pool = global_mean_pool
        self.criterion = lambda x, y: F.cosine_similarity(x, y, dim=1).mean()
        self.eqv_coef = eqv_coef
        self.use_comp = False

    def gnn_forward(self, data):
        x = self.gnn(data)
        x = self.pool(x, data.batch)
        return x

    def forward2(self, data1, data2, mix_data, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        lamb: shape = [B, 1]
        '''
        z1 = self.gnn_forward(data1)
        z2 = self.gnn_forward(data2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        raw_mix_p = self.projector(self.gnn_forward(mix_data))

        if self.use_comp:
            man_mix_p = self.projector(lamb * z2 + (1 - lamb) * z1[permutation])
        else:
            ## mixup: f(lamb * T1 * g1 + (1-lamb) * T2 * g2) = lamb * f(T1 * g1) + (1-lamb) * f(T2 * g2)
            man_mix_p = self.projector(lamb * z1 + (1 - lamb) * z2[permutation])

        inv_loss = (self.criterion(self.predictor(p1), p2.detach()) + self.criterion(self.predictor(p2),
                                                                                     p1.detach())) / 2
        eqv_loss = (self.criterion(self.predictor(raw_mix_p), man_mix_p.detach()) + self.criterion(
            self.predictor(man_mix_p), raw_mix_p.detach())) / 2

        loss = (1 - self.eqv_coef) * inv_loss + self.eqv_coef * eqv_loss
        return loss, inv_loss, eqv_loss

    def forward(self, data1, data2, mix_data, permutation, lamb):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        lamb: shape = [B, 1]
        '''

        p1 = self.projector(self.gnn_forward(data1))
        p2 = self.projector(self.gnn_forward(data2))
        raw_mix_p = self.projector(self.gnn_forward(mix_data))
        ## mixup: f(lamb * T1 * g1 + (1-lamb) * T2 * g2) = lamb * f(T1 * g1) + (1-lamb) * f(T2 * g2)
        man_mix_p = self.projector(lamb * p1 + (1 - lamb) * p2[permutation])
        inv_loss = (self.criterion(self.predictor(p1), p2.detach()) + self.criterion(self.predictor(p2),
                                                                                     p1.detach())) / 2
        eqv_loss = self.criterion(self.predictor(raw_mix_p), man_mix_p.detach()) / 2
        loss = (1 - self.eqv_coef) * inv_loss + self.eqv_coef * eqv_loss
        return loss, inv_loss, eqv_loss


class SimSiam(nn.Module):
    def __init__(self, model, emb_dim):
        super().__init__()
        self.gnn = model
        self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim, bias=False),
                                       nn.BatchNorm1d(emb_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(emb_dim, emb_dim, bias=False),
                                       nn.BatchNorm1d(emb_dim, affine=False))
        h_dim = emb_dim
        self.predictor = nn.Sequential(nn.Linear(emb_dim, h_dim, bias=False),
                                       nn.BatchNorm1d(h_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(h_dim, emb_dim))
        self.pool = global_mean_pool
        self.criterion = lambda x, y: F.cosine_similarity(x, y, dim=1).mean()

    def gnn_forward(self, data):
        x = self.gnn(data)
        x = self.pool(x, data.batch)
        return x

    def forward(self, data1, data2):
        '''
        Note: data1 and data2 are the same graphs of different augmentations
        data1: augmented graphs
        data2: augmented graphs
        mix_data: lamb * data1 + (1-lamb) * data2[permutation]
        lamb: shape = [B, 1]
        '''
        z1 = self.gnn_forward(data1)
        z2 = self.gnn_forward(data2)
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        loss = (self.criterion(self.predictor(p1), p2.detach()) + self.criterion(self.predictor(p2), p1.detach())) / 2
        return loss
