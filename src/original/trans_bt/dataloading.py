import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

num_atom_type = 120
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3


def sort_data(data):
    N = data.x.shape[0]
    x = data.x
    sort_values = x[:, 0] * num_chirality_tag + x[:, 1]
    perm = torch.argsort(sort_values)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.x = x[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data


def rand_permute(data):
    N = data.x.shape[0]
    perm = torch.randperm(N)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.x = data.x[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data


def wrong_rand_permute(data):
    N = data.x.shape[0]
    perm = torch.randperm(N)
    data.edge_index = perm[data.edge_index]
    return data


def pseudo_randperm(N):
    '''Make sure no data is permuted to itself'''
    perm = torch.randperm(N)
    identity = torch.arange(N)
    while torch.any(perm == identity):
        perm = torch.randperm(N)
    return perm


class MixChemCollater(Collater):
    def __init__(self, beta1, beta2, use_rand_perm, follow_batch, exclude_keys, add_self_loop, rebatch):
        self.beta1 = beta1
        self.beta2 = beta2
        self.use_rand_perm = use_rand_perm
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.pseudo_randperm = False
        self.use_arraylamb = False
        self.add_self_loop = add_self_loop
        self.rebatch = rebatch

    @staticmethod
    def _rebatch(data1, data2, mix_data):
        assert data1.num_graphs == data2.num_graphs == mix_data.num_graphs
        device = mix_data.x.device
        # update data1
        data1.x = torch.cat(
            [
                torch.nn.functional.one_hot(data1.x[:, 0], num_atom_type).float(),
                torch.nn.functional.one_hot(data1.x[:, 1], num_chirality_tag).float(),
            ],
            dim=1
        )
        data1.edge_index, _ = add_self_loops(data1.edge_index, num_nodes=data1.num_nodes)
        self_loop_attr = torch.zeros((data1.num_nodes, 2), device=device)
        self_loop_attr[:, 0] = 4
        data1.edge_attr = torch.cat((data1.edge_attr, self_loop_attr), dim=0).long()
        data1.edge_attr = torch.cat(
            [
                torch.nn.functional.one_hot(data1.edge_attr[:, 0], num_bond_type).float(),
                torch.nn.functional.one_hot(data1.edge_attr[:, 1], num_bond_direction).float(),
            ],
            dim=1
        )
        # update data2
        data2.x = torch.cat(
            [
                torch.nn.functional.one_hot(data2.x[:, 0], num_atom_type).float(),
                torch.nn.functional.one_hot(data2.x[:, 1], num_chirality_tag).float(),
            ],
            dim=1
        )
        data2.edge_index, _ = add_self_loops(data2.edge_index, num_nodes=data2.num_nodes)
        self_loop_attr = torch.zeros((data2.num_nodes, 2), device=device)
        self_loop_attr[:, 0] = 4
        data2.edge_attr = torch.cat((data2.edge_attr, self_loop_attr), dim=0).long()
        data2.edge_attr = torch.cat(
            [
                torch.nn.functional.one_hot(data2.edge_attr[:, 0], num_bond_type).float(),
                torch.nn.functional.one_hot(data2.edge_attr[:, 1], num_bond_direction).float(),
            ],
            dim=1
        )
        # rebatch
        new_batch = Batch(
            x=torch.cat([data1.x, data2.x, mix_data.x]),
            edge_index=torch.cat(
                [
                    data1.edge_index,
                    data2.edge_index + data1.x.shape[0],
                    mix_data.edge_index + data2.x.shape[0] + data1.x.shape[0],
                ],
                dim=1,
            ),
            edge_attr=torch.cat([data1.edge_attr, data2.edge_attr, mix_data.edge_attr]),
            edge_weight=torch.cat(
                [
                    torch.ones(data1.num_edges + data2.num_edges, device=device),
                    mix_data.edge_weight,
                ]
            ),
            ptr=torch.cat(
                [
                    data1.ptr,
                    data2.ptr[1:] + data1.ptr[-1],
                    mix_data.ptr[1:] + data2.ptr[-1] + data1.ptr[-1],
                ]
            ),
            batch=torch.cat(
                [
                    data1.batch,
                    data2.batch + data1.batch[-1] + 1,
                    mix_data.batch + data2.batch[-1] + data1.batch[-1] + 2,
                ]
            ),
            node_weights=torch.cat(
                [
                    torch.ones(data1.num_nodes + data2.num_nodes, device=mix_data.node_weights.device),
                    mix_data.node_weights,
                ]
            ),
        )
        new_batch.__num_graphs__ = new_batch.ptr.numel() - 1
        return new_batch.to(device)

    def __call__(self, data_list):
        # batch_1, batch_2, _ = super(MixChemCollater, self).__call__(data_list)
        batch_1, batch_2, _ = [super(MixChemCollater, self).__call__(s) for s in zip(*data_list)]
        ## data_list0 and data_list1 are the same graphs with different augmentations
        if len(data_list[0]) == 3:
            data_list0, data_list1, index = zip(*data_list)
        elif len(data_list[0]) == 2:
            data_list0, data_list1 = zip(*data_list)
        else:
            raise NotImplementedError
        assert len(data_list0) == len(data_list1)

        if self.pseudo_randperm:
            permutation = pseudo_randperm(len(data_list1))
        else:
            permutation = torch.randperm(len(data_list1))

        data_list1 = [data_list1[i] for i in permutation]
        if not self.use_arraylamb:
            lamb = np.random.beta(self.beta1, self.beta2)
            mixed_batch = mix_data_list_add_self_loop(data_list0, data_list1, lamb, self.use_rand_perm)
        else:
            lamb = torch.from_numpy(np.random.beta(self.beta1, self.beta2, size=(len(data_list0),))).to(torch.float)
            mixed_batch = mix_data_list_add_self_loop_arraylamb(
                data_list0,
                data_list1,
                lamb,
                self.use_rand_perm,
                self.add_self_loop,
            )
            lamb = lamb.view(-1, 1)
            if self.rebatch:
                integrated_batch = self._rebatch(batch_1, batch_2, mixed_batch)
                return integrated_batch, (mixed_batch, permutation, lamb)
            else:
                return batch_1, batch_2, (mixed_batch, permutation, lamb)


@torch.no_grad()
def mix_data_list(data_list1, data_list2, lamb, use_rand_perm):
    device = data_list1[0].x.device
    DN = num_atom_type + num_chirality_tag
    DE = num_bond_type + num_bond_direction

    if use_rand_perm:
        ## ensure the relative permutations between the graphs are random
        # data_list1 = [rand_permute(data) for data in data_list1]
        data_list2 = [rand_permute(data) for data in data_list2]

    ptr = [0]
    batch_index = []
    for i, (data1, data2) in enumerate(zip(data_list1, data_list2)):
        N = max(data1.x.shape[0], data2.x.shape[0])
        ptr.append(ptr[-1] + N)
        batch_index.append(torch.full((N,), i, dtype=torch.long, device=device))
    batch_index = torch.cat(batch_index, dim=0)

    edge_attr_list1, edge_attr_list2 = [], []
    edge_index_list1, edge_index_list2 = [], []

    x_pos1, x_pos2 = [], []
    x_f1, x_f2 = [], []
    for s, data1, data2 in zip(ptr, data_list1, data_list2):
        N1, N2 = data1.x.shape[0], data2.x.shape[0]
        x_pos1.append(torch.arange(s, s + N1, device=device))
        x_pos2.append(torch.arange(s, s + N2, device=device))
        x_f1.append(data1.x)
        x_f2.append(data2.x)
        edge_index_list1.append(data1.edge_index + s)
        edge_index_list2.append(data2.edge_index + s)
        edge_attr_list1.append(data1.edge_attr)
        edge_attr_list2.append(data2.edge_attr)

    ## deal with node features
    x_pos1 = torch.cat(x_pos1, dim=0)
    x_pos2 = torch.cat(x_pos2, dim=0)
    x_f1 = torch.cat(x_f1, dim=0)  # shape = [bach_N1, 2]
    x_f2 = torch.cat(x_f2, dim=0)  # shape = [bach_N2, 2]

    self_loop_weight = torch.zeros((ptr[-1],), device=device)
    self_loop_weight[x_pos1] += lamb
    self_loop_weight[x_pos2] += 1 - lamb

    mix_x = torch.zeros((ptr[-1], DN), device=device)
    mix_x[x_pos1, x_f1[:, 0]] += lamb
    mix_x[x_pos1, x_f1[:, 1] + num_atom_type] += lamb
    mix_x[x_pos2, x_f2[:, 0]] += 1 - lamb
    mix_x[x_pos2, x_f2[:, 1] + num_atom_type] += 1 - lamb

    edge_index1 = torch.cat(edge_index_list1, dim=1)  # shape = [2, batch_E1]
    edge_index2 = torch.cat(edge_index_list2, dim=1)  # shape = [2, batch_E2]

    ## deal with the edge attributes
    edge_attr_list1 = torch.cat(edge_attr_list1, dim=0)  # shape = [batch_E1, 2]
    edge_attr_list2 = torch.cat(edge_attr_list2, dim=0)  # shape = [batch_E2, 2]
    edge_attr1 = torch.cat(
        (F.one_hot(edge_attr_list1[:, 0], num_bond_type), F.one_hot(edge_attr_list1[:, 1], num_bond_direction)), dim=1)

    edge_attr2 = torch.cat(
        (F.one_hot(edge_attr_list2[:, 0], num_bond_type), F.one_hot(edge_attr_list2[:, 1], num_bond_direction)), dim=1)

    ## concatenate the edge weights
    edge_attr1 = torch.cat((edge_attr1, torch.ones((edge_attr1.shape[0], 1), device=device)), dim=1)
    edge_attr2 = torch.cat((edge_attr2, torch.ones((edge_attr2.shape[0], 1), device=device)), dim=1)

    adj_attr1 = torch.sparse_coo_tensor(edge_index1, edge_attr1, (ptr[-1], ptr[-1], DE + 1))
    adj_attr2 = torch.sparse_coo_tensor(edge_index2, edge_attr2, (ptr[-1], ptr[-1], DE + 1))
    mix_adj_attr = lamb * adj_attr1 + (1 - lamb) * adj_attr2
    mix_adj_attr = mix_adj_attr.coalesce()
    mix_edge_index = mix_adj_attr._indices()
    mix_edge_attr = mix_adj_attr._values()

    ## find self loop weights. The weights for the self-looped nodes are lower.
    batch = Batch(x=mix_x, edge_index=mix_edge_index, edge_attr=mix_edge_attr[:, :-1], edge_weight=mix_edge_attr[:, -1],
                  ptr=torch.tensor(ptr, device=device), batch=batch_index, self_loop_weight=self_loop_weight)

    batch.__num_graphs__ = len(data_list1)
    return batch.contiguous()


@torch.no_grad()
def mix_data_list_add_self_loop(data_list1, data_list2, lamb, use_rand_perm):
    device = data_list1[0].x.device
    DN = num_atom_type + num_chirality_tag
    DE = num_bond_type + num_bond_direction

    if use_rand_perm:
        data_list1 = [data.clone() for data in data_list1]
        data_list2 = [data.clone() for data in data_list2]
        for i in range(len(data_list1)):
            if data_list1[i].num_nodes > data_list2[i].num_nodes:
                data_list1[i] = rand_permute(data_list1[i])
            else:
                data_list2[i] = rand_permute(data_list2[i])

    ptr = [0]
    batch_index = []
    for i, (data1, data2) in enumerate(zip(data_list1, data_list2)):
        N = max(data1.x.shape[0], data2.x.shape[0])
        ptr.append(ptr[-1] + N)
        batch_index.append(torch.full((N,), i, dtype=torch.long, device=device))
    batch_index = torch.cat(batch_index, dim=0)

    edge_attr_list1, edge_attr_list2 = [], []
    edge_index_list1, edge_index_list2 = [], []

    x_pos1, x_pos2 = [], []
    x_f1, x_f2 = [], []
    for s, data1, data2 in zip(ptr, data_list1, data_list2):
        N1, N2 = data1.x.shape[0], data2.x.shape[0]
        x_pos1.append(torch.arange(s, s + N1, device=device))
        x_pos2.append(torch.arange(s, s + N2, device=device))
        x_f1.append(data1.x)
        x_f2.append(data2.x)
        edge_index_list1.append(data1.edge_index + s)
        edge_index_list2.append(data2.edge_index + s)
        edge_attr_list1.append(data1.edge_attr)
        edge_attr_list2.append(data2.edge_attr)

    ## deal with node features
    x_pos1 = torch.cat(x_pos1, dim=0)
    x_pos2 = torch.cat(x_pos2, dim=0)
    x_f1 = torch.cat(x_f1, dim=0)  # shape = [bach_N1, 2]
    x_f2 = torch.cat(x_f2, dim=0)  # shape = [bach_N2, 2]

    mix_x = torch.zeros((ptr[-1], DN), device=device)
    mix_x[x_pos1, x_f1[:, 0]] += lamb
    mix_x[x_pos1, x_f1[:, 1] + num_atom_type] += lamb
    mix_x[x_pos2, x_f2[:, 0]] += 1 - lamb
    mix_x[x_pos2, x_f2[:, 1] + num_atom_type] += 1 - lamb

    node_weights = torch.zeros((ptr[-1],), device=device)
    node_weights[x_pos1] += lamb
    node_weights[x_pos2] += 1 - lamb

    edge_index1 = torch.cat(edge_index_list1, dim=1)  # shape = [2, batch_E1]
    edge_index2 = torch.cat(edge_index_list2, dim=1)  # shape = [2, batch_E2]
    edge_attr1 = torch.cat(edge_attr_list1, dim=0)  # shape = [batch_E1, 2]
    edge_attr2 = torch.cat(edge_attr_list2, dim=0)  # shape = [batch_E2, 2]

    zero_selfloop_for_dummy_nodes = False
    if zero_selfloop_for_dummy_nodes:
        # deal with self loop
        ## add self loop index
        self_loop_index1 = x_pos1.reshape(1, -1).repeat(2, 1)  # shape = [2, N1]
        self_loop_index2 = x_pos2.reshape(1, -1).repeat(2, 1)  # shape = [2, N1]
        edge_index1 = torch.cat((edge_index1, self_loop_index1), dim=1)  # shape = [2, batch_E1 + N1]
        edge_index2 = torch.cat((edge_index2, self_loop_index2), dim=1)  # shape = [2, batch_E2 + N2]
        ## add self loop attributes
        self_loop_attr1 = edge_attr1.new_zeros((x_pos1.shape[0], 2))
        self_loop_attr1[:, 0] = 4  # the default attribute of self loop
        edge_attr1 = torch.cat((edge_attr1, self_loop_attr1), dim=0)
        self_loop_attr2 = edge_attr2.new_zeros((x_pos2.shape[0], 2))
        self_loop_attr2[:, 0] = 4  # the default attribute of self loop
        edge_attr2 = torch.cat((edge_attr2, self_loop_attr2), dim=0)
    else:
        self_loop_index = torch.arange(ptr[-1], device=device).reshape(1, -1).repeat(2, 1)
        edge_index1 = torch.cat((edge_index1, self_loop_index), dim=1)  # shape = [2, batch_E1 + N1]
        edge_index2 = torch.cat((edge_index2, self_loop_index), dim=1)  # shape = [2, batch_E2 + N2]
        self_loop_attr = edge_attr1.new_zeros((ptr[-1], 2))
        self_loop_attr[:, 0] = 4
        edge_attr1 = torch.cat((edge_attr1, self_loop_attr), dim=0)
        edge_attr2 = torch.cat((edge_attr2, self_loop_attr), dim=0)

    # deal with the edge attributes
    edge_attr1 = torch.cat(
        (F.one_hot(edge_attr1[:, 0], num_bond_type), F.one_hot(edge_attr1[:, 1], num_bond_direction)), dim=1)
    edge_attr2 = torch.cat(
        (F.one_hot(edge_attr2[:, 0], num_bond_type), F.one_hot(edge_attr2[:, 1], num_bond_direction)), dim=1)
    # concatenate the edge weights
    edge_attr1 = torch.cat((edge_attr1, torch.ones((edge_attr1.shape[0], 1), device=device)), dim=1)
    edge_attr2 = torch.cat((edge_attr2, torch.ones((edge_attr2.shape[0], 1), device=device)), dim=1)

    adj_attr1 = torch.sparse_coo_tensor(edge_index1, edge_attr1, (ptr[-1], ptr[-1], DE + 1))
    adj_attr2 = torch.sparse_coo_tensor(edge_index2, edge_attr2, (ptr[-1], ptr[-1], DE + 1))
    mix_adj_attr = lamb * adj_attr1 + (1 - lamb) * adj_attr2
    mix_adj_attr = mix_adj_attr.coalesce()
    mix_edge_index = mix_adj_attr._indices()
    mix_edge_attr = mix_adj_attr._values()

    ## find self loop weights. The weights for the self-looped nodes are lower.
    batch = Batch(x=mix_x, edge_index=mix_edge_index, edge_attr=mix_edge_attr[:, :-1], edge_weight=mix_edge_attr[:, -1],
                  ptr=torch.tensor(ptr, device=device), batch=batch_index, node_weights=node_weights)

    batch.__num_graphs__ = len(data_list1)
    return batch.contiguous()


@torch.no_grad()
def mix_data_list_add_self_loop_arraylamb(data_list1, data_list2, lamb, use_rand_perm, add_self_loop=True):
    '''
    lamb, shape = [B,]. different lamb for mixing different graphs
    '''
    device = data_list1[0].x.device
    DN = num_atom_type + num_chirality_tag
    DE = num_bond_type + num_bond_direction

    if use_rand_perm:
        data_list1 = [data.clone() for data in data_list1]
        data_list2 = [data.clone() for data in data_list2]
        for i in range(len(data_list1)):
            if data_list1[i].num_nodes > data_list2[i].num_nodes:
                data_list1[i] = rand_permute(data_list1[i])
            else:
                data_list2[i] = rand_permute(data_list2[i])

    ptr = [0]
    batch_index = []
    for i, (data1, data2) in enumerate(zip(data_list1, data_list2)):
        N = max(data1.x.shape[0], data2.x.shape[0])
        ptr.append(ptr[-1] + N)
        batch_index.append(torch.full((N,), i, dtype=torch.long, device=device))
    batch_index = torch.cat(batch_index, dim=0)

    edge_attr_list1, edge_attr_list2 = [], []
    edge_index_list1, edge_index_list2 = [], []

    x_pos1, x_pos2 = [], []
    x_f1, x_f2 = [], []
    x_ls1, x_ls2 = [], []
    e_ls1, e_ls2 = [], []
    for l, s, data1, data2 in zip(lamb, ptr, data_list1, data_list2):
        N1, N2 = data1.x.shape[0], data2.x.shape[0]
        x_pos1.append(torch.arange(s, s + N1, device=device))
        x_pos2.append(torch.arange(s, s + N2, device=device))
        x_f1.append(data1.x)
        x_f2.append(data2.x)
        edge_index_list1.append(data1.edge_index + s)
        edge_index_list2.append(data2.edge_index + s)
        edge_attr_list1.append(data1.edge_attr)
        edge_attr_list2.append(data2.edge_attr)

        x_ls1.append(torch.full_like(x_pos1[-1], l, dtype=torch.float))
        x_ls2.append(torch.full_like(x_pos2[-1], 1 - l, dtype=torch.float))
        e_ls1.append(torch.full((edge_attr_list1[-1].shape[0],), l))
        e_ls2.append(torch.full((edge_attr_list2[-1].shape[0],), 1 - l))

    ## deal with node features
    x_pos1 = torch.cat(x_pos1, dim=0)
    x_pos2 = torch.cat(x_pos2, dim=0)
    x_f1 = torch.cat(x_f1, dim=0)  # shape = [bach_N1, 2]
    x_f2 = torch.cat(x_f2, dim=0)  # shape = [bach_N2, 2]
    x_ls1 = torch.cat(x_ls1, dim=0)
    x_ls2 = torch.cat(x_ls2, dim=0)
    e_ls1 = torch.cat(e_ls1, dim=0)
    e_ls2 = torch.cat(e_ls2, dim=0)

    mix_x = torch.zeros((ptr[-1], DN), device=device)
    mix_x[x_pos1, x_f1[:, 0]] += x_ls1
    mix_x[x_pos1, x_f1[:, 1] + num_atom_type] += x_ls1
    mix_x[x_pos2, x_f2[:, 0]] += x_ls2
    mix_x[x_pos2, x_f2[:, 1] + num_atom_type] += x_ls2

    node_weights = torch.zeros((ptr[-1],), device=device)
    node_weights[x_pos1] += x_ls1
    node_weights[x_pos2] += x_ls2

    edge_index1 = torch.cat(edge_index_list1, dim=1)  # shape = [2, batch_E1]
    edge_index2 = torch.cat(edge_index_list2, dim=1)  # shape = [2, batch_E2]
    edge_attr1 = torch.cat(edge_attr_list1, dim=0)  # shape = [batch_E1, 2]
    edge_attr2 = torch.cat(edge_attr_list2, dim=0)  # shape = [batch_E2, 2]

    if add_self_loop:
        # add self loop attributes. The dummy nodes also have full weighted self loop edge attributes.
        self_loop_index = torch.arange(ptr[-1], device=device).reshape(1, -1).repeat(2, 1)
        edge_index1 = torch.cat((edge_index1, self_loop_index), dim=1)  # shape = [2, batch_E1 + N1]
        edge_index2 = torch.cat((edge_index2, self_loop_index), dim=1)  # shape = [2, batch_E2 + N2]
        self_loop_attr = edge_attr1.new_zeros((ptr[-1], 2))
        self_loop_attr[:, 0] = 4
        edge_attr1 = torch.cat((edge_attr1, self_loop_attr), dim=0)
        edge_attr2 = torch.cat((edge_attr2, self_loop_attr), dim=0)
        e_ls1 = torch.cat((e_ls1, lamb[batch_index]), dim=0)  # concatenate the lamb of self loops
        e_ls2 = torch.cat((e_ls2, 1 - lamb[batch_index]), dim=0)  # concatenate the lamb of self loops

    # deal with the edge attributes
    edge_attr1 = torch.cat(
        (F.one_hot(edge_attr1[:, 0], num_bond_type), F.one_hot(edge_attr1[:, 1], num_bond_direction)), dim=1)
    edge_attr2 = torch.cat(
        (F.one_hot(edge_attr2[:, 0], num_bond_type), F.one_hot(edge_attr2[:, 1], num_bond_direction)), dim=1)

    # concatenate the edge weights
    edge_attr1 = torch.cat((edge_attr1, torch.ones((edge_attr1.shape[0], 1), device=device)), dim=1)
    edge_attr2 = torch.cat((edge_attr2, torch.ones((edge_attr2.shape[0], 1), device=device)), dim=1)

    # scale the edge_attr
    edge_attr1 = edge_attr1 * e_ls1.view(-1, 1)
    edge_attr2 = edge_attr2 * e_ls2.view(-1, 1)

    adj_attr1 = torch.sparse_coo_tensor(edge_index1, edge_attr1, (ptr[-1], ptr[-1], DE + 1))
    adj_attr2 = torch.sparse_coo_tensor(edge_index2, edge_attr2, (ptr[-1], ptr[-1], DE + 1))
    mix_adj_attr = adj_attr1 + adj_attr2
    mix_adj_attr = mix_adj_attr.coalesce()
    mix_edge_index = mix_adj_attr._indices()
    mix_edge_attr = mix_adj_attr._values()

    ## find self loop weights. The weights for the self-looped nodes are lower.
    batch = Batch(x=mix_x, edge_index=mix_edge_index, edge_attr=mix_edge_attr[:, :-1], edge_weight=mix_edge_attr[:, -1],
                  ptr=torch.tensor(ptr, device=device), batch=batch_index, node_weights=node_weights)

    batch.__num_graphs__ = len(data_list1)
    return batch.contiguous()


class MixDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 beta1,
                 beta2,
                 use_rand_perm,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 exclude_keys=[],
                 add_self_loop=False,
                 rebatch=True,
                 **kwargs
                 ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.beta1 = beta1
        self.beta2 = beta2
        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super(MixDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=MixChemCollater(beta1, beta2, use_rand_perm, follow_batch, exclude_keys, add_self_loop, rebatch),
                             **kwargs)
