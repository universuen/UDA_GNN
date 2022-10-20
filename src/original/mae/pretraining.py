import argparse
from functools import partial

from loader import MoleculeDataset
from dataloader import DataLoaderMaskingPred

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from tqdm import tqdm
from model import GNN, GNNDecoder



def sce_loss(x, y, alpha:float=1):
    x = F.normalize(x, p=2.0, dim=-1) # shape = [N, D]
    y = F.normalize(y, p=2.0, dim=-1) # shape = [N, D]
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def train_mae(args, model_list, loader, optimizer_list, epoch):
    if args.loss_fn == "sce":
        criterion = partial(sce_loss, alpha=args.alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    loss_meter = utils.AverageMeter()
    model, dec_pred_atoms, dec_pred_bonds = model_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimizer_list
    
    model.train()
    dec_pred_atoms.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    train_bar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(train_bar):
        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        batch = batch.to(args.device)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        
        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = dec_pred_atoms(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)
        if args.loss_fn == "sce":
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep)
            loss += criterion(pred_edge.double(), batch.mask_edge_label[:,0])

        loss.backward()
        optimizer_model.step()
        optimizer_dec_pred_atoms.step()
        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss = float(loss)
        loss_meter.update(loss, batch.num_graphs)
        train_bar.set_description('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count//1000, len(loader.dataset)//1000, loss_meter.avg, loss))
        if step % args.log_steps == 0:
            utils.write_log('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count // 1000, len(loader.dataset) // 1000, loss_meter.avg, loss), log_file=args.log_file, print_=False)
    
    utils.write_log('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count // 1000, len(loader.dataset) // 1000, loss_meter.avg, loss), log_file=args.log_file, print_=False)
    return loss_meter.avg



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    
    parser.add_argument("--name", type=str, help='Name for log dir')
    parser.add_argument("--save_epochs", type=int, default=20)
    parser.add_argument('--log_steps', type=int, default=100)
    args = parser.parse_args()
    print(args)

    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    utils.set_seed(0)
    log_dir = './results/{}'.format(args.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_file = os.path.join(log_dir, 'log.txt')
    utils.write_log(str(args), log_file=args.log_file, print_=True)
    
    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.mask_edge))
    dataset_name = args.dataset
    # set up dataset and transform function.
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)

    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    

    model = model.to(args.device)

    NUM_NODE_ATTR = 119 # + 3 
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(args.device)
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(args.device)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    ## disable selfloop in training. Because we will add that in dataloader
    for gnn in model.gnns:
        gnn.add_selfloop = False
    atom_pred_decoder.conv.add_selfloop = False

    model_list = [model, atom_pred_decoder, bond_pred_decoder] 

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)


    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds]
    for epoch in range(1, args.epochs+1):
        train_mae(args, model_list, loader, optimizer_list, epoch)
        if epoch % args.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))
        
    ## Save a final model
    torch.save(model.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))


if __name__ == "__main__":
    main()
