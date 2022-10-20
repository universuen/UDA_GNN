import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import utils
from loader import MoleculeDataset
from splitters import scaffold_split
from model import GNN



def load_dataset_chem(args):
    dataset = MoleculeDataset(
        "./dataset/" + args.tuning_dataset, dataset=args.tuning_dataset)
    print(dataset)
    if args.split == "scaffold":
        smiles_list = pd.read_csv(
            './dataset/' + args.tuning_dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")
    return train_dataset, valid_dataset, test_dataset


def load_chem_gnn_model(args):
    print("Loading model from %s." % args.model_file)
    model_state_dict = torch.load(
        args.model_file, map_location=lambda storage, loc: storage)
    
    gnn = GNN(args.num_layer, args.emb_dim, args.JK,
                args.dropout_ratio, gnn_type=args.gnn_type)
    gnn.load_state_dict(model_state_dict)
    return gnn


@utils.timeit
def train_chem(args, model, device, loader, optimizer, scheduler, log_file):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    loss_meter = utils.AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape) #.to(torch.float64)
        # Whether y is non-null or not.
        is_valid = torch.abs(y) > 0  # shape = [N, C]

        # Loss matrix
        loss_mat = criterion(pred, (y+1)/2)  # shape = [N, C]
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        loss_meter.update(float(loss), pred.shape[0])

    if scheduler:
        scheduler.step()
    utils.write_log('Avg loss {:.4f}'.format(loss_meter.avg), log_file)
    return loss_meter.avg


@utils.timeit
@torch.no_grad()
def eval_chem(args, model, loader):
    model.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        batch = batch.to(args.device)
        pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score(
                (y_true[is_valid, i] + 1)/2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list))/y_true.shape[1]))
    mean_roc = sum(roc_list) / len(roc_list)
    return mean_roc


class GraphClf(nn.Module):
    def __init__(self, gnn, emb_dim, num_tasks):
        super(GraphClf, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.linear = nn.Linear(emb_dim, num_tasks)

    def forward(self, data):
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(data)
        graph_rep = self.pool(node_representation, data.batch)
        return self.linear(graph_rep)



def get_eval_loader(dataset, args):
    if len(dataset) > 2048:
        return DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers, pin_memory=args.num_workers>2)
    else:
        full_batch_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        for full_batch in full_batch_loader:
            pass
        full_batch = full_batch.to(args.device)
        return [full_batch]


def tuning_chem(args, train_dataset, valid_dataset, test_dataset, running_seeds):
    args.use_schedule = args.tuning_dataset in {'bace',}
    train_eval_loader = get_eval_loader(train_dataset, args)
    val_loader = get_eval_loader(valid_dataset, args)
    test_loader = get_eval_loader(test_dataset, args)
    for runseed in tqdm(running_seeds):
        args.runseed = runseed
        print('Training For Running Seed {}'.format(args.runseed))
        utils.set_seed(args.runseed, args.device)
        pin_memory = args.num_workers > 0
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)

        # set up model
        gnn = load_chem_gnn_model(args)
        model = GraphClf(gnn, args.emb_dim, args.num_tasks)
        

        model.to(args.device)
        # set up optimizer
        lr = args.lr if args.tuning_dataset != 'muv' else 1e-4

        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.decay)
        scheduler = StepLR(optimizer, step_size=30,
                           gamma=0.3) if args.use_schedule else None
        print(optimizer)

        # naming tuning scheme
        tuning_scheme = 'linear'
        if args.scheme_prefix is not None:
            tuning_scheme = '{}_{}'.format(args.scheme_prefix, tuning_scheme)

        log_file = 'results/{}/{}_{}_log_{}.txt'.format(args.name, tuning_scheme, args.tuning_dataset, args.epochs)
        utils.write_log(str(args), log_file=log_file, print_=True)
        utils.write_log('Runseed {}; Epochs {}; WeightDecay {}; ModelFile {}.'.format(
            args.runseed, args.epochs, args.decay, args.model_file), log_file)

        train_roc_list = []
        val_roc_list = []
        test_roc_list = []

        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            train_chem(args, model, args.device, train_loader,
                       optimizer, scheduler, log_file)

            if (not args.skip_evaluation) or (epoch == args.epochs):
                val_roc = eval_chem(args, model, val_loader)
                val_roc_list.append(val_roc)
                test_roc = eval_chem(args, model, test_loader)
                test_roc_list.append(test_roc)

                if args.eval_train:
                    train_roc = eval_chem(args, model, train_eval_loader)
                    train_roc_list.append(train_roc)
                    utils.write_log('Epoch {}: {} {} {}'.format(
                        epoch, train_roc_list[-1], val_roc_list[-1], test_roc_list[-1]), log_file)
                else:
                    utils.write_log('Epoch {}: {} {}'.format(
                        epoch, val_roc_list[-1], test_roc_list[-1]), log_file)

        train_roc = eval_chem(args, model, train_eval_loader)
        utils.write_log('Final Epoch Train ROC: {}'.format(train_roc), log_file)
        result_path = 'results/{}/{}_{}.txt'.format(
            args.name, tuning_scheme, args.epochs)
        with open(result_path, 'a') as f:
            line = '{} {} {} {} {}\n'.format(
                args.tuning_dataset, args.model_file, args.runseed, val_roc_list[-1], test_roc_list[-1])
            f.write(line)



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default="species",
                        help='Bio dataset: Random or species split; Chem dataset: random or scaffold or random_scaffold.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataset loading')
    parser.add_argument('--use_schedule', action="store_true",
                        default=False, help='Use learning rate scheduler?')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--scheme_prefix', type=str,
                        default=None, help='The name for tuning logs.')
    parser.add_argument('--dataset', type=str, default="chem",
                        help='bio or chem. The domain of the pretrained model.')
    parser.add_argument('--tuning_dataset', type=str, default=None,
                        help='Used only for CHEM dataset. The dataset used for fine-tuning.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation to speed up training.')
    parser.add_argument('--eval_train', action='store_true',
                        help='Evaluate the training dataset or not.')
    # number of random seeds
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--double_precision', action='store_true', default=False)
    parser.add_argument('--graph_trans', action='store_true', default=False)
    args = parser.parse_args()
    

    if args.double_precision:
        # use double precision to ensure reproducibility
        torch.set_default_tensor_type(torch.DoubleTensor)

    running_seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    if not os.path.exists('results/{}'.format(args.name)):
        os.makedirs('results/{}'.format(args.name))

    args.device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    assert args.dataset == 'chem'
    datasets = ["bbbp", "tox21", "clintox",
                    "sider", "bace", "muv", "toxcast", "hiv"]
    num_task_dict = {'tox21': 12, 'hiv': 1, 'muv': 17, 'bace': 1,
                         'bbbp': 1, 'toxcast': 617, 'sider': 27, 'clintox': 2}
    if args.tuning_dataset is None:
        for dataset in datasets:
            args.tuning_dataset = dataset
            args.num_tasks = num_task_dict[dataset]
            train_dataset, valid_dataset, test_dataset = load_dataset_chem(args)
            tuning_chem(args, train_dataset, valid_dataset,
                        test_dataset, running_seeds)
    else:
        args.num_tasks = num_task_dict[args.tuning_dataset]
        train_dataset, valid_dataset, test_dataset = load_dataset_chem(args)
        tuning_chem(args, train_dataset, valid_dataset,
                    test_dataset, running_seeds)
    


if __name__ == "__main__":
    start = time.time()
    utils.print_time_info('Start Tuning')
    main()
    end = time.time()
    utils.print_time_info('End Tuning. Spend %d seconds' % (end - start))
