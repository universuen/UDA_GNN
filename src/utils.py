import random

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score

import config
import src
from src.original.splitters import scaffold_split
from src import logger as _logger


def set_debug_mode():
    print('**********DEBUG MODE IS ON!**********')
    config.config_name = 'debug'
    config.Pretraining.epochs = 1
    config.Tuning.epochs = 1
    config.Logger.level = 'DEBUG'
    config.PretrainingLoader.num_workers = 0


def training_bar(epoch: int, idx: int, total: int, **kwargs):
    content = f'epoch {epoch + 1}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    content = ' '.join([content, f'[progress:{(idx + 1) / total:0>6.2%}]'])
    return content


def set_seed(seed: int):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_eval_loader(
        dataset,
        num_workers=config.TuningLoader.num_workers,
):
    if len(dataset) > 2048:
        return DataLoader(
            dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=num_workers > 2,
        )
    else:
        full_batch_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        full_batch = next(iter(full_batch_loader))
        return [full_batch]


def split_dataset(dataset):
    smiles_list = pd.read_csv(
        config.Paths.datasets / config.TuningDataset.dataset / 'processed/smiles.csv',
        header=None,
    )[0].tolist()
    tr_dataset, va_dataset, te_dataset = scaffold_split(
        dataset,
        smiles_list,
        null_value=0,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
    )
    return tr_dataset, va_dataset, te_dataset


@torch.no_grad()
def eval_chem(model, loader):
    model.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        batch = batch.to(config.device)
        pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score(
                (y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))
    mean_roc = sum(roc_list) / len(roc_list)
    return mean_roc

def tune(dataset_name: str):
    logger = _logger.Logger(f'tune_{dataset_name}')
    logger.info('Started Tuning')
    gnn = src.model.gnn.EqvGNN()
    gnn.load_state_dict(
        torch.load(
            config.Paths.models / config.config_name / f'pretraining_model_final.pt',
            map_location=lambda storage, loc: storage,
        )
    )
    tr_dataset, va_dataset, te_dataset = split_dataset(
        src.dataset.MoleculeDataset(
            dataset=dataset_name
        )
    )
    tr_loader = get_eval_loader(tr_dataset)
    va_loader = get_eval_loader(va_dataset)
    te_loader = get_eval_loader(te_dataset)
    training_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=config.Tuning.batch_size,
        shuffle=True,
        num_workers=config.TuningLoader.num_workers,
        pin_memory=True,
    )
    clf = src.model.GraphClf(gnn).to(config.device)
    optimizer = torch.optim.Adam(clf.parameters(), config.Tuning.lr)
    logger.info(f'seed: {config.seed}')
    logger.log_config_info(config.Tuning)
    loss_history = src.History(f'{dataset_name}_tuning_losses_{config.seed}')
    tr_auc_history = src.History(f'{dataset_name}_tr_auc_{config.seed}')
    va_auc_history = src.History(f'{dataset_name}_va_auc_{config.seed}')
    te_auc_history = src.History(f'{dataset_name}_te_auc_{config.seed}')
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for e in range(config.Tuning.epochs):
        for idx, batch in enumerate(training_loader):
            torch.cuda.empty_cache()
            batch = batch.to(config.device)
            pred = clf(batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            is_valid = y ** 2 > 0  # shape = [N, C]
            loss_mat = criterion(pred, (y + 1) / 2)  # shape = [N, C]
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
            optimizer.zero_grad()
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            logger.info(training_bar(e, idx, len(training_loader), loss=loss))
        loss_history.save()
        # evaluation
        tr_auc_history.append(eval_chem(clf, tr_loader))
        va_auc_history.append(eval_chem(clf, va_loader))
        te_auc_history.append(eval_chem(clf, te_loader))
        tr_auc_history.save()
        va_auc_history.save()
        te_auc_history.save()
