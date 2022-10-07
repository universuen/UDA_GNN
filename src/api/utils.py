import random

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score

import src
from src.original.splitters import scaffold_split
from src import config, api


def log_config_info(logger, config_cls: config.ConfigType, end: bool = True):
    logger.info(f'{config_cls.__name__:*^100}')
    for k, v in config_cls.to_dict().items():
        logger.info(f'{k}: {v}')
    if end:
        logger.info('*' * 100)


def log_all_config(logger):
    logger.info(f"{'General':*^100}")
    logger.info(f'config_name: {config.config_name}')
    logger.info(f'seed: {config.seed}')
    logger.info(f'device: {config.device}')
    for i in config.get_all_configs():
        log_config_info(logger, i, end=False)
    logger.info('*' * 100)


def set_debug_mode():
    print('**********DEBUG MODE IS ON!**********')
    config.config_name = 'debug'
    config.Pretraining.epochs = 1
    config.Pretraining.batch_size = 100
    config.Tuning.epochs = 1
    config.Logger.level = 'DEBUG'
    config.PretrainingLoader.num_workers = 0
    config.TuningLoader.num_workers = 0
    config.device = 'cuda:0'


def training_bar(epoch: int, total_epochs: int, **kwargs) -> str:
    content = f'epoch {epoch + 1} / {total_epochs}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    return content


def set_seed(seed: int = config.seed):
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
def eval_chem(model, loader) -> float:
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


def pretrain(model: src.types.PretrainingModel):
    logger = api.get_configured_logger(
        name=f'pretrain_{config.PretrainingDataset.dataset}'
    )
    log_all_config(logger)
    logger.info('Started Pretraining')

    logger.debug('Prepare')
    model.to(config.device)
    dataset = api.get_configured_pretraining_dataset()
    loader = api.get_configured_pretraining_loader(dataset)
    optimizer = torch.optim.Adam(model.parameters(), config.Pretraining.lr)
    loss_history = api.get_configured_history('pretraining_losses')

    logger.debug('Training loop')
    for e in range(config.Pretraining.epochs):
        for idx, (b1, b2, _) in enumerate(loader):
            torch.cuda.empty_cache()
            b1.to(config.device)
            b2.to(config.device)
            optimizer.zero_grad()
            loss = model(b1, b2)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            logger.debug(f'epoch: {e}, loss: {loss}, process: {(idx + 1) / len(loader)}')
        logger.info(training_bar(e, config.Pretraining.epochs, loss=loss_history.last_one))
        if (e + 1) % 20 == 0:
            models_dir = config.Paths.models / config.config_name
            models_dir.mkdir(exist_ok=True)
            torch.save(
                model.state_dict(),
                models_dir / f'pretraining_model_{config.PretrainingDataset.dataset}_{e + 1}.pt'
            )
            logger.info(
                f"model saved at {models_dir / f'pretraining_model_{config.PretrainingDataset.dataset}_{e + 1}.pt'}"
            )

    logger.debug('Save the final model')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        model.gnn.state_dict(),
        models_dir / f'pretraining_model_{config.PretrainingDataset.dataset}_final.pt'
    )
    loss_history.save()


def tune(dataset_name: str, gnn: src.types.GNNModel):
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Tuning')

    logger.debug('Prepare')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    # set up dataloaders
    tr_loader = get_eval_loader(tr_dataset)
    va_loader = get_eval_loader(va_dataset)
    te_loader = get_eval_loader(te_dataset)
    training_loader = api.get_configured_tuning_dataloader(tr_dataset)
    # set up classifying model, optimizer, and criterion
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
    ).to(config.device)
    optimizer = torch.optim.Adam(clf.parameters(), config.Tuning.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=30,
        gamma=0.3,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # prepare to record evaluations
    loss_history = api.get_configured_history(f'{dataset_name}_tuning_losses_{config.seed}')
    tr_auc_history = api.get_configured_history(f'{dataset_name}_tr_auc_{config.seed}')
    va_auc_history = api.get_configured_history(f'{dataset_name}_va_auc_{config.seed}')
    te_auc_history = api.get_configured_history(f'{dataset_name}_te_auc_{config.seed}')

    logger.debug('Training loop')
    for e in range(config.Tuning.epochs):
        clf.train()
        for idx, batch in enumerate(training_loader):
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
            logger.debug(f'epoch: {e}, loss: {loss}, process: {(idx + 1) / len(training_loader)}')
        tr_auc_history.append(eval_chem(clf, tr_loader))
        va_auc_history.append(eval_chem(clf, va_loader))
        te_auc_history.append(eval_chem(clf, te_loader))
        tr_auc_history.save()
        va_auc_history.save()
        te_auc_history.save()
        logger.info(
            training_bar(
                e,
                config.Tuning.epochs,
                loss=loss_history.last_one,
                tr_auc=tr_auc_history.last_one,
                va_auc=va_auc_history.last_one,
                te_auc=te_auc_history.last_one,
            )
        )
        if config.Tuning.use_lr_scheduler:
            lr_scheduler.step()
            logger.info(f'current LR: {lr_scheduler.get_last_lr()[0]}')
