from __future__ import annotations

import random

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

import src
import copy
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
    config.PretrainingDataset.dataset = 'bbbp'
    config.Tuning.epochs = 2
    config.Logger.level = 'DEBUG'
    config.PretrainingLoader.num_workers = 0
    config.TuningLoader.num_workers = 0
    config.device = 'cuda:0'
    config.loop_seeds = [0, 1]


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
    if config.Pretraining.use_dual_dataset:
        if config.Pretraining.dual_ds_version == 1:
            dataset = api.get_configured_dual_dataset()
        elif config.Pretraining.dual_ds_version == 2:
            dataset = api.get_configured_dual_dataset_v2()
        else:
            raise ValueError('Dual dataset version should be in {1, 2}')
    else:
        dataset = api.get_configured_pretraining_dataset()
    loader = api.get_configured_pretraining_loader(dataset)
    optimizer = torch.optim.Adam(model.parameters(), config.Pretraining.lr)
    loss_history = api.get_configured_history('pretraining_losses')

    logger.debug('Training loop')
    for e in range(config.Pretraining.epochs):
        for idx, (b1, b2, _) in enumerate(loader):
            # torch.cuda.empty_cache()
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


def tune(gnn: src.types.GNNModel):
    dataset_name = config.TuningDataset.dataset
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
    # set up classifier, optimizer, and criterion
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
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

    logger.debug('Save the final model')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        gnn.state_dict(),
        models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}.pt'
    )


def safe_mean(list_: list[src.types.Numeric]) -> src.types.Numeric:
    return 0 if len(list_) == 0 else round(sum(list_) / len(list_), 1)


def analyze_results(steps: list[int] = None):
    return analyze_results_by_ratio()
    # if steps is None:
    #     steps = list(range(10, 101, 10))
    # results = {
    #     k: {
    #         kk: []
    #         for kk in steps
    #     }
    #     for k in [*config.datasets, 'mean']
    # }
    #
    # for ds in config.datasets:
    #     for step in steps:
    #         for seed in config.loop_seeds:
    #             try:
    #                 history = api.get_configured_history(f'{ds}_te_auc_{seed}')
    #                 history.load()
    #                 results[ds][step].append(history[step - 1] * 100)
    #             except (FileNotFoundError, IndexError):
    #                 pass
    #         results[ds][step] = safe_mean(results[ds][step])
    #         results['mean'][step].append(results[ds][step])
    #
    # for step in steps:
    #     results['mean'][step] = safe_mean(results['mean'][step])
    #
    # results = pd.DataFrame.from_dict(results)
    # print(results)
    # results.to_excel(config.Paths.results / config.config_name / 'analyzed_results.xlsx')


def analyze_results_by_ratio(ratios: list[int] = None):
    config.datasets = [
        "bbbp",
        "tox21",
        "toxcast",
        "sider",
        "clintox",
        "muv",
        "hiv",
        "bace",
    ]

    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {
        k: {
            kk: []
            for kk in ratios
        }
        for k in [*config.datasets, 'mean']
    }

    for ds in config.datasets:
        for ratio in ratios:
            for seed in config.loop_seeds:
                try:
                    history = api.get_configured_history(f'{ds}_te_auc_{seed}')
                    history.load()
                    results[ds][ratio].append(history[int(len(history.values) * ratio) - 1] * 100)
                except (FileNotFoundError, IndexError):
                    pass
            results[ds][ratio] = safe_mean(results[ds][ratio])
            results['mean'][ratio].append(results[ds][ratio])

    for ratio in ratios:
        results['mean'][ratio] = safe_mean(results['mean'][ratio])

    results = pd.DataFrame.from_dict(results)
    print(results)
    results.to_excel(config.Paths.results / config.config_name / 'analyzed_results.xlsx')


def tune_with_prompt(gnn: src.types.GNNModel):
    # link the prompt to gnn
    if config.Tuning.use_node_prompt:
        gnn.node_prompts = nn.ModuleList(
            [src.model.NodePrompt().to(config.device) for _ in range(config.GNN.num_layer)]
        )
    if config.Tuning.use_edge_prompt:
        gnn.edge_prompt = src.model.EdgePrompt()
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Tuning')

    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    # set up dataloaders
    tr_loader = get_eval_loader(tr_dataset)
    va_loader = get_eval_loader(va_dataset)
    te_loader = get_eval_loader(te_dataset)
    training_loader = api.get_configured_tuning_dataloader(tr_dataset)
    # set up classifier, optimizer, and criterion
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)
    parameters = list(clf.linear.parameters())
    if config.Tuning.use_node_prompt:
        for prompt in gnn.node_prompts:
            parameters += list(prompt.parameters())
    if config.Tuning.use_edge_prompt:
        parameters += list(gnn.edge_prompt.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        config.Tuning.lr
    )
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

    for e in range(config.Tuning.epochs):
        clf.train()
        for idx, batch in enumerate(training_loader):
            batch = batch.to(config.device)
            # apply prompt
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

    logger.debug('Save the final model')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        gnn.state_dict(),
        models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}.pt'
    )


def tune_linear(gnn: src.types.GNNModel):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Tuning')

    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    # set up dataloaders
    tr_loader = get_eval_loader(tr_dataset)
    va_loader = get_eval_loader(va_dataset)
    te_loader = get_eval_loader(te_dataset)
    training_loader = api.get_configured_tuning_dataloader(tr_dataset)
    # set up classifier, optimizer, and criterion
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)
    optimizer = torch.optim.Adam(
        clf.linear.parameters(),
        config.Tuning.lr
    )
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

    for e in range(config.Tuning.epochs):
        clf.train()
        for idx, batch in enumerate(training_loader):
            batch = batch.to(config.device)
            # apply prompt
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

    logger.debug('Save the final model')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        gnn.state_dict(),
        models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}.pt'
    )


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def marginal_entropy_bce(outputs):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    ps = torch.sigmoid(outputs)  # shape = [N, C]
    avg_ps = ps.mean(dim=0)  # shape = [C]
    entropy = - (avg_ps * avg_ps.log() + (1 - avg_ps) * (1 - avg_ps).log()).mean()
    return entropy, None


def confidence_selection(outputs, ratio=0.5):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    N = outputs.shape[0]
    ps = torch.sigmoid(outputs)  # shape = [N, C]
    avg_entropy = - (ps * ps.log() + (1 - ps) * (1 - ps).log()).mean(1) # shape = [N]
    _, idx = torch.topk(avg_entropy, int(N * ratio), largest=False)
    outputs = outputs[idx]
    return outputs


def marginal_entropy_bce_v2(outputs):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    zeros = torch.zeros_like(outputs)
    outputs = torch.stack((outputs, zeros), dim=-1)  # shape = [N, C, 2]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # shape = [N, C, 2]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # shape = [C, 2]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1).mean()


def ttt_eval(clf_model, loader):
    clf_model.eval()
    optimizer = torch.optim.Adam(
            params=clf_model.parameters(),
            lr=config.Tuning.lr,
        )
    # back up
    clf_states = copy.deepcopy(clf_model.state_dict())
    optim_states = copy.deepcopy(optimizer.state_dict())
    
    y_true = []
    y_scores = []
    for data, augmentations in loader:
        data.to(config.device)
        augmentations.to(config.device)
        # adapt
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            # outputs = confidence_selection(outputs, 0.5)
            loss, _ = marginal_entropy_bce(outputs)
            loss.backward()
            optimizer.step()
        # test
        with torch.no_grad():
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        # reset
        clf_model.load_state_dict(clf_states)
        optimizer.load_state_dict(optim_states)

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


def test_time_tuning(gnn):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Tuning')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    tr_loader = get_eval_loader(tr_dataset)
    va_loader = get_eval_loader(va_dataset)
    te_loader = get_eval_loader(te_dataset)
    # transform to TTT dataset
    te_dataset = api.get_configured_ttt_dataset(te_dataset)
    te_ttt_loader = DataLoader(
        dataset=te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    training_loader = api.get_configured_tuning_dataloader(tr_dataset)
    # set up classifier, optimizer, and criterion
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)
    # optimizers
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
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')

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
        te_ttt_auc_history.append(ttt_eval(clf, te_ttt_loader))
        tr_auc_history.save()
        va_auc_history.save()
        te_auc_history.save()
        te_ttt_auc_history.save()
        logger.info(
            training_bar(
                e,
                config.Tuning.epochs,
                loss=loss_history.last_one,
                tr_auc=tr_auc_history.last_one,
                va_auc=va_auc_history.last_one,
                te_auc=te_auc_history.last_one,
                te_ttt_auc=te_ttt_auc_history.last_one,
            )
        )
        if config.Tuning.use_lr_scheduler:
            lr_scheduler.step()
            logger.info(f'current LR: {lr_scheduler.get_last_lr()[0]}')

    logger.debug('Save the final model')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        gnn.state_dict(),
        models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}.pt'
    )
