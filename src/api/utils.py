from __future__ import annotations

import random
from pathlib import Path
import copy

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch import nn
from sklearn.metrics import roc_auc_score

import src
from src.original.trans_bt.splitters import scaffold_split
from src import config, api
from src.original.trans_bt.loader import augment


def replace_bn():
    nn.BatchNorm1d = src.model.OneSampleBN


def replace_with_ssf():
    nn.Linear = src.model.SSLinear
    nn.BatchNorm1d = src.model.SSBatchNorm


def set_bn_prior(value: float):
    src.model.OneSampleBN.prior = value


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
    config.device = 'cpu'
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


def safe_mean(list_: list[src.types.Numeric], round_: int = 1) -> src.types.Numeric:
    return 0 if len(list_) == 0 else round(sum(list_) / len(list_), round_)


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
            mean = safe_mean(results[ds][ratio])
            std = round(float(np.std(results[ds][ratio])), 1)
            results[ds][ratio] = f'{mean}±{std}'
            results['mean'][ratio].append((mean, std))

    for ratio in ratios:
        means = [i for i, _ in results['mean'][ratio]]
        results['mean'][ratio] = f"{safe_mean(means)}"

    pd.options.display.max_columns = None
    results = pd.DataFrame.from_dict(results)
    print(results)
    results.to_excel(config.Paths.results / config.config_name / 'analyzed_results.xlsx')


def analyze_ttt_results_by_ratio(ratios: list[int] = None, item_name: str = 'te_auc'):
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
                    history = api.get_configured_history(f'{ds}_{item_name}_{seed}')
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
    results.to_excel(config.Paths.results / config.config_name / 'analyzed_ttt_results.xlsx')


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
    aug_pre = avg_ps.reshape(1, -1).detach()  # shape = [1, C]
    return entropy, aug_pre


def confidence_selection(outputs, ratio=0.5):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    N = outputs.shape[0]
    ps = torch.sigmoid(outputs)  # shape = [N, C]
    avg_entropy = - (ps * ps.log() + (1 - ps) * (1 - ps).log()).mean(1)  # shape = [N]
    _, idx = torch.topk(avg_entropy, int(N * ratio), largest=False)
    outputs = outputs[idx]
    return outputs


@torch.no_grad()
def confidence_selection_v2(outputs, ratio=0.5):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    outputs = outputs.detach()
    N = outputs.shape[0]
    ps = torch.sigmoid(outputs)  # shape = [N, C]
    entropy = - (ps * ps.log() + (1 - ps) * (1 - ps).log())  # shape = [N, C]
    _, idx = torch.topk(entropy, int(N * ratio), largest=False, dim=0)  # shape = [k, C]
    idx = idx.T  # shape = [C, k]
    mask = torch.zeros_like(outputs.T, dtype=torch.bool)  # shape = [C, N]
    mask.scatter_(1, idx, True)
    mask = mask.T  # shape = [N, C]
    return mask


def marginal_entropy_bce_v3(outputs, ratio=0.5):
    """
    outputs: shape = [N, C]. N is the number of augmentations, C is the number of binary classes
    """
    device = outputs.device
    mask = None
    if ratio < 1:
        mask = confidence_selection_v2(outputs, ratio)  # shape = [N, C]

    ## calculating logits
    zeros = torch.zeros_like(outputs)
    outputs = torch.stack((outputs, zeros), dim=-1)  # shape = [N, C, 2]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # shape = [N, C, 2]

    if mask is not None:
        ## mask high entropy entries
        logits = torch.where(mask.unsqueeze(-1), logits, torch.FloatTensor([float('-inf')]).to(device))
        avg_logits = logits.logsumexp(dim=0) - mask.sum(0).unsqueeze(-1).log()  # shape = [C, 2]
    else:
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # shape = [C, 2]

    ## for numerical stability
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    avg_ps = avg_logits[:, 0].detach().reshape(1, -1).exp()  # shape = []
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1).mean(), avg_ps


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
    avg_ps = avg_logits[:, 0].detach().reshape(1, -1).exp()
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1).mean(), avg_ps


def marginal_entropy_bce_batch(outputs, N):
    """
    outputs: shape = [B * N, C]. N is the number of augmentations, C is the number of binary classes, B is the batch size
    """
    C = outputs.shape[-1]
    zeros = torch.zeros_like(outputs)
    outputs = torch.stack((outputs, zeros), dim=-1)  # shape = [B * N, C, 2]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # shape = [B * N, C, 2]
    logits = logits.reshape(-1, N, C, 2)  # shape = [B, N, C, 2]
    avg_logits = logits.logsumexp(dim=1) - np.log(N)  # shape = [B, C, 2]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    avg_ps = avg_logits[:, :, 0].detach().reshape(-1, C).exp()
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1).mean(), avg_ps


def tent_loss(outputs):
    """
    outputs: shape = [B, C]. C is the number of binary classes, B is the batch size
    """
    zeros = torch.zeros_like(outputs)
    outputs = torch.stack((outputs, zeros), dim=-1)  # shape = [B, C, 2]
    entropy = -(outputs.softmax(-1) * outputs.log_softmax(-1)).sum(-1)  # shape = [B, C]
    loss = entropy.mean()
    return loss


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.eval()


def freeze_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def ttt_eval(clf_model, loader):
    set_bn_prior(config.OneSampleBN.strength / (1 + config.OneSampleBN.strength))

    def _evaluate(y_true, y_scores):
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
    y_aug_scores = []
    for data, augmentations in loader:
        if config.TestTimeTuning.aug == 'dropout' or config.TestTimeTuning.aug == 'featM':
            clf_model.train()
            freeze_bn(clf_model)

        data.to(config.device)
        augmentations.to(config.device)

        # adapt
        aug_pre_list = []
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            # loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss.backward()
            optimizer.step()
            aug_pre_list.append(aug_pre)
        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)
        # reset
        if config.OnlineLearning.is_enabled:
            pass
        else:
            clf_model.load_state_dict(clf_states)
            optimizer.load_state_dict(optim_states)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)
    set_bn_prior(1)
    return mean_roc, mean_aug_roc


def cl_ssf_eval(clf_model, loader):
    def _evaluate(y_true, y_scores):
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

    ss_parameters = list(clf_model.linear.parameters())
    for i in clf_model.modules():
        if type(i) in [src.model.SSLinear, src.model.SSBatchNorm]:
            ss_parameters.append(i.gamma)
            ss_parameters.append(i.beta)
        if config.OnlineLearning.optimize_tent_bn:
            if isinstance(i, nn.BatchNorm1d):
                ss_parameters.append(i.weight)
                ss_parameters.append(i.bias)
    if config.Tuning.use_node_prompt:
        for prompt in clf_model.gnn.node_prompts:
            ss_parameters += list(prompt.parameters())
    optimizer = torch.optim.Adam(
        params=ss_parameters,
        lr=config.Tuning.lr,
    )
    y_true = []
    y_scores = []
    y_aug_scores = []
    for data, augmentations in loader:
        clf_model.train()
        if config.TestTimeTuning.aug != 'dropout':
            freeze_dropout(clf_model)

        data.to(config.device)
        augmentations.to(config.device)
        # adapt
        aug_pre_list = []
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            loss, aug_pre = marginal_entropy_bce_batch(outputs, config.TestTimeTuning.num_augmentations)
            loss.backward()
            optimizer.step()
            aug_pre_list.append(aug_pre)
        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)

    return mean_roc, mean_aug_roc


def cl_ssf_eval_v2(clf_model, loader):
    set_bn_prior(config.OneSampleBN.strength / (1 + config.OneSampleBN.strength))
    if config.BatchRenormalization.is_enabled:
        src.model.TBR.enable = True

    def _evaluate(y_true, y_scores):
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

    ss_parameters = []
    # ss_parameters = list(clf_model.linear.parameters())
    for i in clf_model.modules():
        if type(i) in [src.model.SSLinear, src.model.SSBatchNorm]:
            ss_parameters.append(i.gamma)
            ss_parameters.append(i.beta)
        if isinstance(i, nn.BatchNorm1d):
            ss_parameters.append(i.weight)
            ss_parameters.append(i.bias)

    optimizer = torch.optim.Adam(
        params=ss_parameters,
        lr=config.Tuning.lr,
    )
    y_true = []
    y_scores = []

    for data, augmentations in loader:
        optimizer.zero_grad()
        ## training
        clf_model.train()

        data.to(config.device)
        augmentations.to(config.device)
        if True:
            freeze_dropout(clf_model)
            pred = clf_model(data)
            loss = tent_loss(pred)
        else:
            pred = clf_model(augmentations)
            if True:
                loss, aug_pre = marginal_entropy_bce_batch(pred, config.TestTimeTuning.num_augmentations)
            else:
                freeze_dropout(clf_model)
                loss = tent_loss(pred)
        if config.TestTimeTuning.tuning:
            loss.backward()
            optimizer.step()

        if True:
            ## evaluation
            with torch.no_grad():
                clf_model.eval()
                pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred.detach())

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    if config.BatchRenormalization.is_enabled:
        src.model.TBR.enable = False
    set_bn_prior(1)
    return mean_roc, 0


def cl_eval(clf_model, loader):
    def _evaluate(y_true, y_scores):
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

    optimizer = torch.optim.Adam(
        params=clf_model.parameters(),
        lr=config.Tuning.lr,
    )

    y_true = []
    y_scores = []
    y_aug_scores = []
    for data, augmentations in loader:
        clf_model.train()
        if config.TestTimeTuning.aug != 'dropout':
            freeze_dropout(clf_model)

        data.to(config.device)
        augmentations.to(config.device)
        # adapt
        aug_pre_list = []
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            loss, aug_pre = marginal_entropy_bce_batch(outputs, config.TestTimeTuning.num_augmentations)
            loss.backward()
            optimizer.step()
            aug_pre_list.append(aug_pre)
        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)

    return mean_roc, mean_aug_roc


def ttt_prompt_eval(clf_model, loader):
    set_bn_prior(config.OneSampleBN.strength / (1 + config.OneSampleBN.strength))

    def _evaluate(y_true, y_scores):
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

    clf_model.eval()
    parameters = list(clf_model.gnn.node_prompts.parameters())
    if config.TestTimeTuning.include_linear:
        parameters.extend(clf_model.linear.parameters())
    optimizer = torch.optim.Adam(
        params=parameters,
        lr=config.Tuning.lr,
    )

    # back up
    clf_model.remember_prompt()
    optim_states = copy.deepcopy(optimizer.state_dict())

    y_true = []
    y_scores = []
    y_aug_scores = []
    for data, augmentations in loader:
        if config.TestTimeTuning.aug == 'dropout':
            clf_model.train()
            freeze_bn(clf_model)

        data.to(config.device)
        augmentations.to(config.device)

        # adapt
        aug_pre_list = []
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss.backward()
            optimizer.step()
            aug_pre_list.append(aug_pre)
        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)

        # reset
        clf_model.reset_prompt()
        optimizer.load_state_dict(optim_states)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)
    set_bn_prior(1)
    return mean_roc, mean_aug_roc


def ttt_ssf_eval(clf_model, loader):
    set_bn_prior(config.OneSampleBN.strength / (1 + config.OneSampleBN.strength))

    def _evaluate(y_true, y_scores):
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

    clf_model.eval()
    # collect ss linear parameters
    ss_parameters = list(clf_model.linear.parameters())
    for i in clf_model.modules():
        if type(i) in [src.model.SSLinear, src.model.SSBatchNorm]:
            ss_parameters.append(i.gamma)
            ss_parameters.append(i.beta)
    if config.Tuning.use_node_prompt:
        for prompt in clf_model.gnn.node_prompts:
            ss_parameters += list(prompt.parameters())

    optimizer = torch.optim.Adam(
        params=ss_parameters,
        lr=config.Tuning.lr,
    )
    # back up
    clf_states = copy.deepcopy(clf_model.state_dict())
    optim_states = copy.deepcopy(optimizer.state_dict())

    y_true = []
    y_scores = []
    y_aug_scores = []
    for data, augmentations in loader:
        if config.TestTimeTuning.aug == 'dropout' or config.TestTimeTuning.aug == 'featM':
            clf_model.train()
            freeze_bn(clf_model)

        data.to(config.device)
        augmentations.to(config.device)

        # adapt
        aug_pre_list = []
        for _ in range(config.TestTimeTuning.num_iterations):
            optimizer.zero_grad()
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            # loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss.backward()
            optimizer.step()
            aug_pre_list.append(aug_pre)
        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            pred = clf_model(data)
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)
        # reset
        clf_model.load_state_dict(clf_states)
        optimizer.load_state_dict(optim_states)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)
    set_bn_prior(1)
    return mean_roc, mean_aug_roc


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
        num_workers=2,
        pin_memory=True,
    )
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
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')
    te_aug_auc_history = api.get_configured_history(f'{dataset_name}_te_aug_auc_{config.seed}')

    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)

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
        te_auc_history.append(eval_chem(clf, te_loader))
        va_auc_history.append(eval_chem(clf, va_loader))

        tr_auc_history.save()
        va_auc_history.save()
        te_auc_history.save()

        if (e + 1) % config.TestTimeTuning.eval_epoch == 0:
            if config.TestTimeTuning.add_prompts:
                ttt_auc, aug_auc = ttt_prompt_eval(clf, te_ttt_loader)
            elif config.SSF.is_enabled:
                ttt_auc, aug_auc = ttt_ssf_eval(clf, te_ttt_loader)
            else:
                ttt_auc, aug_auc = ttt_eval(clf, te_ttt_loader)
            te_ttt_auc_history.append(ttt_auc)
            te_aug_auc_history.append(aug_auc)

            te_ttt_auc_history.save()
            te_aug_auc_history.save()
            logger.info(
                training_bar(
                    e,
                    config.Tuning.epochs,
                    te_auc=te_auc_history.last_one,
                    te_ttt_auc=te_ttt_auc_history.last_one,
                    ttt_impr=te_ttt_auc_history.last_one - te_auc_history.last_one,
                    te_aug_auc=te_aug_auc_history.last_one,
                    aug_impr=te_aug_auc_history.last_one - te_auc_history.last_one,
                )
            )
        else:
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


def tune_and_save_models(gnn):
    if config.TestTimeTuning.add_prompts:
        gnn.node_prompts = nn.ModuleList(
            [src.model.NodePrompt().to(config.device) for _ in range(config.GNN.num_layer)]
        )
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

    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)

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

        if (e + 1) % config.TestTimeTuning.save_epoch == 0:
            logger.debug(f'Save the {e + 1} epoch model.')
            torch.save(
                clf.state_dict(),
                models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}_e{e + 1}.pt'
            )


def cl_presaved_models(gnn):
    return ol_presaved_models(gnn)


def ol_presaved_models(gnn):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Test Time Tuning')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    if config.OnlineLearning.use_shuffle:
        te_dataset = te_dataset.shuffle()
    te_loader = get_eval_loader(te_dataset)
    # transform to TTT dataset
    # te_dataset = api.get_configured_ttt_dataset(te_dataset)
    te_cl_loader = src.loader.CLLoader(
        dataset=te_dataset,
        aug='none' if config.TestTimeTuning.aug in ['dropout', 'featM'] else config.TestTimeTuning.aug,
        aug_ratio=config.TestTimeTuning.aug_ratio,
        num_augs=config.TestTimeTuning.num_augmentations,
        batch_size=config.Tuning.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # prepare to record evaluations
    te_auc_history = api.get_configured_history(f'{dataset_name}_te_auc_{config.seed}')
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')
    te_aug_auc_history = api.get_configured_history(f'{dataset_name}_te_aug_auc_{config.seed}')

    logger.debug('Start loading models and evaluation.')
    gnn_states_backup = copy.deepcopy(gnn.state_dict())
    for e in range(9, config.Tuning.epochs + 1, config.TestTimeTuning.save_epoch):
        model_path = Path(
            config.TestTimeTuning.presaved_model_path + f'/tuning_model_{config.TuningDataset.dataset}_{config.seed}_e{e + 1}.pt')
        if not model_path.exists():
            logger.info(f'{model_path} does not exists. Existing evaluation of seed {config.seed}')
            input()
            break
        gnn.load_state_dict(gnn_states_backup, strict=False)

        if config.TestTimeTuning.add_prompts:
            gnn.node_prompts = nn.ModuleList(
                [
                    src.model.NodePrompt(enable_ssf=config.SSF.is_enabled).to(config.device)
                    for _ in range(config.GNN.num_layer)
                ]
            )
        clf = src.model.GraphClf(
            gnn=gnn,
            dataset=config.TuningDataset.dataset,
            use_graph_trans=config.Pretraining.use_graph_trans,
        ).to(config.device)

        clf.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

        if config.OnlineLearning.enable_tent_bn:
            for m in clf.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        te_auc_history.append(eval_chem(clf, te_loader))
        if config.SSF.is_enabled:
            ttt_auc, aug_auc = cl_ssf_eval(clf, te_cl_loader)
        else:
            ttt_auc, aug_auc = cl_eval(clf, te_cl_loader)
        te_ttt_auc_history.append(ttt_auc)
        te_aug_auc_history.append(aug_auc)

        te_auc_history.save()
        te_ttt_auc_history.save()
        te_aug_auc_history.save()
        logger.info(
            training_bar(
                e,
                config.Tuning.epochs,
                te_auc=te_auc_history.last_one,
                te_ttt_auc=te_ttt_auc_history.last_one,
                ttt_impr=te_ttt_auc_history.last_one - te_auc_history.last_one,
                te_aug_auc=te_aug_auc_history.last_one,
                aug_impr=te_aug_auc_history.last_one - te_auc_history.last_one,
            )
        )


def ol_presaved_models_v2(gnn):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    # log_all_config(logger)
    logger.info('Started Test Time Tuning')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    te_dataset = te_dataset.shuffle()
    te_loader = get_eval_loader(te_dataset)
    # transform to TTT dataset
    # te_dataset = api.get_configured_ttt_dataset(te_dataset)
    te_cl_loader = src.loader.CLLoader(
        dataset=te_dataset,
        aug='none' if config.TestTimeTuning.aug in ['dropout', 'featM'] else config.TestTimeTuning.aug,
        aug_ratio=config.TestTimeTuning.aug_ratio,
        num_augs=config.TestTimeTuning.num_augmentations,
        batch_size=config.Tuning.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # prepare to record evaluations
    te_auc_history = api.get_configured_history(f'{dataset_name}_te_auc_{config.seed}')
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')
    te_aug_auc_history = api.get_configured_history(f'{dataset_name}_te_aug_auc_{config.seed}')

    logger.debug('Start loading models and evaluation.')
    gnn_states_backup = copy.deepcopy(gnn.state_dict())

    for e in range(9, config.Tuning.epochs + 1, config.TestTimeTuning.save_epoch):
        model_path = Path(
            config.TestTimeTuning.presaved_model_path + f'/tuning_model_{config.TuningDataset.dataset}_{config.seed}_e{e + 1}.pt')
        if not model_path.exists():
            logger.info(f'{model_path} does not exists. Existing evaluation of seed {config.seed}')
            input()
            break
        gnn.load_state_dict(gnn_states_backup, strict=False)
        if config.TestTimeTuning.add_prompts:
            gnn.node_prompts = nn.ModuleList(
                [
                    src.model.NodePrompt(enable_ssf=config.SSF.is_enabled).to(config.device)
                    for _ in range(config.GNN.num_layer)
                ]
            )

        clf = src.model.GraphClf(
            gnn=gnn,
            dataset=config.TuningDataset.dataset,
            use_graph_trans=config.Pretraining.use_graph_trans,
        ).to(config.device)

        clf.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        te_auc_history.append(eval_chem(clf, te_loader))

        ttt_auc, aug_auc = cl_ssf_eval_v2(clf, te_cl_loader)

        te_ttt_auc_history.append(ttt_auc)
        te_aug_auc_history.append(aug_auc)

        te_auc_history.save()
        te_ttt_auc_history.save()
        te_aug_auc_history.save()
        logger.info(
            training_bar(
                e,
                config.Tuning.epochs,
                te_auc=te_auc_history.last_one,
                te_ttt_auc=te_ttt_auc_history.last_one,
                ttt_impr=te_ttt_auc_history.last_one - te_auc_history.last_one,
                te_aug_auc=te_aug_auc_history.last_one,
                aug_impr=te_aug_auc_history.last_one - te_auc_history.last_one,
            )
        )


def test_time_tuning_presaved_models(gnn):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Test Time Tuning')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    te_loader = get_eval_loader(te_dataset)
    # transform to TTT dataset
    te_dataset = api.get_configured_ttt_dataset(te_dataset)
    if config.OnlineLearning.use_shuffle:
        te_dataset = te_dataset.shuffle()
    te_ttt_loader = DataLoader(
        dataset=te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # set up classifier, optimizer, and criterion

    # optimizers
    # prepare to record evaluations
    te_auc_history = api.get_configured_history(f'{dataset_name}_te_auc_{config.seed}')
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')
    te_aug_auc_history = api.get_configured_history(f'{dataset_name}_te_aug_auc_{config.seed}')

    logger.debug('Start loading models and evaluation.')
    gnn_states_backup = copy.deepcopy(gnn.state_dict())
    for e in range(9, config.Tuning.epochs + 1, config.TestTimeTuning.save_epoch):
        model_path = Path(
            config.TestTimeTuning.presaved_model_path + f'/tuning_model_{config.TuningDataset.dataset}_{config.seed}_e{e + 1}.pt')
        if not model_path.exists():
            logger.info(f'{model_path} does not exists. Existing evaluation of seed {config.seed}')
            input()
            break
        gnn = api.get_configured_gnn()
        gnn.load_state_dict(gnn_states_backup, strict=False)
        clf = src.model.GraphClf(
            gnn=gnn,
            dataset=config.TuningDataset.dataset,
            use_graph_trans=config.Pretraining.use_graph_trans,
        ).to(config.device)
        clf.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        te_auc_history.append(eval_chem(clf, te_loader))

        if config.TestTimeTuning.add_prompts:
            if config.AdvAug.is_enabled:
                gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.NodePromptPtb(
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                            batch_size=config.TestTimeTuning.num_augmentations,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )
            else:
                gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.NodePrompt(
                            enable_ssf=config.Prompt.enable_ssf or config.SSF.is_enabled,
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )

        if config.Prompt.use_node_wise_prompt:
            ttt_auc, aug_auc = api.node_wise_adv_eval(clf, te_ttt_loader)
        elif config.AdvAug.is_enabled:
            ttt_auc, aug_auc = api.adv_eval(clf, te_ttt_loader)
        elif config.SSF.is_enabled:
            ttt_auc, aug_auc = ttt_ssf_eval(clf, te_ttt_loader)
        elif config.TestTimeTuning.add_prompts:
            ttt_auc, aug_auc = ttt_prompt_eval(clf, te_ttt_loader)
        else:
            ttt_auc, aug_auc = ttt_eval(clf, te_ttt_loader)

        te_ttt_auc_history.append(ttt_auc)
        te_aug_auc_history.append(aug_auc)

        te_auc_history.save()
        te_ttt_auc_history.save()
        te_aug_auc_history.save()
        logger.info(
            training_bar(
                e,
                config.Tuning.epochs,
                te_auc=te_auc_history.last_one,
                te_ttt_auc=te_ttt_auc_history.last_one,
                ttt_impr=te_ttt_auc_history.last_one - te_auc_history.last_one,
                te_aug_auc=te_aug_auc_history.last_one,
                aug_impr=te_aug_auc_history.last_one - te_auc_history.last_one,
            )
        )


def test_time_tuning_presaved_models_only_last(gnn):
    dataset_name = config.TuningDataset.dataset
    logger = api.get_configured_logger(
        name=f'tune_{dataset_name}',
    )
    log_all_config(logger)
    logger.info('Started Test Time Tuning')
    tr_dataset, va_dataset, te_dataset = split_dataset(
        api.get_configured_tuning_dataset()
    )
    te_loader = get_eval_loader(te_dataset)
    # transform to TTT dataset
    te_dataset = api.get_configured_ttt_dataset(te_dataset)
    if config.OnlineLearning.use_shuffle:
        te_dataset = te_dataset.shuffle()
    te_ttt_loader = DataLoader(
        dataset=te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # set up classifier, optimizer, and criterion

    # optimizers
    # prepare to record evaluations
    te_auc_history = api.get_configured_history(f'{dataset_name}_te_auc_{config.seed}')
    te_ttt_auc_history = api.get_configured_history(f'{dataset_name}_te_ttt_auc_{config.seed}')
    te_aug_auc_history = api.get_configured_history(f'{dataset_name}_te_aug_auc_{config.seed}')

    logger.debug('Start loading models and evaluation.')
    gnn_states_backup = copy.deepcopy(gnn.state_dict())

    model_path = Path(
        config.TestTimeTuning.presaved_model_path + f'/tuning_model_{config.TuningDataset.dataset}_{config.seed}_e100.pt')
    if not model_path.exists():
        logger.info(f'{model_path} does not exists. Existing evaluation of seed {config.seed}')
        exit()
    gnn = api.get_configured_gnn()
    gnn.load_state_dict(gnn_states_backup, strict=False)
    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)
    clf.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    te_auc_history.append(eval_chem(clf, te_loader))

    if config.TestTimeTuning.add_prompts:
        if config.AdvAug.is_enabled:
            if config.Prompt.use_ssf_prompt:
                clf.gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.SSFPrompt(
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                            batch_size=config.Tuning.batch_size,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )
            else:
                clf.gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.NodePromptPtb(
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                            batch_size=config.Tuning.batch_size,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )
        else:
            gnn.node_prompts = nn.ModuleList(
                [
                    src.model.NodePrompt(
                        enable_ssf=config.Prompt.enable_ssf or config.SSF.is_enabled,
                        uniform_init_interval=config.Prompt.uniform_init_interval,
                    ).to(config.device)
                    for _ in range(config.Prompt.num)
                ]
            )

    if config.Prompt.use_node_wise_prompt:
        ttt_auc, aug_auc = api.node_wise_adv_eval(clf, te_ttt_loader)
    elif config.AdvAug.is_enabled:
        ttt_auc, aug_auc = api.adv_eval(clf, te_ttt_loader)
    elif config.SSF.is_enabled:
        ttt_auc, aug_auc = ttt_ssf_eval(clf, te_ttt_loader)
    elif config.TestTimeTuning.add_prompts:
        ttt_auc, aug_auc = ttt_prompt_eval(clf, te_ttt_loader)
    else:
        ttt_auc, aug_auc = ttt_eval(clf, te_ttt_loader)

    te_ttt_auc_history.append(ttt_auc)
    te_aug_auc_history.append(aug_auc)

    # te_auc_history.save()
    te_ttt_auc_history.save()
    te_aug_auc_history.save()
    logger.info(
        training_bar(
            99,
            config.Tuning.epochs,
            te_auc=te_auc_history.last_one,
            te_ttt_auc=te_ttt_auc_history.last_one,
            ttt_impr=te_ttt_auc_history.last_one - te_auc_history.last_one,
            te_aug_auc=te_aug_auc_history.last_one,
            aug_impr=te_aug_auc_history.last_one - te_auc_history.last_one,
        )
    )


def get_current_filename(file) -> str:
    return Path(file).name.split('.')[0]


def use_tbr_bn():
    src.model.TBR.replace_bn()
