from __future__ import annotations

import torch.nn
from torch import nn

from .utils import *
from src.original.mae.pretraining import sce_loss
from src import config, api
from pathlib import Path


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
        if (e + 1) % config.Pretraining.save_epoch == 0:
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
        model.state_dict(),
        models_dir / f'pretraining_model_{config.PretrainingDataset.dataset}_final.pt'
    )
    loss_history.save()


def ssf_tune(gnn: src.types.GNNModel):
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
    # collect ss linear parameters
    ss_parameters = list(clf.linear.parameters())
    for i in clf.modules():
        if type(i) in [src.model.SSLinear, src.model.SSBatchNorm]:
            ss_parameters.append(i.gamma)
            ss_parameters.append(i.beta)
    optimizer = torch.optim.Adam(ss_parameters, config.Tuning.lr)
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


def tune_ssf_prompts(gnn: src.types.GNNModel):
    # link the prompt to gnn
    if config.Tuning.use_node_prompt:
        gnn.node_prompts = nn.ModuleList(
            [api.get_configured_node_prompt() for _ in range(config.GNN.num_layer)]
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
    # add prompts parameters
    if config.Tuning.use_node_prompt:
        for prompt in gnn.node_prompts:
            parameters += list(prompt.parameters())
    # add SSF parameters
    if config.SSF.is_enabled:
        for i in clf.modules():
            if type(i) in [src.model.SSLinear, src.model.SSBatchNorm]:
                parameters.append(i.gamma)
                parameters.append(i.beta)

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


def tune_with_prompt(gnn: src.types.GNNModel):
    # link the prompt to gnn
    if config.Tuning.use_node_prompt:
        gnn.node_prompts = nn.ModuleList(
            [api.get_configured_node_prompt() for _ in range(config.GNN.num_layer)]
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


def train_mae(
        encoder: src.model.gnn.mae.Encoder,
        decoder: src.model.gnn.mae.Decoder,
        freeze_decoder: bool = False,
):
    logger = api.get_configured_logger(f'pretrain_{config.PretrainingDataset.dataset}_{config.seed}')
    log_all_config(logger)
    logger.info('started pretraining')

    logger.debug("set models")
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.train()
    decoder.train()

    logger.debug('set optimizers')
    e_optimizer = torch.optim.Adam(
        params=encoder.parameters(),
        lr=config.Pretraining.lr,
    )
    d_optimizer = torch.optim.Adam(
        params=decoder.parameters() if not freeze_decoder else [torch.zeros(1)],
        lr=config.Pretraining.lr,
    )

    logger.debug('set data loader')
    loader = api.get_configured_mae_loader(
        api.get_configured_mae_pretraining_dataset(
            config.PretrainingDataset.dataset
        )
    )

    logger.debug('set loss history')
    loss_history = api.get_configured_history('pretraining_losses')

    logger.debug('training loop')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    for e in range(config.Pretraining.epochs):
        for idx, batch in enumerate(loader):
            batch.to(config.device)
            e_optimizer.zero_grad()
            decoder.zero_grad()
            node_rep = encoder(batch.x, batch.edge_index, batch.edge_attr)
            node_attr_label = batch.node_attr_label
            masked_node_indices = batch.masked_atom_indices
            pred_node = decoder(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)
            loss = sce_loss(node_attr_label, pred_node[masked_node_indices])
            loss.backward()
            e_optimizer.step()
            d_optimizer.step()
            loss_history.append(loss)
            logger.debug(f'epoch: {e}, loss: {loss}, process: {(idx + 1) / len(loader)}')
        logger.info(training_bar(e, config.Pretraining.epochs, loss=loss_history.last_one))

        if (e + 1) % config.Pretraining.save_epoch == 0:
            logger.debug(f'saving mae encoder and decoder at the {e + 1} epoch')
            torch.save(
                {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                },
                models_dir / f'mae_{config.PretrainingDataset.dataset}_e{e + 1}_s{config.seed}.pt'
            )
    loss_history.save()


def adv_eval(clf_model: src.model.GraphClf, loader):
    assert config.AdvAug.is_enabled
    assert config.Tuning.use_node_prompt

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

    others_parameters = []
    for i in clf_model.modules():
        if type(i) != src.model.NodePrompt:
            others_parameters += list(i.parameters())
    others_parameters = set(others_parameters)
    others_optimizer = torch.optim.Adam(
        params=others_parameters,
        lr=config.Tuning.lr,
    )
    # back up
    clf_states = copy.deepcopy(clf_model.state_dict())
    others_optimizer_states = copy.deepcopy(others_optimizer.state_dict())

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

            others_optimizer.zero_grad()
            # calculate loss
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            loss /= config.AdvAug.step_size

            # maximize loss by updating prompts
            for __ in range(config.AdvAug.num_iterations):
                # calculate gradients
                loss.backward()
                # update prompts parameters based on gradients sign
                for i in clf_model.gnn.node_prompts:
                    for j in i.parameters():
                        if j.grad is None:
                            continue
                        j_data = j.detach() + config.AdvAug.step_size * torch.sign(j.grad.detach())
                        j.data = j_data.data
                        j.grad[:] = 0
                # calculate loss
                outputs = clf_model(augmentations)
                if config.TestTimeTuning.conf_ratio < 1:
                    outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
                loss, aug_pre = marginal_entropy_bce_v2(outputs)
                loss /= config.AdvAug.step_size

                aug_pre_list.append(aug_pre)

            # minimize loss by updating others
            loss.backward()
            others_optimizer.step()

        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            # remove prompts
            prompts = clf_model.gnn.node_prompts
            clf_model.gnn.node_prompts = None
            # predict
            pred = clf_model(data)
            # restore
            clf_model.gnn.node_prompts = prompts
        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)
        y_aug_scores.append(aug_pre)
        # reset
        if config.OnlineLearning.is_enabled:
            pass
        else:
            clf_model.load_state_dict(clf_states)
            others_optimizer.load_state_dict(others_optimizer_states)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_aug_scores = torch.cat(y_aug_scores, dim=0).cpu().numpy()

    mean_roc = _evaluate(y_true, y_scores)
    mean_aug_roc = _evaluate(y_true, y_aug_scores)
    set_bn_prior(1)
    return mean_roc, mean_aug_roc
