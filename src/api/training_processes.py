from __future__ import annotations

import torch.nn
from torch import nn

import src.model
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


def node_wise_adv_eval(clf_model: src.model.GraphClf, loader):
    assert config.AdvAug.is_enabled
    assert config.Tuning.use_node_prompt
    # remove original prompts
    clf_model.gnn.node_prompts = None

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
        if type(i) != src.model.NodeWisePromptPtb:
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
        # add prompts
        clf_model.gnn.node_prompts = nn.ModuleList(
            [
                src.model.NodeWisePromptPtb(
                    num_nodes=augmentations.x.shape[0],
                    uniform_init_interval=config.Prompt.uniform_init_interval,
                ).to(config.device)
                for _ in range(config.Prompt.num)
            ]
        )

        for _ in range(config.TestTimeTuning.num_iterations):
            # calculate loss
            others_optimizer.zero_grad()
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            t_loss = tent_loss(outputs) / config.AdvAug.num_iterations
            m_loss = 0
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            m_loss += loss / config.AdvAug.num_iterations
            aug_pre_list.append(aug_pre)
            # maximize loss by updating prompts
            for __ in range(config.AdvAug.num_iterations - 1):
                # calculate gradients
                t_loss.backward(retain_graph=True)
                # update prompts parameters based on gradients sign
                for i in clf_model.gnn.node_prompts.parameters():
                    i_data = i.detach() + config.AdvAug.step_size * torch.sign(i.grad.detach())
                    i.data = i_data.data
                    i.grad[:] = 0
                # calculate loss
                outputs = clf_model(augmentations)
                if config.TestTimeTuning.conf_ratio < 1:
                    outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
                t_loss = tent_loss(outputs) / config.AdvAug.num_iterations
                loss, aug_pre = marginal_entropy_bce_v2(outputs)
                m_loss += loss / config.AdvAug.num_iterations
                aug_pre_list.append(aug_pre)

            t_loss.backward(retain_graph=True)
            for i in clf_model.gnn.node_prompts.parameters():
                i_data = i.detach() + config.AdvAug.step_size * torch.sign(i.grad.detach())
                i.data = i_data.data
                i.grad[:] = 0
            others_optimizer.zero_grad()
            m_loss.backward()
            others_optimizer.step()

        aug_pre = sum(aug_pre_list) / len(aug_pre_list)

        # test
        with torch.no_grad():
            clf_model.eval()
            # remove prompts
            clf_model.gnn.node_prompts = None
            # predict
            pred = clf_model(data)
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
        if type(i) != src.model.NodePromptPtb:
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
            # calculate loss
            others_optimizer.zero_grad()
            outputs = clf_model(augmentations)
            if config.TestTimeTuning.conf_ratio < 1:
                outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
            t_loss = tent_loss(outputs) / config.AdvAug.num_iterations
            m_loss = 0
            loss, aug_pre = marginal_entropy_bce_v2(outputs)
            m_loss += loss / config.AdvAug.num_iterations
            aug_pre_list.append(aug_pre)
            # maximize loss by updating prompts
            for __ in range(config.AdvAug.num_iterations - 1):
                # calculate gradients
                t_loss.backward(retain_graph=True)
                # update prompts parameters based on gradients sign
                if config.Prompt.use_ssf_prompt and config.AdvAug.w_updating_strategy == 'grad':
                    for name, para in clf_model.gnn.node_prompts.named_parameters():
                        if name[-1] == 'w':
                            para_data = para.detach() + config.AdvAug.step_size * para.grad.detach()
                        elif name[-1] == 'b':
                            para_data = para.detach() + config.AdvAug.step_size * torch.sign(para.grad.detach())
                        else:
                            raise ValueError
                        para.data = para_data.data
                        para.grad[:] = 0
                else:
                    for i in clf_model.gnn.node_prompts.parameters():
                        i_data = i.detach() + config.AdvAug.step_size * torch.sign(i.grad.detach())
                        i.data = i_data.data
                        i.grad[:] = 0
                # calculate loss
                outputs = clf_model(augmentations)
                if config.TestTimeTuning.conf_ratio < 1:
                    outputs = confidence_selection(outputs, config.TestTimeTuning.conf_ratio)
                t_loss = tent_loss(outputs) / config.AdvAug.num_iterations
                loss, aug_pre = marginal_entropy_bce_v2(outputs)
                m_loss += loss / config.AdvAug.num_iterations
                aug_pre_list.append(aug_pre)

            t_loss.backward(retain_graph=True)
            for i in clf_model.gnn.node_prompts.parameters():
                i_data = i.detach() + config.AdvAug.step_size * torch.sign(i.grad.detach())
                i.data = i_data.data
                i.grad[:] = 0
            others_optimizer.zero_grad()
            m_loss.backward()
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


def flag_tune_and_save_models(gnn):
    assert config.AdvAug.is_enabled
    assert config.TestTimeTuning.add_prompts
    assert config.Tuning.use_node_prompt

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

    clf = src.model.GraphClf(
        gnn=gnn,
        dataset=config.TuningDataset.dataset,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)
    optimizer = torch.optim.Adam(clf.parameters(), config.Tuning.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.AdvAug.lr_scheduler_step_size,
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

            # add prompts
            if config.Prompt.use_leaky_relu:
                clf.gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.LeakyReLUPrompt(
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                            batch_size=config.Tuning.batch_size,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )
            elif config.Prompt.use_node_wise_prompt:
                clf.gnn.node_prompts = nn.ModuleList(
                    [
                        src.model.NodeWisePromptPtb(
                            num_nodes=batch.x.shape[0],
                            uniform_init_interval=config.Prompt.uniform_init_interval,
                        ).to(config.device)
                        for _ in range(config.Prompt.num)
                    ]
                )
            elif config.Prompt.use_ssf_prompt:
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

            optimizer.zero_grad()
            # calculate loss
            pred = clf(batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            is_valid = y ** 2 > 0  # shape = [N, C]
            loss_mat = criterion(pred, (y + 1) / 2)  # shape = [N, C]
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss /= config.AdvAug.num_iterations

            # maximize loss by updating prompts
            for _ in range(config.AdvAug.num_iterations - 1):
                # calculate gradients
                loss.backward()
                # update prompts parameters based on gradients sign
                if random.random() > config.AdvAug.roulette_ratio:
                    operator = lambda x1, x2: x1 + x2
                else:
                    operator = lambda x1, x2: x1 - x2

                if config.Prompt.use_ssf_prompt and config.AdvAug.w_updating_strategy == 'grad':
                    for name, para in clf.gnn.node_prompts.named_parameters():
                        if name[-1] == 'w':
                            para_data = operator(para.detach(), config.AdvAug.step_size * para.grad.detach())
                        elif name[-1] == 'b':
                            para_data = operator(
                                para.detach(),
                                config.AdvAug.step_size * torch.sign(para.grad.detach())
                            )
                        else:
                            raise ValueError
                        para.data = para_data.data
                        para.grad[:] = 0
                else:
                    for i in clf.gnn.node_prompts.parameters():
                        i_data = operator(i.detach(), config.AdvAug.step_size * torch.sign(i.grad.detach()))
                        i.data = i_data.data
                        i.grad[:] = 0
                # calculate loss
                pred = clf(batch)
                y = batch.y.view(pred.shape).to(torch.float64)
                is_valid = y ** 2 > 0  # shape = [N, C]
                loss_mat = criterion(pred, (y + 1) / 2)  # shape = [N, C]
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
                loss /= config.AdvAug.num_iterations

            # minimize loss by updating others
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            logger.debug(f'epoch: {e}, loss: {loss}, process: {(idx + 1) / len(training_loader)}')

            # remove prompts
            if config.AdvAug.keep_non_linear:
                for i in clf.gnn.node_prompts:
                    i.remove_ptb()
            else:
                clf.gnn.node_prompts = None

            # tune after adv
            if config.AdvAug.tune_after_adv:
                optimizer.zero_grad()
                # calculate loss
                pred = clf(batch)
                y = batch.y.view(pred.shape).to(torch.float64)
                is_valid = y ** 2 > 0  # shape = [N, C]
                loss_mat = criterion(pred, (y + 1) / 2)  # shape = [N, C]
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
                loss.backward()
                optimizer.step()

        if config.AdvAug.full_ds_bn:
            for idx, batch in enumerate(training_loader):
                batch = batch.to(config.device)
                clf(batch)

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

        if config.Tuning.use_lr_scheduler or config.AdvAug.use_lr_scheduler:
            lr_scheduler.step()
            logger.info(f'current LR: {lr_scheduler.get_last_lr()[0]}')

        if (e + 1) % config.TestTimeTuning.save_epoch == 0:
            logger.debug(f'Save the {e + 1} epoch model.')
            torch.save(
                clf.state_dict(),
                models_dir / f'tuning_model_{config.TuningDataset.dataset}_{config.seed}_e{e + 1}.pt'
            )


def flag_tune_and_save_models_v2(gnn):
    assert config.AdvAug.is_enabled
    assert config.TestTimeTuning.add_prompts
    assert config.Tuning.use_node_prompt

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

    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)

    logger.debug('Training loop')
    for e in range(config.Tuning.epochs):
        clf.train()
        for idx, batch in enumerate(training_loader):
            batch = batch.to(config.device)

            # add prompts
            clf.gnn.node_prompts = nn.ModuleList(
                [
                    src.model.NodePromptPtb(
                        uniform_init_interval=config.Prompt.uniform_init_interval,
                        batch_size=config.Tuning.batch_size,
                    ).to(config.device)
                    for _ in range(config.GNN.num_layer)
                ]
            )
            # config prompts optimizer
            prompts_optimizer = torch.optim.Adam(
                params=clf.gnn.node_prompts.parameters(),
                lr=config.AdvAug.step_size,
            )

            total_loss = 0
            # maximize loss by updating prompts
            for _ in range(config.AdvAug.num_iterations):
                # calculate loss
                pred = clf(batch)
                y = batch.y.view(pred.shape).to(torch.float64)
                is_valid = y ** 2 > 0  # shape = [N, C]
                loss_mat = criterion(pred, (y + 1) / 2)  # shape = [N, C]
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
                loss /= config.AdvAug.num_iterations
                # update prompts parameters based on gradients sign
                prompts_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                prompts_optimizer.step()
                total_loss += loss

            # minimize loss by updating others
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss)
            logger.debug(f'epoch: {e}, loss: {total_loss}, process: {(idx + 1) / len(training_loader)}')

            # remove prompts
            clf.gnn.node_prompts = None

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
