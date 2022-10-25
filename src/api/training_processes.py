from __future__ import annotations

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
        model.state_dict(),
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
        params=decoder.parameters(),
        lr=config.Pretraining.lr,
    )

    logger.debug('set data loader')
    loader = api.get_configured_mae_loader(
        api.get_configured_molecule_dataset(
            config.PretrainingDataset.dataset
        )
    )

    logger.debug('set loss history')
    loss_history = api.get_configured_history('pretraining_losses')

    logger.debug('training loop')
    for e in range(config.Pretraining.epochs):
        for idx, batch in enumerate(loader):
            batch.to(config.device)
            e_optimizer.zero_grad()
            d_optimizer.zero_grad()
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

    logger.debug('save encoder')
    models_dir = config.Paths.models / config.config_name
    models_dir.mkdir(exist_ok=True)
    torch.save(
        encoder.state_dict(),
        models_dir / f'encoder_{config.PretrainingDataset.dataset}_{config.seed}.pt'
    )
    loss_history.save()
