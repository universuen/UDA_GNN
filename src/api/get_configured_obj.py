from torch_geometric.loader import DataLoader

import src
from src import config, types


def get_configured_logger(name: str) -> types.Logger:
    if config.config_name is None:
        raise NotImplementedError('config_name is mandatory.')
    logs_dir = config.Paths.logs / config.config_name
    return src.logger.Logger(
        name=name,
        logs_dir=logs_dir,
    )


def get_configured_pretraining_dataset() -> types.Dataset:
    return src.dataset.MoleculeAugDataset(
        dataset=config.PretrainingDataset.dataset,
        aug_1=config.PretrainingDataset.aug_1,
        aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
        aug_2=config.PretrainingDataset.aug_2,
        aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
        use_original=config.PretrainingDataset.use_original,
    )


def get_configured_pretraining_loader(dataset: types.Dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.Pretraining.batch_size,
        shuffle=True,
        num_workers=config.PretrainingLoader.num_workers,
        pin_memory=config.PretrainingLoader.pin_memory,
        drop_last=config.PretrainingLoader.drop_last,
        worker_init_fn=config.PretrainingLoader.worker_init_fn,
    )


def get_configured_tuning_dataset() -> types.Dataset:
    return src.dataset.MoleculeDataset(
        dataset=config.TuningDataset.dataset,
    )


def get_configured_tuning_dataloader(dataset: types.Dataset) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.Tuning.batch_size,
        shuffle=True,
        num_workers=config.TuningLoader.num_workers,
        pin_memory=True,
    )


def get_configured_gnn() -> types.GNNModel:
    return src.model.gnn.GNN(
        num_layer=config.GNN.num_layer,
        emb_dim=config.GNN.emb_dim,
        jk=config.GNN.jk,
        drop_ratio=config.GNN.drop_ratio,
    ).to(config.device)


def get_configured_barlow_twins(gnn: src.types.GNNModel) -> types.PretrainingModel:
    return src.model.pretraining.BarlowTwins(
        model=gnn,
        lambda_=config.BarlowTwins.lambda_,
        sizes=config.BarlowTwins.sizes,
    ).to(config.device)
