from torch_geometric.loader import DataLoader

import src
from src import config, types
from src.original.mae.splitters import scaffold_split
import pandas as pd

def get_configured_logger(name: str) -> types.Logger:
    if config.config_name is None:
        raise NotImplementedError('config_name is mandatory.')
    logs_dir = config.Paths.logs / config.config_name
    return src.logger.Logger(
        name=name,
        level=config.Logger.level,
        logs_dir=logs_dir,
    )


def get_configured_pretraining_dataset() -> types.Dataset:
    dataset = config.PretrainingDataset.dataset
    mol_dataset = src.dataset.MoleculeAugDataset(
        root=str(config.Paths.datasets / dataset),
        dataset=dataset,
        aug_1=config.PretrainingDataset.aug_1,
        aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
        aug_2=config.PretrainingDataset.aug_2,
        aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
        use_original=config.PretrainingDataset.use_original,
    )

    if dataset == 'zinc_standard_agent':
        return mol_dataset
    train_dataset, valid_dataset, test_dataset = get_scaffold_split(mol_dataset, config.Paths.datasets / dataset)
    return train_dataset

def get_configured_dual_dataset():
    dataset = config.PretrainingDataset.dataset
    return src.dataset.DualDataset(
        dataset_path=str(config.Paths.datasets),
        dataset=dataset,
        aug_1=config.PretrainingDataset.aug_1,
        aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
        aug_2=config.PretrainingDataset.aug_2,
        aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
        use_original=config.PretrainingDataset.use_original,
    )


def get_configured_dual_dataset_v2():
    dataset = config.PretrainingDataset.dataset
    return src.dataset.DualDatasetV2(
        dataset_path=str(config.Paths.datasets),
        dataset=dataset,
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
    dataset = config.TuningDataset.dataset
    return src.dataset.MoleculeDataset(
        root=str(config.Paths.datasets / dataset),
        dataset=dataset,
    )


def get_configured_ttt_dataset(test_dataset) -> types.Dataset:
    if config.TestTimeTuning.aug in ['dropout', 'featM']:
        aug = 'none'
    else:
        aug = config.TestTimeTuning.aug
    dataset = src.dataset.TTTAugDataset(
        dataset=test_dataset,
        num_augmentations=config.TestTimeTuning.num_augmentations,
        aug=aug,
        aug_ratio=config.TestTimeTuning.aug_ratio,
    )
    return dataset


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


def get_configured_barlow_twins(gnn: src.types.GNNModel) -> src.model.pretraining.BarlowTwins:
    return src.model.pretraining.BarlowTwins(
        model=gnn,
        lambda_=config.BarlowTwins.lambda_,
        sizes=config.BarlowTwins.sizes,
        use_graph_trans=config.Pretraining.use_graph_trans,
    ).to(config.device)


def get_configured_history(name: str, values: list = None) -> types.History:
    return src.History(
        name=name,
        values=values,
        result_dir=config.Paths.results / config.config_name,
    )


def get_configured_graph_trans() -> types.GNNModel:
    return src.model.gnn.GraphTrans(
        gnn_dropout=config.GraphTrans.gnn_drop_ratio,
        d_model=config.GraphTrans.d_model,
        transformer_dropout=config.GraphTrans.trans_drop_ratio,
    )


def get_configured_mae_loader(dataset: src.dataset.MoleculeDataset) -> src.loader.MAELoader:
    return src.loader.MAELoader(
        dataset=dataset,
        batch_size=config.Pretraining.batch_size,
        shuffle=config.PretrainingLoader.shuffle,
        num_workers=config.PretrainingLoader.num_workers,
        mask_rate=config.MAELoader.mask_rate,
    )


def get_scaffold_split(dataset, dataset_path):
    smiles_list = pd.read_csv(
        dataset_path / 'processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    return train_dataset, valid_dataset, test_dataset


def get_configured_mae_pretraining_dataset(dataset: src) -> src.dataset.MoleculeDataset:
    mol_dataset = src.dataset.MoleculeDataset(
            root=str(config.Paths.datasets / dataset),
            dataset=dataset,
        )
    if dataset == 'zinc_standard_agent':
        return mol_dataset
    train_dataset, valid_dataset, test_dataset = get_scaffold_split(mol_dataset, config.Paths.datasets / dataset)
    return train_dataset


def get_configured_encoder() -> src.model.gnn.mae.Encoder:
    return src.model.gnn.mae.Encoder(
        num_layer=config.Encoder.num_layer,
        emb_dim=config.Encoder.emb_dim,
        jk=config.Encoder.jk,
        drop_ratio=config.Encoder.drop_ratio,
    )


def get_configured_decoder() -> src.model.gnn.mae.Decoder:
    return src.model.gnn.mae.Decoder(
        hidden_dim=config.Decoder.hidden_dim,
        out_dim=config.Decoder.out_dim,
    )


def get_configured_node_prompt() -> src.model.NodePrompt:
    return src.model.NodePrompt(
        mode=config.Prompt.mode,
    ).to(config.device)
