from __future__ import annotations
import context

from multiprocessing import Process

import torch

import src.dataset
from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'uda_dual_v2'
DEVICE: int = 0


def search_epochs_uda_dual(epochs, device):
    # set config
    config.config_name = f'{CONFIG_NAME}_e{epochs}'
    config.Pretraining.batch_size = 256
    config.device = f'cuda:{device}'
    config.Pretraining.use_dual_dataset = True
    config.Pretraining.epochs = epochs
    config.Pretraining.dual_ds_version = 1
    if DEBUG:
        api.set_debug_mode()

    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed()
        for ds in config.datasets:
            gnn_model = api.get_configured_gnn()
            bt_model = api.get_configured_barlow_twins(gnn_model)
            state_dict = torch.load(
                config.Paths.models / 'base_bt_model.pt',
                map_location=lambda storage, loc: storage,
            )
            bt_model.load_state_dict(state_dict)
            bt_model.train()
            """
            Pretraining
            """
            config.PretrainingDataset.dataset = ds
            config.GNN.drop_ratio = 0
            api.pretrain(bt_model)
            """
            Tuning
            """
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = 0.5
            new_gnn = api.get_configured_gnn()
            new_gnn.load_state_dict(bt_model.gnn.state_dict())
            api.tune(new_gnn)

    api.analyze_results()


if __name__ == '__main__':
    epochs_device_pairs = [
        (10, 0),
        (20, 0),
        (50, 0),
        (100, 3),
        (200, 3)
    ]
    for e, d in epochs_device_pairs:
        Process(
            target=search_epochs_uda_dual,
            args=(e, d),
        ).start()
