import context

from copy import deepcopy
import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'uda_trans_pretrain2'
DEVICE: int = 0


def train_with_para(aug: str, epochs: int, batch_size: int, device: int):
    # set general config
    config.config_name = f'{CONFIG_NAME}_{aug}_e{epochs}_bs{batch_size}'
    config.Pretraining.use_graph_trans = True
    config.Pretraining.lr = 1e-4
    config.GraphTrans.drop_ratio = 0
    config.device = f'cuda:{device}'
    config.BarlowTwins.sizes = (128, 1200, 1200, 1200)
    config.Tuning.use_lr_scheduler = False
    if DEBUG:
        api.set_debug_mode()

    """
    Load backbone
    """
    trans_model = api.get_configured_graph_trans()
    bt_model = api.get_configured_barlow_twins(trans_model)
    state_dict = torch.load(
        config.Paths.models / f'trans_bs256_backbone.pt',
        map_location=lambda storage, loc: storage,
    )
    bt_model.load_state_dict(state_dict)
    bt_model.to(config.device)
    bt_model.train()

    original_bt_states = deepcopy(bt_model.state_dict())
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            """
            Pretraining 2
            """
            config.PretrainingDataset.dataset = ds
            config.Pretraining.batch_size = batch_size
            config.GraphTrans.drop_ratio = 0
            config.Pretraining.epochs = epochs
            config.PretrainingDataset.aug_1 = aug
            config.PretrainingDataset.aug_2 = aug
            trans_model = api.get_configured_graph_trans()
            bt_model = api.get_configured_barlow_twins(trans_model)
            bt_model.load_state_dict(original_bt_states)
            api.pretrain(bt_model)
            """
            Tuning
            """
            # set specific config
            config.TuningDataset.dataset = ds
            config.Tuning.lr = 1e-5 if ds == 'muv' else 1e-4
            config.GraphTrans.drop_ratio = 0.3 if ds == 'clintox' else 0.5
            config.Tuning.epochs = 128 if ds == 'clintox' else 100
            # tune
            trans_model = api.get_configured_graph_trans()
            trans_model.load_state_dict(
                bt_model.gnn.state_dict()
            )
            api.tune(trans_model)
    api.analyze_results_by_ratio()


if __name__ == '__main__':
    from multiprocessing import Process

    devices = [0, 1] * 6

    idx = 0
    for aug in ('random', 'dropN'):
        for epochs in (20, 50, 100):
            for batch_size in (128, 256):
                Process(
                    target=train_with_para,
                    args=(aug, epochs, batch_size, devices[idx]),
                ).start()
                idx += 1
