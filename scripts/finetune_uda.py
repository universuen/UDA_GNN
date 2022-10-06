import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'finetune_UDA'
DEVICE: int = 0

if __name__ == '__main__':
    # set config
    if DEBUG:
        api.set_debug_mode()
    else:
        config.config_name = CONFIG_NAME
        config.Pretraining.batch_size = 256
        config.device = f'cuda:{DEVICE}'

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
            config.GNN.drop_ratio = config.Pretraining.gnn_dropout_ratio
            api.pretrain(bt_model)
            """
            Tuning
            """
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
            api.tune(config.TuningDataset.dataset, bt_model.gnn)
