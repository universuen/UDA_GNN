import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'finetune_baseline'
DEVICE: int = 0

if __name__ == '__main__':
    # set config
    if DEBUG:
        api.set_debug_mode()
    else:
        config.config_name = CONFIG_NAME
        config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
        config.device = f'cuda:{DEVICE}'

    """
    Tuning
    """
    for seed in config.loop_seeds:
        # set seed
        config.seed = seed
        api.set_seed()
        # tune all datasets
        for ds in config.datasets:
            # load model
            gnn = api.get_configured_gnn()
            model = api.get_configured_barlow_twins(gnn)
            state_dict = torch.load(
                config.Paths.models / 'base_bt_model.pt',
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)
            model.train()
            # set tuning dataset
            config.TuningDataset.dataset = ds
            # tune
            api.tune(config.TuningDataset.dataset, model.gnn)
