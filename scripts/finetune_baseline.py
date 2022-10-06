import context

import torch
from torch_geometric.loader import DataLoader

import src
import config

CONFIG_NAME = 'finetune_baseline'
DEVICE = 0

if __name__ == '__main__':
    """DEBUG"""
    config.datasets = ['bbbp']
    """"""

    # set config name
    config.config_name = CONFIG_NAME
    # set device
    config.device = f'cuda:{DEVICE}'
    # set seed
    src.utils.set_seed(config.seed)
    # set logger
    logger = src.Logger('main')
    """
    Tuning
    """
    for seed in range(10):
        # set seed
        config.seed = seed
        src.utils.set_seed(config.seed)
        # tune all datasets
        for ds in config.datasets:
            # load model
            model = src.model.pretraining.BarlowTwins()
            state_dict = torch.load(
                config.Paths.models / 'base_bt_model.pt',
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)
            model.to(config.device)
            model.train()
            # set tuning dataset
            config.TuningDataset.dataset = ds
            # log all config for later check
            logger.log_all_config()
            # tune
            src.utils.tune(config.TuningDataset.dataset, model.gnn)
