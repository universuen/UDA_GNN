import context

import torch

import config
import utils

CONFIG_NAME = 'finetune_baseline'
DEVICE = 0

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
    # set device
    config.device = f'cuda:{DEVICE}'
    # set seed
    utils.set_seed()

    """
    Tuning
    """
    for seed in config.loop_seeds:
        # set seed
        config.seed = seed
        utils.set_seed()
        # tune all datasets
        for ds in config.datasets:
            # load model
            gnn = utils.load_gnn()
            model = utils.load_barlow_twins(gnn)
            state_dict = torch.load(
                config.Paths.models / 'base_bt_model.pt',
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)
            model.train()
            # set tuning dataset
            config.TuningDataset.dataset = ds
            # tune
            utils.tune(config.TuningDataset.dataset, model.gnn)
