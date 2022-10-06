import context

import torch

import config
import utils

CONFIG_NAME = 'finetune_UDA'
DEVICE = 2

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.Pretraining.batch_size = 256
    # set others
    config.device = f'cuda:{DEVICE}'
    utils.set_seed()

    for seed in config.loop_seeds:
        config.seed = seed
        utils.set_seed()
        for ds in config.datasets:
            gnn_model = utils.load_gnn()
            bt_model = utils.load_barlow_twins(gnn_model)
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
            utils.pretrain(bt_model)
            """
            Tuning
            """
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
            utils.tune(config.TuningDataset.dataset, bt_model.gnn)
