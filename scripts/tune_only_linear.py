import context

from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'fixed_only_linear'
DEVICE: int = 3


if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.GNN.drop_ratio = 0.5
    config.device = f'cuda:{DEVICE}'
    config.Tuning.use_lr_scheduler = False
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    gnn_model = api.get_configured_gnn()
    bt_model = api.get_configured_barlow_twins(gnn_model)
    state_dict = torch.load(
        config.Paths.models / 'base_bt_model.pt',
        map_location=lambda storage, loc: storage,
    )
    bt_model.load_state_dict(state_dict)
    bt_model.train()
    """
    Tuning
    """
    original_states = deepcopy(bt_model.state_dict())
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            bt_model.load_state_dict(original_states)
            api.tune_linear(bt_model.gnn)
    api.analyze_results_by_ratio()
