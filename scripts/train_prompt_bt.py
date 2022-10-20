import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'fixed_prompts'
DEVICE: int = 1


def tune_with_edge_prompt(device: int):
    # set config
    config.config_name = f'{CONFIG_NAME}_with_edge'
    config.GNN.drop_ratio = 0.5
    config.device = f'cuda:{device}'
    config.Tuning.use_lr_scheduler = False
    config.Tuning.use_node_prompt = True
    config.Tuning.use_edge_prompt = True
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    bt_model = api.get_configured_barlow_twins(
        api.get_configured_gnn()
    )
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
            bt_model = api.get_configured_barlow_twins(
                api.get_configured_gnn()
            )
            bt_model.load_state_dict(original_states)
            api.tune_with_prompt(bt_model.gnn)
    api.analyze_results_by_ratio()


def tune_without_edge_prompt(device: int):
    # set config
    config.config_name = f'{CONFIG_NAME}_without_edge'
    config.GNN.drop_ratio = 0.5
    config.device = f'cuda:{device}'
    config.Tuning.use_lr_scheduler = False
    config.Tuning.use_node_prompt = True
    config.Tuning.use_edge_prompt = False
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
            bt_model = api.get_configured_barlow_twins(
                api.get_configured_gnn()
            )
            bt_model.load_state_dict(original_states)
            api.tune_with_prompt(bt_model.gnn)
    api.analyze_results_by_ratio()


if __name__ == '__main__':
    Process(
        target=tune_with_edge_prompt,
        args=(3,),
    ).start()
    Process(
        target=tune_without_edge_prompt,
        args=(3,),
    ).start()
