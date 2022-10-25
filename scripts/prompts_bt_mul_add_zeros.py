import context

from copy import deepcopy

import torch
from torch_geometric.nn.inits import glorot

from src import config
from src import api
import src

DEBUG: bool = False
CONFIG_NAME: str = 'prompts_mul_add_zeros'
DEVICE: int = 0


# set init method
def _init_weights(self: src.model.NodePrompt):
    glorot(self.value)
    self.b = torch.nn.Parameter(
        torch.zeros_like(self.b)
    )


if __name__ == '__main__':
    src.model.NodePrompt.init_weights = _init_weights
    # set config
    config.config_name = f'{CONFIG_NAME}'
    config.GNN.drop_ratio = 0.5
    config.device = f'cuda:{DEVICE}'
    config.Tuning.use_lr_scheduler = False
    config.Tuning.use_node_prompt = True
    config.Tuning.use_edge_prompt = False
    config.Prompt.mode = 'mul_add'
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
