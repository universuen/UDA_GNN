import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0


def search_para(epochs: int, e_drop_ratio: float):
    # set configs
    config.config_name = f'{CONFIG_NAME}_e{epochs}_edr{e_drop_ratio}'
    config.device = f'cuda:{DEVICE}'
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = 8
    config.SSF.is_enabled = True

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    if config.SSF.is_enabled:
        api.replace_with_ssf()

    # load models from ROM
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']
    d_states = states_dict['decoder']

    # TTT
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TestTimeTuning.add_prompts = True
            config.Tuning.use_node_prompt = True
            config.Tuning.use_edge_prompt = False
            config.Prompt.mode = 'add'
            config.TuningDataset.dataset = ds
            config.TuningLoader.num_workers = 2
            config.Tuning.batch_size = 32
            config.Encoder.drop_ratio = e_drop_ratio
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'
            config.Tuning.epochs = epochs

            encoder = api.get_configured_encoder()
            encoder.load_state_dict(e_states, strict=False)
            encoder.enable_selfloop()

            api.tune_ssf_prompts(encoder)
    api.analyze_results_by_ratio()


if __name__ == '__main__':
    from multiprocessing import Process

    Process(target=search_para, args=(200, 0.5)).start()
    Process(target=search_para, args=(200, 0)).start()
