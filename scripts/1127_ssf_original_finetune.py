import context

from multiprocessing import Process
from copy import deepcopy

import torch
from torch.nn import functional

from src import config
from src import api
from src import model

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0


# define original forward method
def original_forward(
        self: model.SSLinear,
        x: torch.Tensor
) -> torch.Tensor:
    y = functional.linear(x, self.weight, self.bias)
    return self.gamma * y + self.beta


# replace
model.SSLinear.forward = original_forward

if __name__ == '__main__':
    # set configs
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = 8
    config.SSF.is_enabled = True

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    if config.SSF.is_enabled:
        api.replace_linear()

    # load models from ROM
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']
    d_states = states_dict['decoder']

    # tune
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            """
            tune
            """
            config.TuningDataset.dataset = ds
            config.TuningLoader.num_workers = 2
            config.Tuning.batch_size = 32
            config.TestTimeTuning.aug = 'dropout'
            config.TestTimeTuning.aug_ratio = 0.5
            config.TestTimeTuning.num_iterations = 1
            config.TestTimeTuning.num_augmentations = 32
            config.Encoder.drop_ratio = 0.5
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'

            encoder = api.get_configured_encoder()
            encoder.load_state_dict(e_states, strict=False)
            encoder.enable_selfloop()

            api.ssf_tune(encoder)

    api.analyze_results_by_ratio()
