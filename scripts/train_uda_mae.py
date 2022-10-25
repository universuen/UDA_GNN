import torch

import context

from copy import deepcopy

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'uda_mae'
DEVICE: int = 0
FROM_SCRATCH: bool = False

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.Pretraining.batch_size = 256
    config.PretrainingLoader.num_workers = 6
    config.MAELoader.mask_rate = 0.35
    config.device = f'cuda:{DEVICE}'
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining 1
    """
    encoder = api.get_configured_encoder()
    decoder = api.get_configured_decoder()
    if FROM_SCRATCH:
        api.train_mae(encoder, decoder)
    else:
        encoder.load_state_dict(
            torch.load(
                config.Paths.models / 'pretrained_encoder.pt',
                map_location=lambda storage, loc: storage
            )
        )
        encoder.to(config.device)
        encoder.train()

    original_e_states = deepcopy(encoder.state_dict())
    original_d_states = deepcopy(decoder.state_dict())
    # set config
    config.TuningLoader.num_workers = 2
    config.Tuning.batch_size = 32
    config.Tuning.epochs = 200
    config.Encoder.drop_ratio = 0.5
    config.Tuning.use_lr_scheduler = False
    encoder = api.get_configured_encoder()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            encoder.load_state_dict(original_e_states)
            decoder.load_state_dict(original_d_states)
            """
            Pretraining 2
            """
            config.PretrainingDataset.dataset = ds
            api.train_mae(encoder, decoder)
            """
            Tuning
            """
            config.TuningDataset.dataset = ds
            api.tune(encoder)
    api.analyze_results_by_ratio()
