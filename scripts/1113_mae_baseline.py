import context

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'mae_m35_baseline'
DEVICE: int = 0

if __name__ == '__main__':
    # set config
    config.Tuning.batch_size = 32
    config.Encoder.drop_ratio = 0.5
    config.Encoder.emb_dim = 300
    config.Encoder.num_layer = 5
    config.config_name = CONFIG_NAME
    config.MAELoader.mask_rate = 0.35
    config.device = f'cuda:{DEVICE}'
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    # load model from ROM
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']

    """
    Tuning
    """
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = True if ds == 'bace' else False
            config.TuningDataset.dataset = ds
            encoder = api.get_configured_encoder()
            encoder.load_state_dict(e_states)
            encoder.enable_selfloop()
            api.tune(encoder)

    api.analyze_results_by_ratio()
