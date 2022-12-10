import context

import torch
from multiprocessing import Process
from src import config
from src import api

DEBUG: bool = False
DEVICE: int = 0


def search_para(rate):
    # set config
    config.Tuning.batch_size = 32
    config.Encoder.drop_ratio = 0.5
    config.Encoder.emb_dim = 300
    config.Encoder.num_layer = 5
    config.Encoder.survival_rate = rate
    CONFIG_NAME: str = f'mae_m35_baseline_sd_{rate}'
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
            encoder = api.get_configured_encoder_sd()
            encoder.load_state_dict(e_states)
            encoder.enable_selfloop()
            api.tune(encoder)

    api.analyze_results_by_ratio()


if __name__ == '__main__':
    pool = []
    for rate in [0.8, 0.9]:
        pool.append(Process(target=search_para, args=(rate,), ).start())
    
    for j in pool:
        j.join()

    pool = []
    for rate in [0.7, 0.6]:
        pool.append(Process(target=search_para, args=(rate,), ).start())
    