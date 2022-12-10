import context

from copy import deepcopy
from multiprocessing import Process

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0


def search_para(device: int, prior: float):
    config.config_name = f'1130_conf_p{prior}'
    config.device = f'cuda:{device}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 64
    config.TestTimeTuning.conf_ratio = 0.5
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / '1116_test_ttt')
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = prior
    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            if config.TestTimeTuning.aug == 'dropout':
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio

            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()

            if config.TestTimeTuning.aug == 'featM':
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio

            api.test_time_tuning_presaved_models(gnn)
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    # search_para(0, 8)
    # search_para(0, 16)
    for d, p in enumerate([8, 16]):
        Process(
            target=search_para,
            args=(0, p),
        ).start()
