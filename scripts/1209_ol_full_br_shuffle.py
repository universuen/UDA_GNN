import context

from copy import deepcopy
from multiprocessing import Process

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0

if __name__ == '__main__':
    config.datasets = [
        "clintox",
        'bace',
        'bbbp',
    ]
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 32
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / '1116_test_ttt')
    # config.OneSampleBN.is_enabled = True
    # config.OneSampleBN.strength = 8
    config.BatchRenormalization.is_enabled = True
    config.OnlineLearning.use_shuffle = True

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    if config.BatchRenormalization.is_enabled:
        api.use_tbr_bn()

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
            api.ol_presaved_models(gnn)

        api.analyze_results_by_ratio()
        api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')
