import context

import torch

from multiprocessing import Process
from src import config
from src import api

DEBUG: bool = False


def search_ttt_para(aug, conf_ratio, num_aug, device):
    # set config
    config.config_name = f'test_ttt_{num_aug}{aug}_c{conf_ratio}'
    config.device = f'cuda:{device}'
    config.GNN.drop_ratio = 0.5
    config.TestTimeTuning.aug = aug
    config.TestTimeTuning.num_augmentations = num_aug
    config.TestTimeTuning.conf_ratio = conf_ratio
    config.TestTimeTuning.presaved_model_path = './data/models/graphmae80_models_v2'
    if DEBUG:
        api.set_debug_mode()

    """
    Load saved models and conduct test time tuning
    """
    
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = 0.5
            config.Tuning.use_lr_scheduler = ds == 'bace' 
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()
            api.test_time_tuning_presaved_models(gnn)
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')

if __name__ == '__main__':
    # if False:
    for aug in ['random', 'dropN']:
        # for conf_ratio in [0.5, 1]:
        Process(
                target=search_ttt_para,
                args=(aug, 0.5, 32, 2),
                ).start()
        Process(
                target=search_ttt_para,
                args=(aug, 1.0, 32, 3),
                ).start()
    # search_ttt_para('random', 1, 32, 3)