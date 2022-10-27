import context

import torch

from multiprocessing import Process
from src import config
from src import api

DEBUG: bool = False


def search_ttt_para(aug, aug_ratio, conf_ratio, num_aug, device):
    # set config
    config.config_name = f'ttt_osbn_{num_aug}{aug}{aug_ratio}_c{conf_ratio}'
    config.device = f'cuda:{device}'
    config.GNN.drop_ratio = 0
    config.TestTimeTuning.aug = aug
    config.TestTimeTuning.aug_ratio = aug_ratio
    config.TestTimeTuning.num_augmentations = num_aug
    config.TestTimeTuning.conf_ratio = conf_ratio
    config.TestTimeTuning.presaved_model_path = './data/models/graphmae80_models_v2'
    config.OneSampleBN.is_enabled = True
    if DEBUG:
        api.set_debug_mode()
    # enable OSBN
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    """
    Load saved models and conduct test time tuning
    """

    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            if config.TestTimeTuning.aug == 'dropout':
                ## setup dropout augmentation
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio

            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()

            if config.TestTimeTuning.aug == 'featM':
                ## setup feature masking augmentation
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio

            api.test_time_tuning_presaved_models(gnn)
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    Process(target=search_ttt_para, args=('dropout', 0.3, 1.0, 32, 0)).start()
    Process(target=search_ttt_para, args=('dropout', 0.4, 1.0, 32, 1)).start()
    Process(target=search_ttt_para, args=('dropout', 0.5, 1.0, 32, 1)).start()
