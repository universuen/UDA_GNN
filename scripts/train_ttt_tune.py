import context

import torch

from multiprocessing import Process
from src import config
from src import api

DEBUG: bool = False


def search_ttt_para(aug, aug_ratio, conf_ratio, num_aug, device):
    # set config
    config.config_name = f'test_ttt_{num_aug}{aug}{aug_ratio}_c{conf_ratio}'
    config.device = f'cuda:{device}'
    config.GNN.drop_ratio = 0.5
    config.TestTimeTuning.aug = aug
    config.TestTimeTuning.aug_ratio = aug_ratio
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
            if config.TestTimeTuning.aug == 'dropout':
                ## setup dropout augmentation
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio

            config.Tuning.use_lr_scheduler = ds == 'bace' 
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()
            
            if config.TestTimeTuning.aug == 'featM':
                ## setup feature masking augmentation
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio

            api.test_time_tuning_presaved_models(gnn)
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')

if __name__ == '__main__':
    # if False:
    # for aug in ['random', 'dropN']:
    #     # for conf_ratio in [0.5, 1]:
    #     Process(
    #             target=search_ttt_para,
    #             args=(aug, 0.5, 32, 2),
    #             ).start()
    #     Process(
    #             target=search_ttt_para,
    #             args=(aug, 1.0, 32, 3),
    #             ).start()
    
    if False:
        Process(target=search_ttt_para, args=('dropN', 0.05, 1, 32, 0)).start()
        Process(target=search_ttt_para, args=('subgraph', 0.95, 1, 32, 5)).start()
        Process(target=search_ttt_para, args=('dropE', 0.1, 1, 32, 2)).start()
        Process(target=search_ttt_para, args=('dropE', 0.05, 1, 32, 2)).start()

        Process(target=search_ttt_para, args=('dropout', 0.4, 1, 32, 1)).start()
        Process(target=search_ttt_para, args=('dropout', 0.5, 1, 32, 2)).start()
        Process(target=search_ttt_para, args=('dropout', 0.3, 0.5, 64, 1)).start()
        Process(target=search_ttt_para, args=('dropout', 0.2, 0.5, 64, 2)).start()
    
    search_ttt_para('featM', 0.2, 1, 32, 1)
    # Process(target=search_ttt_para, args=('maskN', 0.2, 1, 32, 1)).start()