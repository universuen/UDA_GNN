import context

import torch

from multiprocessing import Process
from src import config
from src import api

DEBUG: bool = False


def search_ttt_para(aug, aug_ratio, conf_ratio, num_aug, device):
    # set config
    config.config_name = f'ttt_{num_aug}{aug}{aug_ratio}_c{conf_ratio}'
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
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio
            config.Tuning.use_lr_scheduler = ds == 'bace' 
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()
            api.test_time_tuning_presaved_models(gnn)
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


def search_ttt_prompt_para(aug, aug_ratio, include_linear, conf_ratio=1, num_aug=32, device=0):
    # set config
    config.config_name = f'ttt_prompt_linear{include_linear}_{num_aug}{aug}{aug_ratio}_c{conf_ratio}'
    config.device = f'cuda:{device}'
    config.GNN.drop_ratio = 0.5
    config.TestTimeTuning.add_prompts = True
    
    config.TestTimeTuning.aug = aug
    config.TestTimeTuning.aug_ratio = aug_ratio
    config.TestTimeTuning.num_augmentations = num_aug
    config.TestTimeTuning.conf_ratio = conf_ratio
    config.TestTimeTuning.include_linear = include_linear
    # config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.presaved_model_path = '/gpu-work/gp9000/gp0900/next/ysun/UDA_GNN/data/models/ttt_add_prompts'
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
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()
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
        # Process(target=search_ttt_para, args=('subgraph', 0.95, 1, 32, 5)).start()
        Process(target=search_ttt_para, args=('dropE', 0.1, 1, 32, 2)).start()
        Process(target=search_ttt_para, args=('dropE', 0.05, 1, 32, 2)).start()

        Process(target=search_ttt_para, args=('dropout', 0.1, 1, 32, 0)).start()
        Process(target=search_ttt_para, args=('dropout', 0.2, 1, 32, 0)).start()
        Process(target=search_ttt_para, args=('dropout', 0.3, 1, 32, 1)).start()
        Process(target=search_ttt_para, args=('dropout', 0.05, 1, 32, 1)).start()
    
    ## batch1
    if False:
        Process(target=search_ttt_prompt_para, args=('dropout', 0.1, True)).start()
        Process(target=search_ttt_prompt_para, args=('dropout', 0.2, True)).start()

    
    ## batch2
    if False:
        Process(target=search_ttt_prompt_para, args=('dropout', 0.1, False)).start()
        Process(target=search_ttt_prompt_para, args=('dropout', 0.2, False)).start()

    if False:
        ## batch2
        Process(target=search_ttt_prompt_para, args=('dropN', 0.1, True)).start()
        Process(target=search_ttt_prompt_para, args=('dropN', 0.2, True)).start()

    if True:
        ## batch3
        Process(target=search_ttt_prompt_para, args=('dropN', 0.1, False)).start()
        Process(target=search_ttt_prompt_para, args=('dropN', 0.2, False)).start()