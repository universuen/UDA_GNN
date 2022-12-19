import context

from copy import deepcopy
from multiprocessing import Process

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0


def search_para(lr_times: int, adv_step_size: float, adv_num_iter: int, device: int):
    config.config_name = f'{CONFIG_NAME}_lrt_{lr_times}_ss_{adv_step_size}_ni_{adv_num_iter}'
    config.device = f'cuda:{device}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 32
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / '1216_flag_tune_n4')
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = 8
    config.AdvAug.is_enabled = True
    config.AdvAug.step_size = adv_step_size
    config.AdvAug.num_iterations = adv_num_iter
    config.Tuning.use_node_prompt = True
    config.TestTimeTuning.add_prompts = True
    config.Prompt.mode = 'add'
    config.Prompt.uniform_init_interval = [-config.AdvAug.step_size, config.AdvAug.step_size]

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
            config.Tuning.lr *= lr_times
            gnn = api.get_configured_gnn()

            if config.TestTimeTuning.aug == 'featM':
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio

            config.TestTimeTuning.save_epoch = 100
            api.test_time_tuning_presaved_models_only_last(gnn)

        api.analyze_results_by_ratio()
        api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    cnt = 0
    lrt = [2, 3]
    ss = [0.01, 0.02]
    ni = [3, 4, 5]
    for i in lrt:
        for j in ss:
            for k in ni:
                search_para(i, j, k, 0)
