import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0

if __name__ == '__main__':
    # set configs
    config.config_name = CONFIG_NAME
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / 'graphmae80_models_v2')
    config.device = f'cuda:{DEVICE}'
    config.SSF.is_enabled = True

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    if config.SSF.is_enabled:
        api.replace_with_ssf()

    # TTT
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TestTimeTuning.add_prompts = True
            config.Tuning.use_node_prompt = True
            config.Tuning.use_edge_prompt = False
            config.Prompt.mode = 'add'
            config.TuningDataset.dataset = ds
            config.TuningLoader.num_workers = 2
            config.Tuning.batch_size = 32
            config.Encoder.drop_ratio = 0.5
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'

            encoder = api.get_configured_encoder()
            encoder.enable_selfloop()

            api.test_time_tuning_presaved_models(encoder)

    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')
