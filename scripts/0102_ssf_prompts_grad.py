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
    # set config
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.num_augmentations = 64
    config.TestTimeTuning.save_epoch = 10
    config.TestTimeTuning.eval_epoch = 10
    config.AdvAug.is_enabled = True
    config.Tuning.use_node_prompt = True
    config.TestTimeTuning.add_prompts = True
    config.Prompt.uniform_init_interval = [-config.AdvAug.step_size, config.AdvAug.step_size]
    config.AdvAug.num_iterations = 4
    config.Prompt.use_ssf_prompt = True
    config.AdvAug.w_updating_strategy = 'grad'
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    # load models from ROM
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']

    """
    Tune and save the model
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
            api.flag_tune_and_save_models(encoder)
        api.analyze_results_by_ratio()

    """
    ADV TTT
    """
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 32
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / CONFIG_NAME)
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = 8
    config.AdvAug.is_enabled = True
    config.AdvAug.num_iterations = 5
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
            gnn = api.get_configured_gnn()

            if config.TestTimeTuning.aug == 'featM':
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio

            api.test_time_tuning_presaved_models_only_last(gnn)

        api.analyze_results_by_ratio()
        api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')
