import context

from copy import deepcopy
from multiprocessing import Process

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0
print(CONFIG_NAME)


def tune_and_save():
    # set config
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.num_augmentations = 64
    config.TestTimeTuning.save_epoch = 10
    config.TestTimeTuning.eval_epoch = 10
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    # load models from ROM
    encoder = api.get_configured_encoder()
    decoder = api.get_configured_decoder()
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(states_dict['encoder'])
        decoder.load_state_dict(states_dict['decoder'])
    e_states = deepcopy(encoder.state_dict())
    d_states = deepcopy(decoder.state_dict())

    """
    Tune and save the model
    """
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            encoder.load_state_dict(e_states)
            decoder.load_state_dict(d_states)
            config.TuningDataset.dataset = ds
            config.Tuning.use_lr_scheduler = ds == 'bace'
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            encoder.enable_selfloop()
            api.tune_and_save_models(encoder)
    api.analyze_ttt_results_by_ratio(item_name='te_auc')


def search_para(num_iter: int, num_aug: int, d: int):
    config.config_name = f'1115_ttt_ni{num_iter}_na{num_aug}'
    config.device = f'cuda:{d}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = num_iter
    config.TestTimeTuning.num_augmentations = num_aug
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / CONFIG_NAME)
    if DEBUG:
        api.set_debug_mode()
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
    tune_and_save()
    # for num_iter in (1, 2, 5):
    #     for d, num_aug in enumerate((8, 16, 32)):
    #         Process(
    #             target=search_para,
    #             args=(num_iter, num_aug, d),
    #         ).start()
