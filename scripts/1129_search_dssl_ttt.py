import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)


def search_para(mask_rate, pretraining_epochs, device):
    # set configs
    config.config_name = f'{CONFIG_NAME}_mr{mask_rate}_pe{pretraining_epochs}'
    config.device = f'cuda:{device}'
    config.OneSampleBN.is_enabled = True
    config.OneSampleBN.strength = 8

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()

    # load models from ROM
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']
    d_states = states_dict['decoder']

    # UDA and TTT
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            """
            Downstream Pretraining
            """
            config.PretrainingDataset.dataset = ds
            config.Pretraining.lr = 1e-3
            config.Pretraining.epochs = pretraining_epochs
            config.Pretraining.batch_size = 128
            config.Pretraining.save_epoch = config.Pretraining.epochs
            config.MAELoader.mask_rate = mask_rate
            config.Encoder.drop_ratio = 0

            encoder = api.get_configured_encoder()
            decoder = api.get_configured_decoder()
            encoder.load_state_dict(e_states)
            decoder.load_state_dict(d_states)

            api.train_mae(encoder, decoder)
            """
            TTT
            """
            config.TuningDataset.dataset = ds
            config.TuningLoader.num_workers = 2
            config.Tuning.batch_size = 32
            config.TestTimeTuning.aug = 'dropout'
            config.TestTimeTuning.aug_ratio = 0.5
            config.TestTimeTuning.num_iterations = 1
            config.TestTimeTuning.num_augmentations = 32
            config.Encoder.drop_ratio = 0.5
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'

            new_encoder = api.get_configured_encoder()
            new_encoder.load_state_dict(encoder.state_dict())
            new_encoder.enable_selfloop()

            api.test_time_tuning(new_encoder)

    api.analyze_results_by_ratio()
    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    for mr in (0.25, 0.40, 0.45):
        for d, pe in enumerate((10, 20, 50, 100)):
            Process(
                target=search_para,
                args=(mr, pe, d),
            ).start()
