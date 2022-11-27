import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)


def search_para(freeze_decoder: bool, dssl_lr: float, device: int):
    # set configs
    config.config_name = f'{CONFIG_NAME}_fd{freeze_decoder}_lr{lr}'
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
            config.Pretraining.lr = dssl_lr
            config.Pretraining.epochs = 10
            config.Pretraining.batch_size = 128
            config.Pretraining.save_epoch = config.Pretraining.epochs
            config.MAELoader.mask_rate = 0.35
            config.Encoder.drop_ratio = 0

            encoder = api.get_configured_encoder()
            decoder = api.get_configured_decoder()
            encoder.load_state_dict(e_states)
            decoder.load_state_dict(d_states)

            api.train_mae(encoder, decoder, freeze_decoder)
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

    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    from multiprocessing import Process
    device_matrix = [
        [0, 1, 2],
        [0, 1, 2],
    ]
    for i, fd in enumerate([True, False]):
        for j, lr in enumerate([1e-4, 5e-4, 1e-3]):
            Process(
                target=search_para,
                args=(fd, lr, device_matrix[i][j]),
            ).start()
