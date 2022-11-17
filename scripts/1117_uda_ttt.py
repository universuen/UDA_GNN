import context

from multiprocessing import Process
from copy import deepcopy

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)


def search_uda_ttt_pretrain_para(mask_rate, pretraining_epochs, device):
    # set configs
    config.config_name = f'{CONFIG_NAME}_mr{mask_rate}_pe{pretraining_epochs}'
    config.device = f'cuda:{device}'
    config.OneSampleBN.is_enabled = True
    config.GNN.drop_ratio = 0
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.3
    config.TestTimeTuning.num_augmentations = 64
    config.TestTimeTuning.conf_ratio = 1.0
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / 'graphmae80_models_v2')
    config.OneSampleBN.is_enabled = True
    config.MAELoader.mask_rate = mask_rate
    config.Pretraining.epochs = pretraining_epochs
    config.Encoder.drop_ratio = 0.5
    config.TuningLoader.num_workers = 2
    config.Tuning.batch_size = 32
    config.Pretraining.save_epoch = config.Pretraining.epochs
    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()

    # load models from ROM
    encoder = api.get_configured_encoder()
    decoder = api.get_configured_decoder()
    with open(config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(states_dict['encoder'])
        decoder.load_state_dict(states_dict['decoder'])
    e_states = deepcopy(encoder.state_dict())
    d_states = deepcopy(decoder.state_dict())

    # UDA and TTT
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            encoder.load_state_dict(e_states)
            decoder.load_state_dict(d_states)
            """
            UDA
            """
            config.PretrainingDataset.dataset = ds
            api.train_mae(encoder, decoder)
            """
            TTT
            """
            config.TuningDataset.dataset = ds
            new_encoder = api.get_configured_encoder()
            new_encoder.load_state_dict(encoder.state_dict())
            new_encoder.enable_selfloop()
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'
            api.test_time_tuning(new_encoder)

    api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')


if __name__ == '__main__':
    for mr in (0.25, 0.35, 0.45):
        for d, pe in enumerate((10, 20, 50, 100)):
            Process(
                target=search_uda_ttt_pretrain_para,
                args=(mr, pe, d),
            ).start()
