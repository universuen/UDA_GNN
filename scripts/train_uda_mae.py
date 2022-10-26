import torch

import context

from copy import deepcopy

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'test_uda_mae'
DEVICE: int = 0
FROM_SCRATCH: bool = True

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.Pretraining.batch_size = 256
    config.MAELoader.mask_rate = 0.35
    config.device = f'cuda:{DEVICE}'
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining 1
    """
    config.Pretraining.epochs = 1
    config.Pretraining.save_epoch = 1
    encoder = api.get_configured_encoder()
    decoder = api.get_configured_decoder()
    if FROM_SCRATCH:
        api.train_mae(encoder, decoder)
    else:
        cache_path = '/storage_fast/zyliu/code/UDA_GNN/data/models/test_uda_mae/mae_zinc_standard_agent_e1_s0.pt'
        mae_state_dict = torch.load(cache_path, map_location=torch.device('cpu'))
        encoder.load_state_dict(mae_state_dict['encoder'])
        decoder.load_state_dict(mae_state_dict['decoder'])

    ## load the 80th epoch checkpoint
    cache_path = '/storage_fast/zyliu/code/UDA_GNN/data/models/uda_mae/mae_zinc_standard_agent_e80_s0.pt'
    mae_state_dict = torch.load(cache_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(mae_state_dict['encoder'])
    decoder.load_state_dict(mae_state_dict['decoder'])

    original_e_states = deepcopy(encoder.state_dict())
    original_d_states = deepcopy(decoder.state_dict())
    
    # set pretrain phase2 config
    config.Pretraining.epochs = 50
    config.TuningLoader.num_workers = 2
    config.Tuning.batch_size = 32
    
    encoder = api.get_configured_encoder()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            encoder.load_state_dict(original_e_states)
            decoder.load_state_dict(original_d_states)
            """
            Pretraining 2
            """
            config.PretrainingDataset.dataset = ds
            config.Pretraining.save_epoch = config.Pretraining.epochs
            api.train_mae(encoder, decoder)
            """
            Tuning
            """
            config.Encoder.drop_ratio = 0.5
            new_encoder = api.get_configured_encoder()
            new_encoder.load_state_dict(encoder.state_dict())
            new_encoder.enable_selfloop()

            config.TuningDataset.dataset = ds
            config.Tuning.lr = 1e-3 if ds != 'muv' else 1e-4
            config.Tuning.use_lr_scheduler = ds == 'bace'
            api.tune_and_save_models(new_encoder)
    api.analyze_results_by_ratio()
