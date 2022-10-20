import context

import torch

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'graphmae80_models'
DEVICE: int = 1

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.TestTimeTuning.aug = 'dropN'
    config.TestTimeTuning.num_augmentations = 64
    config.TestTimeTuning.save_epoch = 10
    config.TestTimeTuning.eval_epoch = 10
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    gnn_model = api.get_configured_gnn()
    bt_model = api.get_configured_barlow_twins(gnn_model)
    state_dict = torch.load(
        config.Paths.models / 'graphmae_m35' / 'model_80.pth',
        map_location=lambda storage, loc: storage,
    )
    bt_model.train()
    """
    Tuning
    """
    # original_states = bt_model.gnn.state_dict()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = 0.5
            config.Tuning.use_lr_scheduler = ds == 'bace' 
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            bt_model.gnn.load_state_dict(state_dict)
            api.tune_and_save_models(bt_model.gnn)
    api.analyze_results_by_ratio(item_name='te_auc')
