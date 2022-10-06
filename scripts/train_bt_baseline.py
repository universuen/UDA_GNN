import context

import config
import utils

CONFIG_NAME = 'barlow_twins_baseline'
DEVICE = 0

if __name__ == '__main__':
    """
    Pretraining
    """
    # set config name
    config.config_name = CONFIG_NAME
    # set device
    config.device = f'cuda:{DEVICE}'
    # set seed
    utils.set_seed(config.seed)
    gnn_model = utils.load_gnn()
    bt_model = utils.load_barlow_twins(gnn_model)
    utils.pretrain(bt_model)
    """
    Tuning
    """
    original_states = bt_model.gnn.state_dict()
    for seed in range(10):
        config.seed = seed
        utils.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            bt_model.gnn.load_state_dict(original_states)
            utils.tune(config.TuningDataset.dataset, bt_model.gnn)
