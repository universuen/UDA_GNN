from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'barlow_twins_baseline'
DEVICE: int = 0

if __name__ == '__main__':
    # set config
    if DEBUG:
        api.set_debug_mode()
    else:
        config.config_name = CONFIG_NAME
        config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
        config.device = f'cuda:{DEVICE}'
    """
    Pretraining
    """
    gnn_model = api.get_configured_gnn()
    bt_model = api.get_configured_barlow_twins(gnn_model)
    api.pretrain(bt_model)
    """
    Tuning
    """
    original_states = bt_model.gnn.state_dict()
    for seed in range(10):
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            bt_model.gnn.load_state_dict(original_states)
            api.tune(config.TuningDataset.dataset, bt_model.gnn)
