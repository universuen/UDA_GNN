from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'barlow_twins_baseline'
DEVICE: int = 0

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.GNN.drop_ratio = 0.5
    config.device = f'cuda:{DEVICE}'
    if DEBUG:
        api.set_debug_mode()

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
            config.GNN.drop_ratio = 0.5
            bt_model.gnn.load_state_dict(original_states)
            api.tune(bt_model.gnn)
    api.analyze_results()
