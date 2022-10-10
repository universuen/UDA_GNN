from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'trans'
DEVICE: int = 0

if __name__ == '__main__':
    # set general config
    config.config_name = CONFIG_NAME
    config.Pretraining.use_graph_trans = True
    config.Pretraining.lr = 1e-4
    config.GraphTrans.drop_ratio = 0
    config.device = f'cuda:{DEVICE}'
    config.BarlowTwins.sizes = (128, 1200, 1200, 1200)
    if DEBUG:
        api.set_debug_mode()

    """
    Pretraining
    """
    trans_model = api.get_configured_graph_trans()
    bt_model = api.get_configured_barlow_twins(trans_model)
    api.pretrain(bt_model)

    original_states = bt_model.gnn.state_dict()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            trans_model = api.get_configured_graph_trans()
            bt_model = api.get_configured_barlow_twins(trans_model)
            bt_model.load_state_dict(original_states)
            """
            Tuning
            """
            # set specific config
            config.TuningDataset.dataset = ds
            config.Tuning.lr = 1e-5 if ds == 'muv' else 1e-4
            config.GraphTrans.drop_ratio = 0.3 if ds == 'clintox' else 0.5
            config.Tuning.epochs = 300 if ds == 'clintox' else 100
            # tune
            trans_model = bt_model.gnn
            api.tune(trans_model)
    api.analyze_results()