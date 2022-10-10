from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'uda_trans'
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
    Pretraining 1
    """
    trans_model = api.get_configured_graph_trans()
    bt_model = api.get_configured_barlow_twins(trans_model)
    api.pretrain(bt_model)

    original_bt_states = bt_model.state_dict()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            """
            Pretraining 2
            """
            config.PretrainingDataset.dataset = ds
            config.GraphTrans.drop_ratio = 0
            trans_model = api.get_configured_graph_trans()
            bt_model = api.get_configured_barlow_twins(trans_model)
            bt_model.load_state_dict(original_bt_states)
            api.pretrain(bt_model)
            """
            Tuning
            """
            # set specific config
            config.TuningDataset.dataset = ds
            config.Tuning.lr = 1e-5 if ds == 'muv' else 1e-4
            config.GraphTrans.drop_ratio = 0.3 if ds == 'clintox' else 0.5
            config.Tuning.epochs = 128 if ds == 'clintox' else 100
            # tune
            trans_model = api.get_configured_graph_trans()
            trans_model.load_state_dict(
                bt_model.gnn.state_dict()
            )
            api.tune(trans_model)
    api.analyze_results()