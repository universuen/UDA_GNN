import src.model
from src import config
from src import api

if __name__ == '__main__':
    if config.OneSampleBN.is_enabled:
        api.replace_bn()

    # when set to strength
    api.set_bn_prior()
    # when set to 1
    api.set_bn_prior(1)
    """
    Pretraining
    """
    api.set_debug_mode()
    gnn_model = api.get_configured_gnn()
    bt_model = api.get_configured_barlow_twins(gnn_model)
    api.pretrain(bt_model)
    """
    Tuning
    """
    original_states = bt_model.gnn.state_dict()
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            bt_model.gnn.load_state_dict(original_states)
            api.test_time_tuning(bt_model.gnn)
    api.analyze_results([1, 2])
