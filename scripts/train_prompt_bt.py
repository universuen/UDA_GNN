import context
from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = 'prompt_bt'
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
    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            prompt = api.get_prompt()
            api.tune_with_prompt(bt_model.gnn, prompt)
    api.analyze_results()
