import context

from multiprocessing import Process

from src import config
from src import api

DEBUG: bool = False
CONFIG_NAME: str = api.get_current_filename(__file__)
DEVICE: int = 0

if __name__ == '__main__':

    config.config_name = CONFIG_NAME
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 32
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / '1116_test_ttt')
    # config.OneSampleBN.is_enabled = True
    # config.OneSampleBN.strength = 8
    config.SSF.is_enabled = True
    config.TestTimeTuning.add_prompts = True
    config.Tuning.use_node_prompt = True
    config.Tuning.use_edge_prompt = False
    config.Prompt.mode = 'add'
    config.OnlineLearning.enable_tent_bn = True
    config.OnlineLearning.optimize_tent_bn = True
    config.OnlineLearning.is_enabled = True
    config.OnlineLearning.use_shuffle = True

    if DEBUG:
        api.set_debug_mode()
    if config.OneSampleBN.is_enabled:
        api.replace_bn()
    if config.SSF.is_enabled:
        api.replace_with_ssf()

    for seed in config.loop_seeds:
        config.seed = seed
        api.set_seed(config.seed)
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            if config.TestTimeTuning.aug == 'dropout':
                config.GNN.drop_ratio = config.TestTimeTuning.aug_ratio
            config.Tuning.lr = 1e-4 if ds == 'muv' else 1e-3
            gnn = api.get_configured_gnn()
            if config.TestTimeTuning.aug == 'featM':
                gnn.mask_ratio = config.TestTimeTuning.aug_ratio
            api.cl_presaved_models(gnn)

        api.analyze_results_by_ratio()
        api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')

