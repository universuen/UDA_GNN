import context
from src import config
from src import api
import src
import torch.nn as nn

DEVICE: int = 0

if __name__ == '__main__':
    config.datasets = [
        "clintox",
        'bace',
        'bbbp',
    ]
    config.config_name = 'test3'
    config.device = f'cuda:{DEVICE}'
    config.Encoder.drop_ratio = 0.5
    config.TestTimeTuning.aug = 'dropout'
    config.TestTimeTuning.aug_ratio = 0.5
    config.TestTimeTuning.num_iterations = 1
    config.TestTimeTuning.num_augmentations = 32
    config.TestTimeTuning.presaved_model_path = str(config.Paths.models / '1116_test_ttt')
    
    config.OnlineLearning.use_shuffle = True
    config.TestTimeTuning.save_epoch = 90
    config.Tuning.batch_size = 32

    
    config.BatchRenormalization.is_enabled = True
    config.SSF.is_enabled = False
    config.TestTimeTuning.tuning = True
    
    config.OneSampleBN.is_enabled = False
    config.OneSampleBN.strength = 4
    

    if config.OneSampleBN.is_enabled:
        api.replace_bn()
        if config.SSF.is_enabled:
            api.replace_with_ssf()
    if config.BatchRenormalization.is_enabled:
        api.use_tbr_bn()
        if config.SSF.is_enabled:
            nn.Linear = src.model.SSLinear

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
            api.ol_presaved_models_v2(gnn)

        api.analyze_results_by_ratio()
        api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')
