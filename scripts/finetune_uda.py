import context

import torch
from torch_geometric.loader import DataLoader

import src
import config

CONFIG_NAME = 'finetune_UDA'
DEVICE = 2

if __name__ == '__main__':
    # set config
    config.config_name = CONFIG_NAME
    config.Pretraining.batch_size = 256
    # set others
    config.device = f'cuda:{DEVICE}'
    src.utils.set_seed(config.seed)
    logger = src.Logger('main')

    for seed in config.loop_seeds:
        config.seed = seed
        src.utils.set_seed(config.seed)
        for ds in config.datasets:
            model = src.model.pretraining.BarlowTwins()
            state_dict = torch.load(
                config.Paths.models / 'base_bt_model.pt',
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)
            model.to(config.device)
            model.train()
            """
            Pretraining
            """
            config.PretrainingDataset.dataset = ds
            config.GNN.drop_ratio = config.Pretraining.gnn_dropout_ratio
            loader = DataLoader(
                dataset=src.dataset.MoleculeAugDataset(
                    dataset=config.PretrainingDataset.dataset,
                    aug_1=config.PretrainingDataset.aug_1,
                    aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
                    aug_2=config.PretrainingDataset.aug_2,
                    aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
                    use_original=config.PretrainingDataset.use_original,
                ),
                batch_size=config.Pretraining.batch_size,
                shuffle=True,
                num_workers=config.PretrainingLoader.num_workers,
                pin_memory=config.PretrainingLoader.pin_memory,
                drop_last=config.PretrainingLoader.drop_last,
                worker_init_fn=config.PretrainingLoader.worker_init_fn,
            )
            logger.log_all_config()
            optimizer = torch.optim.Adam(model.parameters(), config.Pretraining.lr)
            for e in range(config.Pretraining.epochs):
                for idx, (b1, b2, _) in enumerate(loader):
                    # torch.cuda.empty_cache()
                    b1.to(config.device)
                    b2.to(config.device)
                    optimizer.zero_grad()
                    loss = model(b1, b2)
                    loss.backward()
                    optimizer.step()
            """
            Tuning
            """
            config.TuningDataset.dataset = ds
            config.GNN.drop_ratio = config.Tuning.gnn_dropout_ratio
            logger.log_all_config()
            src.utils.tune(config.TuningDataset.dataset, model.gnn)
