from torch_geometric.loader import DataLoader

import src
from src import config

if __name__ == '__main__':
    config.config_name = 'test'
    src.api.set_seed(config.seed)
    dataset = src.dataset.MoleculeAugDataset(
        dataset=config.PretrainingDataset.dataset,
        aug_1=config.PretrainingDataset.aug_1,
        aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
        aug_2=config.PretrainingDataset.aug_2,
        aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
        use_original=config.PretrainingDataset.use_original,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    print(next(iter(loader)))
