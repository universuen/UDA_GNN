import context

from torch_geometric.loader import DataLoader

import src
import config

if __name__ == '__main__':
    config.config_name = 'test'
    src.utils.set_seed(config.seed)
    dataset = src.dataset.IntegratedAugDataset(
        datasets=['zinc_standard_agent', *config.datasets],
        aug_1=config.PretrainingDataset.aug_1,
        aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
        aug_2=config.PretrainingDataset.aug_2,
        aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
        use_original=config.PretrainingDataset.use_original,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
    )
    for i in loader:
        print(i)
