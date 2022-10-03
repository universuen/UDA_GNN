import context
from torch_geometric.loader import DataLoader

import config
from src import MoleculeDataset
from src.model.gnn import EqvGNN

if __name__ == '__main__':
    config.config_name = 'test'
    loader = DataLoader(
        MoleculeDataset(
            dataset=config.PretrainingDataset.dataset,
            aug_1=config.PretrainingDataset.aug_1,
            aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
            aug_2=config.PretrainingDataset.aug_2,
            aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
            use_original=config.PretrainingDataset.use_original,
        ),
        batch_size=32,
    )
    model = EqvGNN()
    b_1, b_2, _ = next(iter(loader))
    print(model(b_1))
    print(model(b_2))
