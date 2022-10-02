import context

from torch_geometric.loader import DataLoader

from src import MoleculeDataset
from src.model.gnn import EqvGNN
from src import config
from src.model.pretraining.barlow_twins import BarlowTwins

if __name__ == '__main__':
    config.config_name = 'test'
    loader = DataLoader(
        MoleculeDataset(),
        batch_size=32,
    )
    model = BarlowTwins(EqvGNN())
    b_1, b_2, _ = next(iter(loader))
    print(model(b_1, b_2))


