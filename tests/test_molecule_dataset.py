import context

from src.dataset import MoleculeDataset

if __name__ == '__main__':
    dataset = MoleculeDataset()
    print(dataset[0])
