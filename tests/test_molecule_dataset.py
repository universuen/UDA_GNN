import context

from src import MoleculeDataset

if __name__ == '__main__':
    dataset = MoleculeDataset()
    print(dataset[0])
