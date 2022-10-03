from src import utils, dataset
import config

if __name__ == '__main__':
    dataset = dataset.MoleculeDataset(config.TuningDataset.dataset)
    print(utils.create_loaders(dataset))
