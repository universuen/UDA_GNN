from __future__ import annotations

from torch_geometric.data import Batch

from src.original.loader import augment
from src.dataset.molecule_dataset import MoleculeDataset
from torch.utils.data import Dataset

class TTTAugDataset(Dataset):
    def __init__(
            self,
            dataset: MoleculeDataset,
            num_augmentations: int,
            aug: str = "none",
            aug_ratio: int | float = None,

    ):  
        self.dataset = dataset
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.num_augmentations = num_augmentations

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        augmented_data = [
            augment(data.clone(), self.aug, self.aug_ratio)
            for _ in range(self.num_augmentations)
        ]
        augmented_data = Batch.from_data_list(augmented_data)
        return data, augmented_data
