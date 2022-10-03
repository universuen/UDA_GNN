from __future__ import annotations

from src.dataset.molecule_aug_dataset import MoleculeAugDataset
from src.types import Dataset


class IntegratedAugDataset(Dataset):
    def __init__(
            self,
            datasets: list[str],
            aug_1: str = "none",
            aug_ratio_1: int | float = None,
            aug_2: str = "none",
            aug_ratio_2: int | float = None,
            use_original: bool = False,
    ):
        self.datasets = {
            name: MoleculeAugDataset(name, aug_1, aug_ratio_1, aug_2, aug_ratio_2, use_original)
            for name in datasets
        }
        self.sizes = {
            name: len(self.datasets[name])
            for name in datasets
        }

    def __len__(self):
        return sum(self.sizes.values())

    def __getitem__(self, item: int):
        for name, size in self.sizes.items():
            if item < size:
                return self.datasets[name][item]
            else:
                item -= size
        if item >= 0:
            raise ValueError("Index out of range")
