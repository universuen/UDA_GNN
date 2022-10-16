from __future__ import annotations

import random

from src.dataset import Dataset, MoleculeAugDataset


class DualDatasetV2(Dataset):
    def __init__(
            self,
            dataset_path: str,
            dataset: str = None,
            aug_1: str = "none",
            aug_ratio_1: int | float = None,
            aug_2: str = "none",
            aug_ratio_2: int | float = None,
            use_original: bool = False,
    ):
        super().__init__()
        self.zinc_ds = MoleculeAugDataset(
            root=f'{dataset_path}/zinc_standard_agent',
            dataset='zinc_standard_agent',
            aug_1=aug_1,
            aug_2=aug_2,
            aug_ratio_1=aug_ratio_1,
            aug_ratio_2=aug_ratio_2,
            use_original=use_original,
        )
        self.other_ds = MoleculeAugDataset(
            root=f'{dataset_path}/{dataset}',
            dataset=dataset,
            aug_1=aug_1,
            aug_2=aug_2,
            aug_ratio_1=aug_ratio_1,
            aug_ratio_2=aug_ratio_2,
            use_original=use_original,
        )
        self.half_len = len(self.other_ds)
        self.zinc_len = len(self.zinc_ds)

    def __len__(self):
        return 2 * self.half_len

    def __getitem__(self, item):
        if item < self.half_len:
            record = self.zinc_ds[random.randint(0, self.zinc_len)]
        elif self.half_len <= item < 2 * self.half_len:
            record = self.other_ds[item - self.half_len]
        else:
            raise IndexError
        b1, b2, other = record
        # delete extra attributes that may cause crash
        # delattr(b1, 'fold')
        # delattr(b1, 'y')
        # delattr(b2, 'fold')
        # delattr(b2, 'y')
        return b1, b2, other
