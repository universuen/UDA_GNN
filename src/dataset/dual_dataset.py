from __future__ import annotations

import random

from src.dataset import Dataset, MoleculeAugDataset


class DualDataset(Dataset):
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
        self.zinc_idxes = random.sample(range(len(self.zinc_ds)), self.half_len)

    def __len__(self):
        return 2 * self.half_len

    def __getitem__(self, item):
        if item < self.half_len:
            return self.zinc_ds[self.zinc_idxes[item]]
        else:
            return self.other_ds[item - self.half_len]
