from __future__ import annotations

import random

import src.dataset
from src.dataset import Dataset, MoleculeAugDataset


class DualDataset(Dataset):
    def __init__(
            self,
            zinc_ds: src.dataset.MoleculeAugDataset,
            other_ds: src.dataset.MoleculeAugDataset,
    ):
        super().__init__()
        self.zinc_ds = zinc_ds
        self.other_ds = other_ds
        self.half_len = len(self.other_ds)
        self.zinc_idxes = random.sample(range(len(self.zinc_ds)), self.half_len)

    def __len__(self):
        return 2 * self.half_len

    def __getitem__(self, item):
        if item < self.half_len:
            record = self.zinc_ds[self.zinc_idxes[item]]
        else:
            record = self.other_ds[item - self.half_len]
        b1, b2, other = record
        # delete extra attributes that may cause crash
        delattr(b1, 'fold')
        delattr(b1, 'y')
        delattr(b2, 'fold')
        delattr(b2, 'y')
        return b1, b2, other
