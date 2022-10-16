from __future__ import annotations

from itertools import repeat

from torch_geometric.data import Data

from src.original.loader import MoleculeDataset_aug_v2
from src.dataset import Dataset


class MoleculeAugDataset(MoleculeDataset_aug_v2, Dataset):
    def __init__(
            self,
            root: str,
            dataset: str = None,
            aug_1: str = "none",
            aug_ratio_1: int | float = None,
            aug_2: str = "none",
            aug_ratio_2: int | float = None,
            use_original: bool = False,
    ):
        super().__init__(
            root=root,
            dataset=dataset,
            aug1=aug_1,
            aug2=aug_2,
            aug_ratio1=aug_ratio_1,
            aug_ratio2=aug_ratio_2,
            use_original=use_original,
        )

    def get_data(self, idx):
        data = Data()
        for key in self.data.keys:
            if key in ('y', 'fold'):
                continue
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    def download(self):
        return super().download()
