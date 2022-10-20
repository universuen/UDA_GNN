from __future__ import annotations

from src.original.trans_bt.loader import MoleculeDataset_aug_v2
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

    def download(self):
        return super().download()
