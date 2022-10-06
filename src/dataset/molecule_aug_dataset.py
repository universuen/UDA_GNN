from __future__ import annotations

from src import types, config
from src.original.loader import MoleculeDataset_aug_v2


class MoleculeAugDataset(MoleculeDataset_aug_v2, types.Dataset):
    def __init__(
            self,
            dataset: str = None,
            aug_1: str = "none",
            aug_ratio_1: int | float = None,
            aug_2: str = "none",
            aug_ratio_2: int | float = None,
            use_original: bool = False,
    ):
        super().__init__(
            root=str(config.Paths.datasets / dataset),
            dataset=dataset,
            aug1=aug_1,
            aug2=aug_2,
            aug_ratio1=aug_ratio_1,
            aug_ratio2=aug_ratio_2,
            use_original=use_original,
        )

    def download(self):
        return super().download()
