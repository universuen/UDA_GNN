from pathlib import Path

from src import config
from src.original.loader import MoleculeDataset_aug_v2


class MoleculeDataset(MoleculeDataset_aug_v2):
    def __init__(
            self,
            root: Path = config.PretrainingDataset.root,
            dataset: str = config.PretrainingDataset.dataset,
            aug_1: str = config.PretrainingDataset.aug_1,
            aug_ratio_1: int | float = config.PretrainingDataset.aug_ratio_1,
            aug_2: str = config.PretrainingDataset.aug_2,
            aug_ratio_2: int | float = config.PretrainingDataset.aug_ratio_2,
            use_original: bool = config.PretrainingDataset.use_original,
    ):
        super().__init__(
            root=str(root),
            dataset=dataset,
            aug1=aug_1,
            aug2=aug_2,
            aug_ratio1=aug_ratio_1,
            aug_ratio2=aug_ratio_2,
            use_original=use_original,
        )

    def download(self):
        return super().download()
