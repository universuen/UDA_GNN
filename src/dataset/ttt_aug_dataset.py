from __future__ import annotations

from torch_geometric.data import Batch

from src.dataset import MoleculeAugDataset


class TTTAugDataset(MoleculeAugDataset):
    def __init__(
            self,
            num_augmentations: int,
            root: str,
            dataset: str = None,
            aug: str = "none",
            aug_ratio: int | float = None,

    ):
        super().__init__(
            root=root,
            dataset=dataset,
            aug_1=aug,
            aug_2=aug,
            aug_ratio_1=aug_ratio,
            aug_ratio_2=aug_ratio,
            use_original=False,
        )
        self.num_augmentations = num_augmentations

    def get(self, idx):
        augmented_data = [
            self._MoleculeDataset_aug_v2__get(idx, self.aug1, self.aug_ratio1)
            for _ in range(self.num_augmentations)
        ]
        augmented_data = Batch.from_data_list(augmented_data)
        data = self.get_data(idx)
        return data, augmented_data

    def download(self):
        return super().download()
