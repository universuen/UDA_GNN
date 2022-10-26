from src.original.mae.dataloader import DataLoaderMaskingPred
from src.dataset import MoleculeDataset


class MAELoader(DataLoaderMaskingPred):
    def __init__(
            self,
            dataset: MoleculeDataset,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            mask_rate: float,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            mask_rate=mask_rate,
            mask_edge=False,
        )
