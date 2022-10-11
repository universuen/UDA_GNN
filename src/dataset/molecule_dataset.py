from src import types, config
from src.original.loader import MoleculeDataset as _MoleculeDataset
from src.dataset import Dataset


class MoleculeDataset(_MoleculeDataset, Dataset):
    def __init__(
            self,
            dataset: str,
    ):
        super().__init__(
            root=str(config.Paths.datasets / dataset),
            dataset=dataset,
        )

    def download(self):
        return super().download()
