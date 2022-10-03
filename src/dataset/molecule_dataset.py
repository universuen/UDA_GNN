from src import types
import config
from src.original.loader import MoleculeDataset as _MoleculeDataset


class MoleculeDataset(_MoleculeDataset, types.Dataset):
    def __init__(
            self,
            dataset: str = None,
    ):
        super().__init__(
            root=str(config.Paths.datasets / dataset),
            dataset=dataset,
        )

    def download(self):
        return super().download()
