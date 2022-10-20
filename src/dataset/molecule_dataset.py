from src.original.trans_bt.loader import MoleculeDataset as _MoleculeDataset
from src.dataset import Dataset


class MoleculeDataset(_MoleculeDataset, Dataset):
    def __init__(
            self,
            root: str,
            dataset: str,
    ):
        super().__init__(
            root=root,
            dataset=dataset,
        )

    def download(self):
        return super().download()
