from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.original.trans_bt.loader import augment


class CLLoader(DataLoader):
    def __init__(self, dataset, aug, aug_ratio, num_augs, **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.num_augs = num_augs

    def collate_fn(self, batches):
        data_batch = Batch.from_data_list(batches)
        # build augmentations
        augmented_data = []
        for i in batches:
            augmented_data += [
                augment(
                    i.clone(),
                    aug=self.aug,
                    aug_ratio=self.aug_ratio,
                )
                for _ in range(self.num_augs)
            ]
        augmentations = Batch.from_data_list(augmented_data)

        return data_batch, augmentations
