import torch.utils.data as tud
import datasets as ds


class WholeDataset(tud.Dataset):
    def __init__(self, params: ds.DatasetParams):
        super().__init__()
        self.params = params

    def __len__(self) -> int:
        size = len(self.params.samples)
        return size
