import torch
import torch.utils.data as tud
import datasets as ds
import arguments as ag


class CachedBatchDataset(tud.Dataset):
    def __init__(self, all_data, all_samples: [ds.SampleBase]):
        super().__init__()
        if isinstance(all_data, torch.Tensor):
            self.all_data = all_data.to(ag.Arguments.device)
        else:
            self.all_data = all_data
        self.all_samples = all_samples

    def __getitem__(self, index: int) -> ([[[float]]], int):
        datum = self.all_data[index]
        label = self.all_samples[index].label

        return datum, label

    def __len__(self) -> int:
        return len(self.all_samples)
