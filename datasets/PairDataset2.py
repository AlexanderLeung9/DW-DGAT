import numpy as np
import torch.utils.data as tud
import datasets as ds


class PairDatasetParams(object):
    def __init__(self, network_matrices: [[[[float]]]], all_samples: [ds.PDSample], base_sample_index: int):
        """
        :params network_matrices: shape=(N, 3, ROI_n, ROI_n)
        """
        self.network_matrices = network_matrices
        self.all_samples = all_samples
        self.base_sample_index = base_sample_index


class PairDataset(tud.Dataset):
    """
    A pair-wise sample dataset.
    """
    def __init__(self, params: PairDatasetParams):
        super().__init__()
        self.params = params

    def __getitem__(self, index: int) -> ([[[[float]]]], int):
        pair = [self.params.network_matrices[self.params.base_sample_index], self.params.network_matrices[index]]
        pair = np.array(pair)
        label = 1 if self.params.all_samples[self.params.base_sample_index].label == self.params.all_samples[index].label else 0
        return pair, label
