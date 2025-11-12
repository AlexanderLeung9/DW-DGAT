import os
import numpy as np
import enums as es
import datasets as ds
import arguments as ag
import utils.CommonUtils as cu


class SyntheticDatasetParams(ds.DatasetParams):
    def __init__(self, args: ag.BDArguments, samples: list[ds.SampleBase]):
        super().__init__(args.dataset_dir, samples)
        assert args.txt_indices is not None
        assert args.networks_merge_mode is not None
        self.args = args


class SyntheticDataset(ds.WholeDataset):
    def __init__(self, params: SyntheticDatasetParams):
        super().__init__(params)
        self.args: ag.BDArguments = params.args
        self.samples = params.samples

    def __getitem__(self, index: int) -> ([[[float]]], [float], [[[float]]], [[[float]]]):
        sample = self.samples[index]
        event_path = ag.BDArguments.event_parts[sample.event_index]

        file_path = os.path.join(self.args.dataset_dir, event_path, sample.No)
        assert isinstance(file_path, str)
        paths = []
        for filename_format in ag.BDArguments.TXT_FILE_NAMES:
            path = os.path.join(file_path, filename_format)
            path = path.format(sample.No)
            paths.append(path)

        networks = []
        for i in range(3):
            if i not in self.args.txt_indices:
                continue
            path = paths[i]
            matrix = np.loadtxt(path)
            # Ref 《BrainNetCNN》, 2.2. Preterm data, Paragraph 1.
            matrix = cu.min_max_scale(matrix, False)
            if self.args.networks_merge_mode == es.ENetworksMergeMode.UpperTriangle:
                indices = np.triu_indices_from(matrix, 1)
                network = matrix[indices]
            else:
                network = matrix
            networks.append(network)

        if self.args.networks_merge_mode == es.ENetworksMergeMode.L1Norm:
            networks = np.array(networks, dtype=np.float32)
            networks = np.linalg.norm(networks, ord=1, axis=2)
            networks = networks.transpose((1, 0))

        vector = np.array([], dtype=np.float32)
        if 3 in self.args.txt_indices and 4 in self.args.txt_indices:
            # quantity of voxels in which the fibers terminate in each ROI.
            surfaces = np.loadtxt(paths[3])
            surfaces = surfaces.astype(np.float32)
            # quantity of voxels in each ROI.
            voxels = np.loadtxt(paths[4])
            voxels = voxels.astype(np.float32)

            mask = np.where(voxels == 0)
            voxels[mask] = np.inf
            vector = surfaces / voxels
            vector = vector[:, np.newaxis]

        weights = []
        for i in range(5, 26):
            if i not in self.args.txt_indices:
                continue
            path = paths[i]
            weight = np.loadtxt(path)
            weight = np.round(weight, 6)
            weights.append(weight)

        centroids = []
        for i in range(26, 33):
            if i not in self.args.txt_indices:
                continue
            path = paths[i]
            position = np.loadtxt(path)
            centroids.append(position)

        lengths = []
        for i in range(33, 40):
            if i not in self.args.txt_indices:
                continue
            path = paths[i]
            length = np.loadtxt(path)
            lengths.append(length)

        cos_angles = []
        for i in range(40, 47):
            if i not in self.args.txt_indices:
                continue
            path = paths[i]
            tCos_angles = np.loadtxt(path)
            cos_angles.append(tCos_angles)

        if len(networks) > 0 and not isinstance(networks, np.ndarray):
            networks = np.array(networks, dtype=np.float32)

        if len(weights) > 0:
            weights = np.array(weights, dtype=np.float32).T

        if len(centroids) > 0:
            centroids = np.array(centroids, dtype=np.float32).transpose(1, 0, 2)

        # (7, 90)
        if len(lengths) > 0:
            # (90, 7)
            lengths = np.array(lengths, dtype=np.float32).T

        # (7, 90, 3)
        if len(cos_angles) > 0:
            # (90, 7, 3)
            cos_angles = np.array(cos_angles, dtype=np.float32).transpose(1, 0, 2)

        return networks, vector, weights, centroids, lengths, cos_angles


class SyntheticDatasetW(SyntheticDataset):
    def __getitem__(self, index: int) -> np.ndarray:
        networks, vector, weights, centroids, lengths, cos_angles = super().__getitem__(index)

        metric_len = len(self.args.txt_indices)

        if metric_len <= 3:
            datum = networks.ravel()
        elif metric_len <= 5:
            datum = np.hstack((networks, vector)).ravel()
        else:
            datum = np.hstack((networks, vector, weights)).ravel()
        return datum


class SyntheticDatasetB(SyntheticDatasetW):
    """
    ref. tud.TensorDataset
    """
    def __getitem__(self, index: int) -> [np.ndarray, int]:
        datum = super().__getitem__(index)

        sample = self.params.samples[index]
        assert isinstance(sample, ds.SampleBase)
        return datum, sample.label
