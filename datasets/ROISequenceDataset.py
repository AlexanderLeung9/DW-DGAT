import os
import numpy as np
import scipy.spatial.distance as ssd
import torch.utils.data as tud
import utils.CommonUtils as cu
import utils.GraphUtils as gu
import datasets as ds
import arguments as ag


class ROISequenceDatasetParams(ds.DatasetParams):
    def __init__(self, samples: [ds.PDSample], args: ag.DW_DGATArgs):
        super().__init__(args.dataset_dir, samples)
        self.args = args


class ROISequenceDataset(tud.Dataset):
    def __init__(self, params: ROISequenceDatasetParams):
        super().__init__()
        self.params = params

    def __getitem__(self, index: int) -> [[[float]]]:
        sample = self.params.samples[index]
        assert isinstance(sample, ds.PDSample) or isinstance(sample, ds.ADSample)

        event_path = ag.BDArguments.event_parts[sample.event_index]
        file_path = os.path.join(self.params.root_folder, event_path, sample.No)
        
        paths = []
        file_names = [file_name for i, file_name in enumerate(ag.BDArguments.TXT_FILE_NAMES) if i in self.params.args.txt_indices]
        for txt_file_name in file_names:
            path = os.path.join(file_path, txt_file_name)
            path = path.format(sample.No)
            paths.append(path)

        if len(self.params.args.txt_indices) > 3:
            networks = []
            for i, path in enumerate(paths[:3]):
                network = np.loadtxt(path)
                network = cu.min_max_scale(network, False)
                networks.append(network)
            networks = np.array(networks, dtype=np.float32)
            networks = np.linalg.norm(networks, ord=1, axis=2)
            ROIs3 = networks.transpose((1, 0))

            surfaces = np.loadtxt(paths[3])
            voxels = np.loadtxt(paths[4])
            mask = np.where(voxels == 0)
            voxels[mask] = np.inf
            ratios = surfaces / voxels
            ratios = np.expand_dims(ratios, 1)

            if len(paths) > 5:
                other_metrics = []
                for i, path in enumerate(paths[5:]):
                    other_metric = np.loadtxt(path)
                    other_metric = np.round(other_metric, 6)
                    other_metrics.append(other_metric)
                other_metrics = np.array(other_metrics, dtype=np.float32)

                ROIs = np.concatenate((ROIs3, ratios, other_metrics.T), axis=1)
            else:
                ROIs = np.concatenate((ROIs3, ratios), axis=1)
        else:
            networks = []
            for path in paths:
                network = np.loadtxt(path)
                networks.append(network)

            ROIs = np.hstack(networks)

        distances = ssd.squareform(ssd.pdist(ROIs, "euclidean")).astype(np.float32)
        if self.params.args.related_num == 0:
            # shape = (ROI_n, ROI_n)
            # shortest_distances = ROISequenceDataset.__floyd(distances)

            # shape = (ROI_n,)
            ROI_central_distances = ROISequenceDataset.__total_central_distances(distances)

            # shape = (center_num,)
            ROI_sequence = np.argsort(ROI_central_distances)[:self.params.args.center_num]
        else:
            np.fill_diagonal(distances, 0)
            ROI_seqs = []
            for ROI_No in self.params.args.diseased_ROIs:
                tROI_indices = np.argsort(distances[ROI_No-1])[:self.params.args.related_num+1]
                tROI_seq = tROI_indices.tolist()
                ROI_seqs.append(tROI_seq)

            lsROI_sequence = list({item for sublist in ROI_seqs for item in sublist})
            ROI_sequence = np.array(lsROI_sequence)
            diseased_ROIs = np.array(self.params.args.diseased_ROIs)
            diseased_ROIs -= 1
            # diff = set(self.params.args.diseased_ROIs) - set(lsROI_sequence)
            assert np.isin(diseased_ROIs, ROI_sequence).all()

            ROI_central_distances = ROISequenceDataset.__total_central_distances(distances)

        ROI_central_similarities = gu.to_gaussian_similarities(ROI_central_distances, "mean")
        ROIs = np.hstack((ROIs, ROI_central_similarities[:, np.newaxis]))

        datum = np.zeros_like(ROIs)
        datum[ROI_sequence] = ROIs[ROI_sequence]
        return datum

    def __len__(self):
        size = len(self.params.samples)
        return size

    @staticmethod
    def __floyd(distances: [[float]]) -> [[float]]:
        n = distances.shape[0]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

        return distances

    @staticmethod
    def __total_central_distances(distance_matrix: [[float]]) -> [float]:
        central_distances = np.sum(distance_matrix, axis=1)
        return central_distances

    @staticmethod
    def get_pooled_ROI_sequence(ROIs: np.ndarray, center_num: int) -> np.ndarray:
        distances = ssd.squareform(ssd.pdist(ROIs, "euclidean")).astype(np.float32)

        # shape = (ROI_n, ROI_n)
        # shortest_distances = ROISequenceDataset.__floyd(distances)

        # shape = (ROI_n,)
        ROI_central_distances = ROISequenceDataset.__total_central_distances(distances)

        # shape = (center_num,)
        ROI_sequence = np.argsort(ROI_central_distances)[:center_num]

        return ROI_sequence
