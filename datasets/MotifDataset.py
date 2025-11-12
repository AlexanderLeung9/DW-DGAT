import os
import numpy as np
import collections
import typing as t
import sklearn.metrics as skm
import scipy.spatial.distance as ssd
import torch.utils.data as tud
import utils.CommonUtils as cu
import utils.GraphUtils as gu
import datasets as ds
import enums as es
import arguments as ag


class MotifDatasetParams(ds.DatasetParams):
    def __init__(self, root_folder: str, samples: [ds.PDSample], txt_indices: list[int], center_num: int,
                 motif_nums: list[int], motif_len: int, motif_similarity: es.EMotifSimilarity, AAL_90_Nodes: t.Optional[list] = None):
        super().__init__(root_folder, samples)
        self.txt_indices = txt_indices
        self.center_num = center_num
        self.motif_nums = motif_nums
        self.motif_len = motif_len
        self.motif_similarity = motif_similarity
        self.AAL_90_Nodes = AAL_90_Nodes


class MotifDataset(tud.Dataset):
    def __init__(self, params: MotifDatasetParams):
        super().__init__()
        self.params = params
        self.feature_num = 0
        self.similarity_matrix = None
        assert self.params.center_num % params.motif_len == 0
        self.sim_dir_path = os.path.join(params.root_folder, "similarity_matrices")

    def __getitem__(self, index: int) -> [[[float]]]:
        sample = self.params.samples[index]
        assert isinstance(sample, ds.PDSample) or isinstance(sample, ds.ADSample)

        event_path = ag.BDArguments.event_parts[sample.event_index]
        file_path = os.path.join(self.params.root_folder, event_path, sample.No)

        paths = []
        file_names = [file_name for i, file_name in enumerate(ag.BDArguments.TXT_FILE_NAMES) if i in self.params.txt_indices]
        for txt_file_name in file_names:
            path = os.path.join(file_path, txt_file_name)
            path = path.format(sample.No)
            paths.append(path)

        if len(self.params.txt_indices) > 3:
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

        if self.params.motif_similarity == es.EMotifSimilarity.Euclidean:
            distances = skm.pairwise_distances(ROIs)
        elif self.params.motif_similarity == es.EMotifSimilarity.Gaussian:
            distances = ssd.squareform(ssd.pdist(ROIs, "euclidean")).astype(np.float32)
        elif self.params.motif_similarity == es.EMotifSimilarity.Correlation:
            distances = ssd.squareform(ssd.pdist(ROIs, "correlation")).astype(np.float32)
        else:
            raise NotImplementedError(f"motif_similarity={self.params.motif_similarity}")

        # shape = (ROI_n, ROI_n)
        shortest_distances = MotifDataset.__floyd(distances)
        if self.params.motif_similarity == es.EMotifSimilarity.Gaussian:
            self.similarity_matrix = gu.to_gaussian_similarities(shortest_distances, "mean")
        elif self.params.motif_similarity == es.EMotifSimilarity.Correlation:
            self.similarity_matrix = 1 - distances

        # shape = (ROI_n,)
        ROI_central_distances = MotifDataset.__total_central_distances(shortest_distances)

        # shape = (center_num,)
        ROI_sequence = np.argsort(ROI_central_distances)[:self.params.center_num]
        # shapes = (ROI_n, ROI_n)
        ordered_distances, ordered_indices = MotifDataset.__get_ordered_distances(shortest_distances)

        if self.params.AAL_90_Nodes is not None:
            ROI_Nos = np.array(list(range(1, 91)), dtype=np.float32)[:, np.newaxis]
            ROIs = np.hstack((ROI_Nos, ROIs))
        if self.params.motif_similarity != es.EMotifSimilarity.Euclidean:
            self.feature_num = ROIs.shape[1] + 1
        else:
            self.feature_num = ROIs.shape[1]

        # Every subgraph has 3 blocks (corresponding to 0-2 hop neighbor nodes of the center node), and they have 12,6,3 BFS motifs, respectively, with each motif having 3 ROIs.
        sub_graphs = self.__build_sub_graphs(ROIs, ordered_distances, ordered_indices, ROI_sequence)
        assert sub_graphs.shape == (self.params.center_num, np.sum(self.params.motif_nums) * self.params.motif_len, self.feature_num)

        datum = sub_graphs.transpose(2, 0, 1)
        return datum

    def __len__(self):
        size = len(self.params.samples)
        return size

    @staticmethod
    def __get_ordered_distances(distances: [[float]]) -> ([[float]], [[int]]):
        # (90, 90)
        ordered_indices = np.argsort(distances, axis=1)

        ordered_distances = np.zeros_like(ordered_indices, dtype=distances.dtype)
        for i, row in enumerate(ordered_indices):
            ordered_distances[i] = distances[i][row]
        return ordered_distances, ordered_indices

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

    def __build_sub_graphs(self, roi_features: [[float]], ordered_reduced_distances: [[float]], ordered_indices: [[int]], ROI_sequence: [int])\
            -> [[[float]]]:
        sub_graph_num = ROI_sequence.shape[0]
        motif_num = sum(self.params.motif_nums)
        sub_graphs = np.zeros((sub_graph_num, motif_num, self.params.motif_len, self.feature_num), dtype=np.float32)

        for i in range(sub_graph_num):
            sub_graph = self.__build_sub_graph(roi_features, ordered_reduced_distances, ordered_indices, int(ROI_sequence[i]))
            sub_graphs[i] = sub_graph

        assert sub_graphs.shape == (self.params.center_num, motif_num, self.params.motif_len, self.feature_num)
        sub_graphs = sub_graphs.reshape(sub_graphs.shape[0], -1, sub_graphs.shape[3])
        return sub_graphs

    def __build_sub_graph(self, roi_features: [[float]], ordered_reduced_distances: [[float]], ordered_indices: [[int]], central_node_index: int)\
            -> [[float]]:
        """
        A block corresponds to neighbors of a jump.
        :param roi_features: shape=(ROI_n, feature_n)
        :param ordered_reduced_distances: shape=(ROI_n, ROI_n)
        :return: shape=(motif_num1+motif_num2+motif_num3, params.motif_len, feature_n)
        """
        blocks = []
        block_num = len(self.params.motif_nums)
        start_nodes = [central_node_index]

        for k in range(block_num):
            if k == len(start_nodes):
                break

            start_node = start_nodes[k]
            # n*(n-1)/2 >= self.params.motif_nums[k]
            max_node_num = np.ceil((1+np.sqrt(1+8*self.params.motif_nums[k]))/2)
            core_nodes = MotifDataset.__BFS_confine_node_num(ordered_reduced_distances, ordered_indices, start_node, max_node_num)
            for i in range(k):
                excluded_node = start_nodes[i]
                if excluded_node in core_nodes:
                    core_nodes.remove(excluded_node)

            if len(core_nodes) > 0:
                s = 0
                while len(start_nodes) < block_num:
                    start_nodes.append(core_nodes[s])
                    s += 1

            rows = []
            row_count = 0
            first_ROI = roi_features[start_node].tolist()
            if self.params.motif_similarity != es.EMotifSimilarity.Euclidean:
                first_ROI.append(0)

            # Combinations will not overlap.
            for i, core_node_index in enumerate(core_nodes):
                second_ROI = roi_features[core_node_index].tolist()
                if self.params.motif_similarity != es.EMotifSimilarity.Euclidean:
                    second_ROI.append(0)

                if i + 1 == len(core_nodes):
                    third_ROI = [0] * self.feature_num
                    row = self.__build_a_row([first_ROI, second_ROI, third_ROI], [start_node, core_node_index, -1])
                    rows.append(row)
                    row_count += 1

                for j in range(i + 1, len(core_nodes)):
                    third_ROI = roi_features[core_nodes[j]].tolist()
                    if self.params.motif_similarity != es.EMotifSimilarity.Euclidean:
                        third_ROI.append(0)

                    row = self.__build_a_row([first_ROI, second_ROI, third_ROI], [start_node, core_node_index, core_nodes[j]])
                    rows.append(row)
                    row_count += 1
                    if row_count == self.params.motif_nums[k]:
                        break

                if row_count == self.params.motif_nums[k]:
                    break

            difference = self.params.motif_nums[k] - row_count
            for _ in range(difference):
                second_ROI = [0] * self.feature_num
                third_ROI = [0] * self.feature_num
                row = self.__build_a_row([first_ROI, second_ROI, third_ROI], [start_node, -1, -1])
                rows.append(row)

            block = np.array(rows)
            blocks.append(block)

        sub_graph = np.concatenate(blocks, axis=0)
        assert sub_graph.shape[0] == sum(self.params.motif_nums)
        return sub_graph

    @staticmethod
    def __BFS_confine_node_num(ordered_reduced_distances: [[float]], ordered_indices: [[int]], start_node: int, max_node_num: int) -> [int]:
        assert max_node_num >= 2
        queue = collections.deque()

        neighbors = []
        queue.append(start_node)

        while len(queue) > 0 and len(neighbors) < max_node_num:
            node_index = queue.popleft()
            neighbor_metrics = ordered_reduced_distances[node_index]
            neighbor_indices = ordered_indices[node_index]

            for i, neighbor in enumerate(neighbor_indices):
                if neighbor == node_index:
                    continue

                if neighbor_metrics[i] == np.inf:
                    continue

                if neighbor not in neighbors:
                    neighbors.append(neighbor)
                queue.append(neighbor)

                if len(neighbors) == max_node_num:
                    break

        return neighbors

    def __build_a_row(self, ROIs: [[float]], ROI_indices: [int]) -> [[float]]:
        if self.params.motif_similarity != es.EMotifSimilarity.Euclidean:
            if ROI_indices[0] == -1 or ROI_indices[1] == -1:
                similarity1 = 0
            else:
                similarity1 = self.similarity_matrix[ROI_indices[0]][ROI_indices[1]]
                if similarity1 == np.inf:
                    similarity1 = 0

            if ROI_indices[0] == -1 or ROI_indices[2] == -1:
                similarity2 = 0
            else:
                similarity2 = self.similarity_matrix[ROI_indices[0]][ROI_indices[2]]
                if similarity2 == np.inf:
                    similarity2 = 0

            if ROI_indices[1] == -1 or ROI_indices[2] == -1:
                similarity3 = 0
            else:
                similarity3 = self.similarity_matrix[ROI_indices[1]][ROI_indices[2]]
                if similarity3 == np.inf:
                    similarity3 = 0

            sim1 = (similarity1 + similarity2) / 2
            sim2 = (similarity1 + similarity3) / 2
            sim3 = (similarity2 + similarity3) / 2
            ROIs[0][self.feature_num-1] = sim1
            ROIs[1][self.feature_num-1] = sim2
            ROIs[2][self.feature_num-1] = sim3

        row = np.array(ROIs)
        return row
