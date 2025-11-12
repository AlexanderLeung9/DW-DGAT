import os
import typing as t
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import sklearn.metrics.pairwise as smp
import arguments as ag


def build_kNN_adjacent_graph(matrix: np.ndarray, k: int, nearest_farthest: bool) -> ([[float]], float):
    """
    Compute distances between every two ROIs, and then select the nearest or farthest k ROIs.
     The number of all distances is [ROI_n * (ROI_n-1)] / 2.
    Return the adjacency matrix of a kNN graph.
    :param matrix: shape=(ROI_n, samples)
    :param k: how many neighbours should be kept
    :param nearest_farthest: False: nearest, True: farthest
    :return: shape=(ROIs, ROIs)
    """
    # distances.shape=(ROIs, ROIs)
    distances = smp.pairwise_distances(matrix)
    # masks = np.isnan(distances)
    # test_sum = np.sum(masks)
    # assert test_sum == 0
    # distances[masks] = np.inf
    similarities = to_gaussian_similarities(distances, "mean")

    # Select k neighbors for every ROI. indices.shape=(ROIs, k).
    if not nearest_farthest:
        indices = np.argsort(similarities, axis=1)[:, -1:-k-2:-1]
    else:
        indices = np.argsort(similarities, axis=1)[:, 0:k:1]

    N = indices.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in indices[i]:
            if adj[i][j] == 0 and i in indices[j]:
                adj[i][j] = adj[j][i] = distances[i][j]

    count = np.sum(adj != 0)
    percent = count / (N * N)

    return adj, percent


def normalize_adjacent_graph(adj: np.ndarray, kind: str, rescale: t.Optional[bool] = True, to_csr: bool = False) -> ([[float]], float):
    np.fill_diagonal(adj, 0)

    if kind == "gnn" or kind == "symmetric" or kind == "spectral":
        # Degree matrix
        degree = adj.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Symmetric normalized Laplacian
        norm_adj = np.eye(adj.shape[0], dtype=adj.dtype) - D_inv_sqrt @ adj @ D_inv_sqrt
    elif kind == "gcn" or kind == "renormalized" or kind == "spatial":
        # Add self-loops
        adj_with_self_loops = adj + np.eye(adj.shape[0], dtype=adj.dtype)

        # Degree matrix
        degree = np.sum(adj_with_self_loops, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

        # Re-normalized adjacency matrix
        norm_adj = D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt
    else:
        raise NotImplementedError(f"kind={kind}")

    if (kind == "gnn" or kind == "symmetric" or kind == "spectral") and rescale:
        lambda_max = 2.0  # For symmetric normalized Laplacian, lambda_max is 2
        norm_adj = (2 / lambda_max) * norm_adj - np.eye(norm_adj.shape[0], dtype=norm_adj.dtype)

    tSum = np.sum(adj != 0)
    total = adj.shape[0] ** 2
    percent = tSum / total

    if to_csr:
        norm_adj = sp.csr_matrix(norm_adj, dtype=norm_adj.dtype)

    return norm_adj, percent


def build_cosine_sim_graph(vectors: [[float]], normalization_kind: str = "gcn") -> [[float]]:
    similarities = smp.cosine_similarity(vectors).astype(np.float32)
    # It will impact the median in the Gaussian kernel function.
    np.fill_diagonal(similarities, 0)

    distances = 1 - similarities
    sim_graph = to_gaussian_similarities(distances, "median")

    if normalization_kind != "":
        sim_graph, _ = normalize_adjacent_graph(sim_graph, normalization_kind)
        sim_graph = torch.from_numpy(sim_graph).to(ag.Arguments.device)
    return sim_graph


def build_feature_graph(sample_features: [[float]], normalize_kind: str) -> [[float]]:
    distances = skm.pairwise_distances(sample_features).astype(np.float32)
    graph = to_gaussian_similarities(distances, "mean")

    if normalize_kind != "":
        graph, _ = normalize_adjacent_graph(graph, normalize_kind)
        graph = torch.from_numpy(graph).to(ag.Arguments.device)
    return graph


def draw_heatmap(matrix: [[float]], caption: str, args: ag.Arguments):
    strClasses = "-".join(str(tClass) for tClass in args.classes)
    caption = f"Heatmap of {caption} ({strClasses})"
    file_name = caption + ".png"
    file_path = os.path.join(args.log_dir_path, file_name)
    if os.path.exists(file_path):
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="bwr", aspect="auto")
    # plt.imshow(matrix, cmap='bwr', interpolation='nearest', norm=mc.LogNorm())
    plt.colorbar()
    plt.title(caption)
    plt.savefig(file_path)
    plt.close()


def adjust_adjacent_graph(adjacent_graph: [[float]], percentile: int, first_or_last: bool):
    """
    :params percentile: arranges in ascending order
    """
    percentile_value = np.percentile(adjacent_graph, percentile)
    if not first_or_last:
        adjacent_graph[adjacent_graph > percentile_value] = 0
    else:
        adjacent_graph[adjacent_graph < percentile_value] = 0


def analyze_adjacent_graph(adjacent_graph: [[float]], relationship_graph: [[bool]] = None):
    adj_avg = adjacent_graph.mean()
    indices = np.abs(adjacent_graph) >= 1e-6
    non_zero_elements = adjacent_graph[indices]
    non_zero_count = non_zero_elements.shape[0]
    if non_zero_count == 0:
        nonzero_min = 0
    else:
        nonzero_min = non_zero_elements.min()

    adj_max = adjacent_graph.max()

    total = np.multiply(*adjacent_graph.shape)
    nonzero_percent = non_zero_count / total

    if relationship_graph is not None:
        correct_count = np.sum(indices == relationship_graph)
        correct_percent = correct_count / total
    else:
        correct_percent = 0

    print(f"correct_percent={correct_percent:.5f}, adj_avg={adj_avg:.5f}, nonzero_min={nonzero_min:.5f}, adj_max={adj_max:.5f}, nonzero_percent={nonzero_percent:.5f}")


def to_sparse_tensor(array) -> torch.sparse_coo_tensor:
    if type(array) is np.ndarray:
        array = torch.from_numpy(array)

    indices = torch.nonzero(array, as_tuple=False).t()
    values = array[array != 0]
    sparse_tensor = torch.sparse_coo_tensor(indices, values, array.size())
    return sparse_tensor


def to_gaussian_similarities(distances: [[float]], sigma_method: str) -> np.ndarray:
    if sigma_method == "mean":
        sigma = np.mean(distances)
    elif sigma_method == "median":
        sigma = np.median(distances)
    else:
        raise ValueError("Method should be either 'mean' or 'median'")

    similarity_matrix = np.exp(-(distances**2) / (2 * sigma**2))
    return similarity_matrix


def get_edge_indices_and_weights(adjacent_graph: [[float]]) -> ([[int]], [float]):
    edge_indices = []
    edge_weights = []
    N = adjacent_graph.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            edge_indices.append((i, j))
            edge_weights.append(adjacent_graph[i, j])

    edge_indices = torch.LongTensor(np.array(edge_indices).T).to(ag.Arguments.device)
    edge_weights = torch.FloatTensor(edge_weights).to(ag.Arguments.device)
    return edge_indices, edge_weights
