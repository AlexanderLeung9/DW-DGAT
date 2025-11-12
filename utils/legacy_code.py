import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import scipy.sparse.csgraph as ssc
import scipy.sparse.linalg as ssl
import sklearn.metrics as skm
import sklearn.cluster as skc
import sklearn.ensemble as es
import sklearn.feature_selection as fs
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
import arguments as ag
import utils.CommonUtils as cu
import utils.GraphUtils as gu
import utils.BDGraphUtils as pu
import datasets as ds


def __load_vector(root_folder: str, sample_id: str, path_part: str, filename_format: str) -> [float]:
    file_path = os.path.join(root_folder, sample_id, path_part, filename_format)
    file_path = file_path.format(sample_id)
    vector = np.loadtxt(file_path)
    return vector


def build_shared_roi_graph1(
    filename_formats2: [str],
    args: ag.Arguments,
    samples: [ds.PDSample],
    nearest_farthest: bool,
    normalize_kind: str,
) -> (sp.csr_matrix, int):
    class_sample_ROIs = [[] for _ in range(args.class_num)]
    # (sample_num, ROI_NUM)
    sample_ROI_ratios = []

    for index, sample in enumerate(samples):
        event_path = ag.BDArguments.event_parts[sample.event_index]
        folder_path = os.path.join(args.dataset_dir, event_path)

        # quantity of voxels in which the fibers terminated in each ROI.
        surfaces = __load_vector(
            folder_path, sample.No, filename_formats2[0]
        )
        # quantity of voxels in each ROI.
        voxels = __load_vector(
            folder_path, sample.No, filename_formats2[1]
        )
        mask = np.where(voxels == 0)
        voxels[mask] = np.inf
        ratios = surfaces / voxels

        class_sample_ROIs[sample.label].append(ratios)
        sample_ROI_ratios.append(ratios)

    # intra_class_differences.shape=(class_num, ROI_n)
    # intra_class_differences[i] is 90 ROIs, each ROI contains variance of all samples in class i.
    intra_class_differences = np.empty((args.class_num, ag.BDArguments.ROI_NUM), dtype=np.int32)
    class_ROI_variances = np.empty((args.class_num, ag.BDArguments.ROI_NUM), dtype=np.int32)
    class_ROI_mean = np.empty((args.class_num, ag.BDArguments.ROI_NUM), dtype=np.float32)
    ROI_variances = np.empty(ag.BDArguments.ROI_NUM, dtype=np.float32)

    # Each class has several samples, each sample has 90 ROIs.
    for i in range(args.class_num):
        for j in range(class_ROI_variances.shape[1]):
            sample_ROIs = class_sample_ROIs[i]
            sample_ROIs = np.array(sample_ROIs)

            class_ROI_variances[i][j] = np.var(sample_ROIs[:, j])
            class_ROI_mean[i][j] = np.mean(sample_ROIs[:, j])

        ROI_variance = class_ROI_variances[i]
        intra_class_differences[i, :] = np.argsort(ROI_variance)

    for j in range(class_ROI_mean.shape[1]):
        # ROI_variances[j] is a variance of all classes.
        ROI_variances[j] = np.var(class_ROI_mean[:, j])

    # inter_class_differences.shape=(ROI_n,)
    inter_class_differences = np.argsort(ROI_variances)[::-1]

    # v_sorts are the variances of intra-class, and inter_differences are the variances of inter-class.
    ROI_kNN = __find_k_4_max_count(intra_class_differences, inter_class_differences)

    # ROI_kNN = 90
    print(f"ROI_kNN={ROI_kNN}")
    assert ROI_kNN > 0

    ROI_samples = np.array(sample_ROI_ratios).T
    adj_graph, percent = gu.build_kNN_adjacent_graph(ROI_samples, ROI_kNN, nearest_farthest)
    print(f"shared_roi_graph nonzero percent: {percent * 100:.2f}%")
    laplacian_graph, _ = gu.normalize_adjacent_graph(adj_graph, normalize_kind, True)

    # shared_graph will keep stable.
    return laplacian_graph, ROI_kNN


def __find_k_4_max_count(ascending_ROIs: [[float]], descending_ROIs: [float]) -> int:
    """
    Try to find the maximal number of ROIs that k ROIs vary the least intra classes and k ROIs vary the most inter classes.
    """
    k = -1
    class_num = len(ascending_ROIs)
    common = np.empty(class_num)
    max_count = 0
    for j in range(ag.BDArguments.ROI_NUM):
        for i in range(class_num):
            common[i] = len(list(set(descending_ROIs[0:j]).difference(set(ascending_ROIs[i][0:j]))))

        total = np.sum(common)
        if total >= max_count:
            max_count = total
            k = j
    return k + 1


def rearrange_labels(samples: [ds.SampleBase], labels: [int], classes: [int]):
    if classes[0] == 0 and classes[1] == 1:
        return

    difference = classes[0]
    for i in range(samples.shape[0]):
        if samples[i].label == classes[0]:
            samples[i].label -= difference
            labels[i] -= difference
        else:
            samples[i].label = 1
            labels[i] = 1


def analyze_deterministic_networks(all_samples: [ds.PDSample], filename_formats: [str], args: ag.Arguments):
    total_statistics = np.zeros((len(all_samples), 3), dtype=np.float32)
    for i, sample in enumerate(all_samples):
        event_path = ag.BDArguments.event_parts[sample.event_index]
        file_path = os.path.join(args.dataset_dir, event_path, sample.No)

        filename_format = filename_formats[0]
        path = os.path.join(file_path, filename_format)
        path = path.format(sample.No)
        network_matrix = np.loadtxt(path)
        network_matrix[network_matrix != 0] = 1
        neighbor_nums = np.sum(network_matrix, axis=1)

        min_num = np.min(neighbor_nums)
        max_num = np.max(neighbor_nums)
        avg_num = neighbor_nums.mean()
        single_statistics = [min_num, max_num, avg_num]
        total_statistics[i] = single_statistics

    min_min = total_statistics[:, 0].min()
    max_min = total_statistics[:, 0].max()
    min_max = total_statistics[:, 1].min()
    max_max = total_statistics[:, 1].max()
    avg_avg = total_statistics[:, 2].mean()
    # min_min=0.0, max_min=4.0, min_max=19.0, max_max=39.0, avg_avg=10.612488746643066
    print(f"min_min={min_min}, max_min={max_min}, min_max={min_max}, max_max={max_max}, avg_avg={avg_avg}")
    pass


def eliminate_features(data: np.ndarray, labels: [int]) -> np.ndarray:
    shape = data.shape
    data = data.reshape(shape[0], -1)

    clf = es.RandomForestClassifier(class_weight="balanced")
    skf = sms.StratifiedKFold(10, shuffle=True)
    eliminator = fs.RFECV(clf, cv=skf, scoring="f1")
    eliminator.fit(data, labels)
    selected_features = eliminator.transform(data)

    shape_list = list(shape)
    shape_list[-1] = selected_features.shape[-1]
    data = selected_features.transpose(tuple(shape_list))
    return data


def whiten_data(X: [[float]]) -> [[float]]:
    X -= np.mean(X)  # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0]  # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    X_rot = np.dot(X, U)  # de-correlate the data
    X_white = X_rot / np.sqrt(S + 1e-5)
    return X_white


def plot_loss_history(loss_history: [float], log_dir_path: str, log_file_caption: str):
    avg_loss = np.average(loss_history)
    # sample standard deviation
    std_loss = np.std(loss_history, ddof=1)

    length = len(loss_history)
    history_split_epoch = 1
    for i in range(length):
        loss_value = loss_history[i]
        if loss_value <= avg_loss + std_loss:
            history_split_epoch = i + 1 + 1
            break
    if history_split_epoch >= length-1:
        history_split_epoch = 1

    plt.axis('off')
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(20)
    COL_NUM = 2
    gs = fig.add_gridspec(1, COL_NUM)
    # hspace: vertical, wspace: horizontal
    gs.update(wspace=0.1)

    for col in range(COL_NUM):
        total_length = len(loss_history)
        if col == 0:
            loss_history2 = loss_history[:history_split_epoch-1]
        else:
            loss_history2 = loss_history[history_split_epoch-1:]
        pic_loss = fig.add_subplot(gs[0, col])

        pic_loss.set_title(f"Training Loss (part {col+1})")
        from_epoch = 1 if col == 0 else history_split_epoch
        pic_loss.set_xlabel(f"from the {cu.get_sequence_no(from_epoch)} epoch, total {total_length} epochs")
        pic_loss.grid()

        length = len(loss_history2)
        x_ticker_num = 25.0 if length <= 100 else 20.0
        y_ticker_num = 9.0
        unit_x = length / x_ticker_num
        if unit_x < 1.0:
            unit_x = 1
        else:
            unit_x = np.ceil(unit_x)
        min_y = min(loss_history2)
        max_y = max(loss_history2)
        unit_y = (max_y - min_y) / y_ticker_num
        if unit_x != 0:
            pic_loss.set_xticks(np.arange(0, length, unit_x))
        if unit_y != 0:
            pic_loss.set_yticks(np.arange(min_y, max_y, unit_y))
        pic_loss.plot(loss_history2)
        pic_loss.tick_params(axis="x", rotation=45)

    plt.subplots_adjust(left=0.040, bottom=0.125, right=0.985, top=0.94)

    picture_path = os.path.join(log_dir_path, f"{log_file_caption}-loss_history.png")
    plt.savefig(picture_path)
    plt.close()


# region train adjacent graph
def gaussian_kernel(d, sigma):
    d2 = -(d**2)
    b = 2 * sigma**2
    c = d2 / b
    e = torch.exp(c)
    return e


def nuclear_norm(W):
    W1 = torch.norm(W, p="nuc")
    # W2 = np.sum(np.linalg.svd(W, compute_uv=False))
    # assert W1 == W2
    return W1


def ky_fan_constraint(W, num_classes):
    L = torch.diag(W.sum(1)) - W  # 拉普拉斯矩阵
    eig_vals, eig_vecs = torch.linalg.eigh(L)  # 使用更稳定的linalg.eigh
    Q = eig_vecs[:, :num_classes]
    return Q


def train_adjacent_graph(X: [[float]], samples: [ds.PDSample], args: ag.Arguments) -> [[float]]:
    # hyperparameters
    lambda0 = 1
    lambda1 = 0.5
    lambda2 = 0.5
    learning_rate = 0.01
    num_epochs = 1000

    N, D = X.shape
    X = cu.min_max_scale(X, False)

    pairwise_dists = skm.pairwise_distances(X)
    d = torch.from_numpy(pairwise_dists).to(ag.Arguments.device)

    # indices, sim1 = __compute_kNN_graph2(X, k)
    # E = build_adjacent_graph(indices)
    # sim2 = E * sim1
    E = spectral_clustering(X, args.class_num)
    relationship_graph = pu.build_relationship_graph(samples)
    gu.analyze_adjacent_graph(E, relationship_graph)
    E = torch.from_numpy(E).to(ag.Arguments.device)
    # relationship_graph = torch.from_numpy(relationship_graph).to(ag.Arguments.device)

    # W_init = nn.Parameter(torch.from_numpy(sim2).to(ag.Arguments.device))
    X = torch.from_numpy(X).to(ag.Arguments.device)

    # relationship_graph = torch.from_numpy(relationship_graph).to(ag.Arguments.device)
    # r = torch.ones_like(relationship_graph).to(ag.Arguments.device)
    # r[relationship_graph] = 0

    sigma = nn.Parameter(torch.randn(N, N, dtype=torch.float).to(ag.Arguments.device))
    optimizer = optim.Adam([sigma], lr=learning_rate)
    # optimizer = optim.Adam([W_init], lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs], 0.5)
    loss_history = []

    W = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        W_init = gaussian_kernel(d, sigma)
        # Make sure W is a semi-symmetric matrix.
        W = torch.matmul(W_init.T, W_init)
        # indices = W < 0
        # W[indices] = 0
        W = E * W

        W.fill_diagonal_(0)

        # Compute a bias item.
        # WX = torch.matmul(W, X)
        # diff = X - WX
        # error = lambda0 * torch.norm(diff, p=2, dim=1).sum()

        error = 0
        for i in range(N):
            w_i = W[i].unsqueeze(1)
            wx_j = w_i * X
            s_wx_j = torch.sum(wx_j, dim=0) - wx_j[i]
            diff = X[i] - s_wx_j
            s_diff = torch.norm(diff, p=2).sum()
            error += s_diff
        error /= N**2
        error *= lambda0

        # Compute a sparse item.
        sparse = lambda1 * torch.norm(W, p=1)

        low_rank = lambda2 * nuclear_norm(W)

        # Q = ky_fan_constraint(W, args.class_num)
        # ky_fan_loss = lambda2 * torch.trace(torch.matmul(torch.matmul(Q.T, W), Q))

        # loss4 = torch.norm(W * r, p="fro")

        loss = error + sparse + low_rank
        # loss = loss4
        if epoch >= 100:
            loss_history.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            # analyze_adjacent_graph2(W, relationship_graph)

    plot_loss_history(loss_history, args.log_dir_path, "similarity_matrix")

    W = W.detach()
    W = W.cpu().numpy()
    W, _ = gu.normalize_adjacent_graph(W, "gcn")

    W = torch.from_numpy(W).to(ag.Arguments.device)
    return W


def spectral_clustering(X, n_clusters):
    """
    Perform spectral clustering on the given data.

    Parameters:
    X : array-like, shape (N, D)
        Input data matrix with N samples and D features.
    n_clusters : int
        The number of clusters to form.
    sigma : float, optional (default=1.0)
        Gaussian kernel parameter for affinity matrix computation.

    Returns:
    labels : array, shape (N, )
        The cluster labels for each sample.
    """

    # Step 1: Compute the affinity matrix (similarity matrix)
    pairwise_dists = skm.pairwise_distances(X)
    affinity_matrix = gu.to_gaussian_similarities(pairwise_dists, "mean")

    # Step 2: Compute the Laplacian matrix
    L, D = ssc.laplacian(affinity_matrix, normed=True, return_diag=True)

    # Step 3: Compute the first k eigenvectors of the Laplacian
    _, eig_vecs = ssl.eigsh(L, k=n_clusters, which="SM")

    # Step 4: Normalize the eigenvectors row-wise
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=1, keepdims=True)

    # Step 5: Perform k-means clustering on the eigenvectors
    kmeans = skc.KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(eig_vecs)

    N = X.shape[0]
    matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                matrix[i][j] = matrix[j][i] = 1

    count = np.sum(matrix != matrix.T)
    assert count == 0

    return matrix
# endregion
