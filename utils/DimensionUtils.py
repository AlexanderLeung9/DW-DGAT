import numpy as np
import sklearn.decomposition as skd
import sklearn.preprocessing as skp
import typing as t


def down_sample_3D(data: np.ndarray, ratio: int):
    """
    :param data: data to be down sampled
    :param ratio: being greater than 1 indicates to down sample 1/ration in each dimension.
    """
    [r, c, h] = data.shape
    dr = round(r / ratio)
    dc = round(c / ratio)
    dh = round(h / ratio)
    down_data = np.zeros((dr, dc, dh))
    p = 0
    q = 0
    w = 0
    for i in range(0, r, ratio):
        for j in range(0, c, ratio):
            for k in range(0, h, ratio):
                down_data[p, q, w] = data[i, j, k]
                w = w + 1
                if w >= dh:
                    break
            w = 0
            q = q + 1
            if q >= dc:
                break
        q = 0
        p = p + 1
        if p >= dr:
            break
    return down_data


def reduce_dimensionality_by_PCA(x: np.ndarray, dimensionality: int) -> np.ndarray:
    X = x.reshape(x.shape[0], -1)
    # Step 1: standardize data.
    scaler = skp.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: apply PCA.
    pca = skd.PCA(n_components=dimensionality)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca


def reduce_rank(all_data: [[[float]]], k: t.Optional[int] = None) -> [[float]]:
    """
    all_data.shape = (N, feature_num, width, height)
    """
    N = all_data.shape[0]
    all_data2 = all_data.reshape(N, -1)
    feature_num = all_data2.shape[1]

    U, S, VT = np.linalg.svd(all_data2, full_matrices=False)
    if k is None:
        k = feature_num // 10
    all_data3 = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

    return all_data3
