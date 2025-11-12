import tqdm
import numpy as np
import torch.utils.data as tud
import scipy.sparse as sp
import logging as l
import enums as e
import arguments as ag
import utils.GraphUtils as gu
import utils.BDGraphUtils as pu
import datasets as ds
import networks as nw
import solvers as s


class MV_GCNSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.MV_GCNArgs):
        super().__init__(all_samples, logger, args)
        self.args = args
        self.base_sample_index: int = len(self.all_samples)

    def initialize(self):
        self.__initialize()
        shared_graphs, perm, pool_size = self.__prepare_data()
        super()._split_train_and_test_parts()
        self.__prepare_network(shared_graphs, perm, pool_size)

    # def __initialize(self):
    #     all_labels = []
    #     sample_num = len(self.all_samples)
    #     for i, sample in enumerate(self.all_samples):
    #         for j in range(i + 1, sample_num):
    #             if self.all_samples[i].label == self.all_samples[j].label:
    #                 all_labels.append(1)
    #             else:
    #                 all_labels.append(0)
    #     self.all_labels = np.array(all_labels)

    def __initialize(self):
        assert len(self.args.classes) == 2, "Only support two-class classification."

        base_sample = None
        for i, sample in enumerate(self.all_samples):
            if sample.label == 1:
                self.base_sample_index = i
                base_sample = sample
                break

        all_labels = []
        for sample in self.all_samples:
            label = 1 if base_sample.label == sample.label else 0
            all_labels.append(label)
        self.all_labels = np.array(all_labels)

    def __prepare_data(self):
        # region construct shared_graphs
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDataset(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

        all_data = []
        coordinates = np.zeros((len(self.all_samples), ag.BDArguments.ROI_NUM, 3))
        print(f"Loading all data {self.args.txt_indices}...")
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            assert isinstance(data, list), "Dismiss a warning."
            networks = data[0][0].cpu().numpy()
            all_data.append(networks)
            coordinate = data[3][0]
            coordinates[i] = coordinate[0]

        if self.args.adj_graph_type == e.EAdjGraphType.Phenotype:
            adjacent_graph = pu.build_phenotype_graph(self.all_samples, "")
        elif self.args.adj_graph_type == e.EAdjGraphType.Euclidean:
            coo1, coo2, coo3 = coordinates.shape
            features = np.zeros([coo1 * coo3, coo2])
            for i in range(coo3):
                features[coo1 * i: coo1 * (i + 1), :] = coordinates[:, :, i]

            adjacent_graph, percent = gu.build_kNN_adjacent_graph(features.T, self.args.kNN, False)
            print(f"shared_roi_graph nonzero percent: {percent * 100:.2f}%")
        elif self.args.adj_graph_type == e.EAdjGraphType.Unweighted:
            adjacent_graph = pu.build_RA_GCN_graph1(self.all_samples, "")
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        A = sp.csr_matrix(adjacent_graph)
        graphs, perm = coarsen(A, levels=4, self_connections=False)
        laplacian_graphs = [gu.normalize_adjacent_graph(A.toarray(), "gnn", True) for A in graphs]
        for i, (laplacian_graph, percent) in enumerate(laplacian_graphs):
            print("laplacian_graph" + str(i+1) + ":")
            gu.analyze_adjacent_graph(laplacian_graph)
        shared_graphs = [gu.to_sparse_tensor(laplacian_graphs[0][0]).to(ag.Arguments.device)]
        pool_size = 4
        j = int(np.log2(pool_size))
        shared_graphs.append(gu.to_sparse_tensor(laplacian_graphs[j][0]).to(ag.Arguments.device))
        # endregion

        # region data_loader
        all_data = np.array(all_data)
        all_data = perm_data1(all_data, perm, True)

        dataset_params = ds.PairDatasetParams(all_data, self.all_samples, self.base_sample_index)
        self.dataset = ds.PairDataset(dataset_params)
        # endregion
        return shared_graphs, perm, pool_size

    def __prepare_network(self, shared_graphs, perm, pool_size):
        decay_steps = self.sample_num // int(self.args.vldtn_ratio_OR_k_fold) * 9 // self.args.batch_size
        # assert decay_steps > 1
        self.args.gamma = 0.95 ** (1/decay_steps)
        assert isinstance(self.dataset, ds.PairDataset), "Dismiss a warning."
        M = self.dataset.params.network_matrices.shape[2]

        # F_outs corresponds to `m` and 64, and linear_dim corresponds to `M`.
        create_cls_params = nw.MV_GCNParams(self.args, shared_graphs, perm, ag.BDArguments.ROI_NUM, M, [128, 64], pool_size, linear_dim=512)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)


# The following code is mainly from the original MV-GCN paper.
def coarsen(A, levels, self_connections=False):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple levels.
    """
    graphs, parents = metis(A, levels)
    perms = compute_perm(parents)

    for i, A in enumerate(graphs):
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

        M_new, M_new = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, M_new, M_new-M, A.nnz//2))

    return graphs, perms[0] if levels > 0 else None


def metis(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = list()
    graphs.append(W)
    # supernode_size = np.ones(N)
    # nd_sz = [supernode_size]
    # count = 0

    # while N > maxsize:
    for _ in range(levels):
        # count += 1

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = sp.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr, cc, vv, rid, weights, W_N=weights.shape[0])  # rr is ordered, W_N is the height/width of the matrix
        parents.append(cluster_id)

        # TO DO
        # COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE
        # supernode_size = full(   sparse(cluster_id,  ones(N,1) , supernode_size )     )
        # print(cluster_id)
        # print(supernode_size)
        # nd_sz{count+1}=supernode_size;

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        N_new = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.csr_matrix((nvv, (nrr, ncc)), shape=(N_new, N_new))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        # N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        # degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        # [~, rid]=sort(ss);     # arthur strategy
        # [~, rid]=sort(supernode_size);    #  thomas strategy
        # rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr, cc, vv, rid, weights, W_N=0):
    nnz = rr.shape[0]
    # N = rr[nnz-1] + 1
    N = W_N     # W_N is the height/width of the matrix

    marked = np.zeros(N, np.bool_)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            best_neighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    best_neighbor = nid

            cluster_id[tid] = clustercount

            if best_neighbor > -1:
                cluster_id[best_neighbor] = clustercount
                marked[best_neighbor] = True

            clustercount += 1

    return cluster_id


def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    M_last = 0
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        # print('parent: {}'.format(parent))

        # Fake nodes go after real ones.
        pool_singletons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            # print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singleton.
            if len(indices_node) == 1:
                indices_node.append(pool_singletons)
                pool_singletons += 1
                # print('new singleton: {}'.format(indices_node))
            # Add two nodes as children of a singleton in the parent.
            elif len(indices_node) == 0:
                indices_node.append(pool_singletons+0)
                indices_node.append(pool_singletons+1)
                pool_singletons += 2
                # print('singleton children: {}'.format(indices_node))

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks.
    for i, indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an index.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


def perm_data1(x, indices, multiview=False):
    """
    Permute data tensor, i.e. exchange node ids--the 2nd dimension,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    if multiview:
        N, V, M, F = x.shape
        M_new = len(indices)
        assert M_new >= M
        x_new = np.zeros((N, V, M_new, F), dtype=np.float32)
        for i,j in enumerate(indices):
            # Existing vertex, i.e. real data.
            if j < M:
                x_new[:,:,i,:] = x[:,:,j,:]
    else:
        N, M, F = x.shape
        M_new = len(indices)
        assert M_new >= M
        x_new = np.zeros((N, M_new, F), dtype=np.float32)
        for i,j in enumerate(indices):
            # Existing vertex, i.e. real data.
            if j < M:
                x_new[:,i,:] = x[:,j,:]

    return x_new


def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    M_new = len(indices)
    assert M_new >= M
    A = A.tocoo()

    # Add M_new - M isolated vertices.
    if M_new > M:
        rows = sp.coo_matrix((M_new-M,    M), dtype=np.float32)
        cols = sp.coo_matrix((M_new, M_new-M), dtype=np.float32)
        A = sp.vstack([A, rows])
        A = sp.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is sp.coo.coo_matrix
    return A
