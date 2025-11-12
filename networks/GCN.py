import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import scipy.sparse as sp
import networks as nw
import arguments as ag
import datasets as ds
import enums as es
import utils.GraphUtils as gu
import utils.BDGraphUtils as bu


class GCNParams(nw.NetworkParams):
    def __init__(self, args: ag.GCNArgs, all_samples: [ds.SampleBase]):
        super().__init__(args)
        self.args = args
        self.all_samples = all_samples


class GCN(nw.NetworkBase):
    """
    @article{kipf2016semi,
      title={Semi-supervised classification with graph convolutional networks},
      author={Kipf, Thomas N and Welling, Max},
      journal={arXiv preprint arXiv:1609.02907},
      year={2016}
    }
    """
    def __init__(self, params: GCNParams):
        super().__init__(params)
        self.args = params.args
        self.all_samples = params.all_samples
        self.adjacent_graph = None
        self.normalize_row_features = True

        self.gc1 = GraphConvolution(params.args.feature_num, params.args.hidden_dim)
        self.gc2 = GraphConvolution(params.args.hidden_dim, params.args.class_num)

    def forward(self, nodes: [[float]]) -> [[float]]:
        adjacent_graph = self._construct_adjacency_graph(nodes)

        # This line of code is from the Tensorflow version of source code of the GCN-2017 paper.
        x = f.dropout(nodes, self.args.dropout_rate, training=self.training)
        x = f.relu(self.gc1(x, adjacent_graph))
        x = f.dropout(x, self.args.dropout_rate, training=self.training)
        x = self.gc2(x, adjacent_graph)

        if not self.args.learning_mode:
            scores = x
        else:
            scores = x[self.current_indices]
        return scores

    def _construct_adjacency_graph(self, nodes: [[float]]) -> torch.Tensor:
        if not self.args.learning_mode:
            samples = self.all_samples[self.current_batch_indices]
            adjacent_graph = self.__build_adjacency_graph(samples, nodes)
        else:
            if self.adjacent_graph is None:
                self.adjacent_graph = self.__build_adjacency_graph(self.all_samples, nodes)
            adjacent_graph = self.adjacent_graph
        return adjacent_graph

    def __build_adjacency_graph(self, samples: [ds.SampleBase], nodes: [[float]]) -> torch.Tensor:
        if self.args.adj_graph_type == es.EAdjGraphType.Phenotype:
            adjacent_graph = bu.build_phenotype_graph(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Euclidean:
            adjacent_graph = gu.build_feature_graph(nodes.detach().cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Unweighted:
            adjacent_graph = bu.build_RA_GCN_graph1(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.CityBlock:
            # worse
            adjacent_graph = bu.build_RA_GCN_graph2(nodes.detach().cpu().numpy(), "")
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        if self.normalize_row_features:
            adjacent_graph = GCN.__normalize_row_features(adjacent_graph)
        adjacent_graph = torch.from_numpy(adjacent_graph).to(ag.Arguments.device)
        return adjacent_graph

    @staticmethod
    def __normalize_row_features(adj: [[float]]) -> [[float]]:
        """
        The specific operation of GCN.
        """
        identity = np.eye(adj.shape[0], dtype=adj.dtype)
        adj += identity
        row_sum = np.array(adj.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        normalized = r_mat_inv.dot(adj)
        return normalized


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: [[float]], adjacent_graph: [[float]]) -> [[float]]:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adjacent_graph, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
