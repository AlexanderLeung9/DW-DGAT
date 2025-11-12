import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn.conv as tnc
import torch_geometric.typing as tgt
import torch_geometric.utils as tgu
import networks as nw
import arguments as ag
import datasets as ds
import enums as es
import utils.GraphUtils as gu
import utils.BDGraphUtils as bu


class ChebNetII(nw.NetworkBase):
    """
    @article{he2022convolutional,
      title={Convolutional neural networks on graphs with chebyshev approximation, revisited},
      author={He, Mingguo and Wei, Zhewei and Wen, Ji-Rong},
      journal={Advances in Neural Information Processing Systems},
      volume={35},
      pages={7264--7276},
      year={2022}
    }
    """
    def __init__(self, params: nw.GCNParams):
        super().__init__(params)

        assert isinstance(params.args, ag.ChebNetIIArgs)
        self.args = params.args
        self.all_samples = params.all_samples
        self.dropout_rate = params.args.dropout_rate
        self.edge_indices = None
        self.edge_weights = None

        self.linear1 = nn.Linear(params.args.feature_num, params.args.hidden_dim)
        self.linear2 = nn.Linear(params.args.hidden_dim, params.args.class_num)
        self.prop = ChebNetII_prop(params.args.k_order)

        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, nodes: [[float]]) -> [[float]]:
        if not self.args.learning_mode:
            samples = self.all_samples[self.current_batch_indices]
            edge_indices, edge_weights = self.__build_adjacency_graph(samples, nodes)
        else:
            if self.edge_indices is None or self.edge_weights is None:
                self.edge_indices, self.edge_weights = self.__build_adjacency_graph(self.all_samples, nodes)
            edge_indices, edge_weights = self.edge_indices, self.edge_weights

        x = f.dropout(nodes, self.dropout_rate, training=self.training)
        x = self.linear1.forward(x)
        x = f.relu(x)

        x = f.dropout(x, self.dropout_rate, training=self.training)
        x = self.linear2.forward(x)

        x = f.dropout(x, self.dropout_rate, training=self.training)
        y = self.prop.forward(x, edge_indices, edge_weights)
        
        if not self.args.learning_mode:
            scores = y
        else:
            scores = y[self.current_indices]
        return scores

    def __build_adjacency_graph(self, samples: [ds.SampleBase], nodes: [[float]]) -> (torch.Tensor, torch.Tensor):
        if self.args.adj_graph_type == es.EAdjGraphType.Phenotype:
            adjacent_graph = bu.build_phenotype_graph(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Euclidean:
            adjacent_graph = gu.build_feature_graph(nodes.detach().cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Unweighted:
            adjacent_graph = bu.build_RA_GCN_graph1(samples, "")
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        edge_indices, edge_weights = gu.get_edge_indices_and_weights(adjacent_graph)
        N = adjacent_graph.shape[0]
        edge_indices, edge_weights = tgu.get_laplacian(edge_indices, edge_weights, normalization="sym", dtype=adjacent_graph[0][0].dtype, num_nodes=N)
        edge_indices, edge_weights = tgu.add_self_loops(edge_indices, edge_weights, fill_value=-1.0, num_nodes=N)

        return edge_indices, edge_weights


class ChebNetII_prop(tnc.MessagePassing):
    def __init__(self, K: int, init=False, **kwargs):
        super().__init__(aggr="add", **kwargs)

        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K + 1))
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.init:
            for j in range(self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                self.temp.data[j] = x_j**2

    def forward(self, x: [[float]], edge_indices: [[int]], edge_weights: [float]):
        coe_tmp = f.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * ChebNetII_prop.cheby(
                i, math.cos((self.K + 0.5) * math.pi / (self.K + 1))
            )
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * ChebNetII_prop.cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        Tx_0 = x
        Tx_1 = self.propagate(edge_indices, x=x, norm=edge_weights, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_indices, x=Tx_1, norm=edge_weights, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def edge_update(self) -> torch.Tensor:
        raise NotImplementedError()

    def message_and_aggregate(self, adj_t: tgt.Adj) -> torch.Tensor:
        raise NotImplementedError()

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)

    @staticmethod
    def cheby(i, x):
        if i == 0:
            return 1
        elif i == 1:
            return x
        else:
            T0 = 1
            T1 = x
            T2 = None
            for ii in range(2, i + 1):
                T2 = 2 * x * T1 - T0
                T0, T1 = T1, T2
            return T2
