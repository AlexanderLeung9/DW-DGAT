import torch
import torch.nn.functional as f
import torch_geometric.data as tgd
import torch_geometric.nn as tgn
import numpy as np
import networks as nw
import arguments as ag
import datasets as ds
import enums as es
import utils.GraphUtils as gu
import utils.BDGraphUtils as bu


class GAT(nw.NetworkBase):
    """
    @article{velivckovic2017graph,
      title={Graph attention networks},
      author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
      journal={arXiv preprint arXiv:1710.10903},
      year={2017}
    }
    """
    def __init__(self, params: nw.GCNParams):
        super().__init__(params)

        assert isinstance(params.args, ag.GATArgs)
        self.args = params.args
        self.all_samples = params.all_samples
        self.graph_data = None
        self.dropout_rate = params.args.dropout_rate

        self.conv1 = tgn.GATConv(params.args.feature_num, params.args.hidden_dim, heads=params.args.head_num, dropout=params.args.dropout_rate, add_self_loops=False)
        self.conv2 = tgn.GATConv(params.args.hidden_dim * params.args.head_num, params.args.class_num, heads=1, concat=False, dropout=params.args.dropout_rate, add_self_loops=False)

    def forward(self, nodes: [[float]]) -> [[float]]:
        x, edge_indices, edge_weights = self.__construct_data(nodes)
        y = self._forward(x, edge_indices, edge_weights)

        if not self.args.learning_mode:
            scores = y
        else:
            scores = y[self.current_indices]
        return scores

    def _forward(self, x, edge_indices, edge_weights) -> [[float]]:
        x = f.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv1.forward(x, edge_indices, edge_attr=edge_weights)
        x = f.elu(x)

        x = f.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2.forward(x, edge_indices, edge_attr=edge_weights)

        return x

    def __construct_data(self, nodes: [[float]]):
        if not self.args.learning_mode:
            samples = self.all_samples[self.current_batch_indices]
            graph_data = self.__build_adjacency_graph(samples, nodes)
        else:
            if self.graph_data is None:
                self.graph_data = self.__build_adjacency_graph(self.all_samples, nodes)
            graph_data = self.graph_data

        return graph_data.x, graph_data.edge_index, graph_data.edge_attr.unsqueeze(-1)

    def __build_adjacency_graph(self, samples: [ds.SampleBase], nodes: [[float]]) -> tgd.Data:
        if self.args.adj_graph_type == es.EAdjGraphType.Phenotype:
            adjacent_graph = bu.build_phenotype_graph(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Euclidean:
            adjacent_graph = gu.build_feature_graph(nodes.cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Unweighted:
            adjacent_graph = bu.build_RA_GCN_graph1(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.CityBlock:
            adjacent_graph = bu.build_RA_GCN_graph2(nodes, "")
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        np.fill_diagonal(adjacent_graph, 0)
        edge_index = torch.tensor(np.nonzero(adjacent_graph), dtype=torch.long).to(ag.Arguments.device)
        edge_weight = torch.tensor(adjacent_graph[adjacent_graph > 0], dtype=torch.float).to(ag.Arguments.device)
        graph_data = tgd.Data(x=nodes, edge_index=edge_index, edge_attr=edge_weight)
        return graph_data
