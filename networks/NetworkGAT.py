import torch
import torch.nn.functional as f
import torch_geometric.nn as tgn
import torch_geometric.utils as tgu
import torch_geometric.data as tgd
import networks as nw
import arguments as ag


class NetworkGAT(nw.NetworkBase):
    """
    GAT for brain networks.
    """
    def __init__(self, params: nw.NetworkParams):
        super().__init__(params)

        assert isinstance(params.args, ag.NetworkGATArgs)
        self.dropout_rate = params.args.dropout_rate
        self.conv1 = tgn.GATConv(params.args.feature_num, params.args.hidden_dim, heads=params.args.head_num, dropout=params.args.dropout_rate, add_self_loops=False)
        self.conv2 = tgn.GATConv(params.args.hidden_dim * params.args.head_num, params.args.class_num, heads=1, concat=False, dropout=params.args.dropout_rate, add_self_loops=False)

    def forward(self, data: list) -> torch.Tensor:
        FA_networks = data[0]
        FA_weights = data[1]

        batch_list = []
        for i in range(FA_networks.size(0)):
            FA_network = FA_networks[i]
            FA_weight = FA_weights[i]

            edge_index, edge_weight = tgu.dense_to_sparse(FA_network)
            batch_list.append(tgd.Data(x=FA_weight, edge_index=edge_index, edge_attr=edge_weight))

        graphs = tgd.Batch.from_data_list(batch_list)
        assert isinstance(graphs, tgd.Data)

        x = f.dropout(graphs.x, self.dropout_rate, training=self.training)
        x = self.conv1.forward(x, graphs.edge_index, edge_attr=graphs.edge_weight)
        x = f.elu(x)

        x = f.dropout(x, self.dropout_rate, training=self.training)
        scores = self.conv2.forward(x, graphs.edge_index, edge_attr=graphs.edge_weight)

        mean_scores = NetworkGAT.__scatter_mean_alternative(scores, graphs.batch, graphs.num_graphs)
        return mean_scores

    @staticmethod
    def __scatter_mean_alternative(src, index, dim_size):
        """
        scatter_mean of PyTorch version, which doesn't depend on torch_scatter.
        :param src: (N, feature_dim) tensor to be summed
        :param index: (N, ) indices in this batch
        :param dim_size: total number of this batch
        """
        sum_tensor = torch.zeros(dim_size, src.size(1), device=src.device)
        count_tensor = torch.zeros(dim_size, device=src.device)

        sum_tensor.index_add_(0, index, src)  # sum
        count_tensor.index_add_(0, index, torch.ones_like(index, dtype=torch.float))

        count_tensor = count_tensor.clamp(min=1)  # avoid to divide 0
        average = sum_tensor / count_tensor.unsqueeze(1)
        return average
