import torch.nn.functional as f
import torch_geometric.nn as tgn
import arguments as ag
import networks as nw


class GATv2(nw.GAT):
    """
    @article{brody2021attentive,
      title={How attentive are graph attention networks?},
      author={Brody, Shaked and Alon, Uri and Yahav, Eran},
      journal={arXiv preprint arXiv:2105.14491},
      year={2021}
    }
    """
    def __init__(self, params: nw.GCNParams):
        super().__init__(params)
        assert isinstance(params.args, ag.GATv2Args)

        self.conv1 = tgn.GATv2Conv(params.args.feature_num, params.args.hidden_dim, heads=params.args.head_num, dropout=self.dropout_rate, concat=True, edge_dim=1)
        self.conv2 = tgn.GATv2Conv(params.args.hidden_dim * params.args.head_num, params.args.class_num, heads=1, dropout=self.dropout_rate, concat=False, edge_dim=1)

    def _forward(self, x, edge_indices, edge_weights) -> [[float]]:
        x = f.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv1.forward(x, edge_indices, edge_attr=edge_weights)
        x = f.elu(x)

        x = f.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2.forward(x, edge_indices, edge_attr=edge_weights)

        return x
