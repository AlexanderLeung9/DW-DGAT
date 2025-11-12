import torch_geometric.nn as tgn
import networks as nw
import arguments as ag


class NetworkGATv2(nw.NetworkGAT):
    """
    GATv2 for brain networks.
    """
    def __init__(self, params: nw.NetworkParams):
        super().__init__(params)

        assert isinstance(params.args, ag.NetworkGATv2Args)
        self.conv1 = tgn.GATv2Conv(params.args.feature_num, params.args.hidden_dim, heads=params.args.head_num, dropout=self.dropout_rate, concat=True)
        self.conv2 = tgn.GATv2Conv(params.args.hidden_dim * params.args.head_num, params.args.class_num, heads=1, dropout=self.dropout_rate, concat=False)
