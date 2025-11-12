import torch.nn as nn
import torch.nn.functional as f
import arguments as ag
import networks as nw


class MLPParams(nw.NetworkParams):
    def __init__(self, args: ag.MLPArgs):
        super().__init__(args)
        self.args = args
        self.init_params: bool = True


class MLP(nw.NetworkBase):
    def __init__(self, params: MLPParams):
        super().__init__(params)

        self.layer_num = params.args.layer_num
        self.dropout_rate = params.args.dropout_rate

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()

        dim2 = 0
        for i in range(params.args.layer_num):
            if not params.args.expand_OR_contract:
                dim1 = params.args.feature_num if i == 0 else params.args.feature_num * 2 ** i
                dim2 = params.args.feature_num * 2 ** (i + 1)
            else:
                dim1 = params.args.feature_num if i == 0 else params.args.feature_num // 2 ** i
                dim2 = params.args.feature_num // 2 ** (i + 1)
            linear = nn.Linear(dim1, dim2)
            self.linears.append(linear)

            if self.dropout_rate > 0:
                bn = nn.BatchNorm1d(dim2)
                self.bns.append(bn)

        self.linear = nn.Linear(dim2, params.args.output_dim)

        if params.init_params:
            self.apply(self.init_params)

    def forward(self, x: [[float]]) -> [[float]]:
        """
        x.shape = (N, input_dim)
        y.shape = (N, output_dim)
        """
        outputs = x
        for i in range(self.layer_num):
            if self.dropout_rate > 0:
                outputs = f.dropout(outputs, self.dropout_rate, self.training)

            linear = self.linears[i]
            assert isinstance(linear, nn.Linear)
            outputs = linear.forward(outputs)

            if self.dropout_rate > 0:
                bn = self.bns[i]
                assert isinstance(bn, nn.BatchNorm1d)
                outputs = bn.forward(outputs)

            outputs = f.relu(outputs)

        y = self.linear.forward(outputs)
        return y
