import torch
import torch.nn as nn
import torch.nn.functional as f
import networks as nw
import arguments as ag


class MA_GCNN(nw.NetworkBase):
    """
    @inproceedings{peng2020motif,
      title={Motif-matching based subgraph-level attentional convolutional network for graph classification},
      author={Peng, Hao and Li, Jianxin and Gong, Qiran and Ning, Yuanxin and Wang, Senzhang and He, Lifang},
      booktitle={Proceedings of the AAAI conference on Artificial Intelligence},
      volume={34},
      pages={5387--5394},
      year={2020}
    }
    """
    def __init__(self, params: nw.NetworkParams):
        super().__init__(params)

        assert isinstance(params.args, ag.MA_GCNNArgs)
        self.hidden_dim1 = 32
        self.singleGCN = SingleGCN(params.args.feature_num, self.hidden_dim1, params.args.dropout_rate)

        col_num = (21 * 3 - 3 - 2 * 0) // 3 + 1
        col_num = (col_num - 3 - 2 * 0) // 3 + 1
        input_dim2 = 2 * self.hidden_dim1 * col_num
        self.gat = GAT_without_Graph(input_dim2, 16, params.args.class_num)

    def forward(self, x: [[[float]]]) -> [[float]]:
        x = x.to(ag.Arguments.device)
        hidden_output1 = self.singleGCN.forward(x)
        hidden_output2 = self.gat.forward(hidden_output1)
        soft_layer = f.softmax(hidden_output2, dim=2)
        scores = torch.sum(soft_layer, dim=1)
        return scores


class SingleGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate

        # kernel_size: (height, width)
        self.filter1 = nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, 3), stride=(1, 3))
        self.bn1 = nn.BatchNorm2d(hidden_dim1)

        hidden_dim2 = hidden_dim1 * 2
        self.filter2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(1, 3), stride=(1, 3))
        self.bn2 = nn.BatchNorm2d(hidden_dim2)

    def forward(self, x: [[[float]]]) -> [[float]]:
        logits = self.filter1(x)
        logits = f.dropout(logits, self.dropout_rate, self.training)

        logits = self.filter2(logits)
        logits = f.dropout(logits, self.dropout_rate, self.training)

        logits = logits.permute(0, 3, 1, 2)
        logits = logits.reshape(logits.shape[0], -1, logits.shape[3])
        return logits


class Attention_Head_without_Graph(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, residual: bool = False):
        super().__init__()

        self.filter1 = nn.Conv1d(input_dim, hidden_dim, 1, bias=True)
        self.filter2 = nn.Conv1d(hidden_dim, 1, 1)
        self.filter3 = nn.Conv1d(hidden_dim, 1, 1)

        if residual:
            self.filter4 = nn.Conv1d(input_dim, hidden_dim, 1)
        else:
            self.filter4 = None

    def forward(self, hidden_input: [[[float]]]) -> [[float]]:
        seq_fts = self.filter1(hidden_input)
        f_1 = self.filter2(seq_fts)
        f_2 = self.filter3(seq_fts)

        f_3 = f_2.permute(0, 2, 1)
        logits = f_1 + f_3
        logits = f.relu(logits)

        coefficients = f.softmax(logits)
        seq_fts2 = seq_fts.permute(0, 2, 1)
        output = torch.matmul(coefficients, seq_fts2)

        if self.filter4 is not None:
            x2 = self.filter4(hidden_input)
            x2 = x2.permute(0, 2, 1)
            output = output + x2

        return output


class GAT_without_Graph(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_num: int):
        super().__init__()

        self.attention_heads = nn.ModuleList()
        for i in range(8):
            attention_head = Attention_Head_without_Graph(input_dim, hidden_dim)
            self.attention_heads.append(attention_head)

        self.last_ah = Attention_Head_without_Graph(hidden_dim * 8, class_num)

    def forward(self, x: [[[float]]]) -> [[float]]:
        hidden_outputs = []
        for attention_head in self.attention_heads:
            hidden_output = attention_head.forward(x)
            hidden_output = f.elu(hidden_output)
            hidden_outputs.append(hidden_output)
        hidden_outputs = torch.concatenate(hidden_outputs, dim=-1)

        hidden_outputs = hidden_outputs.permute(0, 2, 1)
        logits = self.last_ah.forward(hidden_outputs)

        return logits
