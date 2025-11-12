import torch
import torch.nn as nn
import torch.nn.functional as f
import networks as nw
import arguments as ag


class GCN_MHSAParams(nw.NetworkParams):
    def __init__(self, args: ag.GCN_MHSAArgs):
        super().__init__(args)
        self.args = args


class GCN_MHSA(nw.NetworkBase):
    """
    @article{chen2024gcn,
      title={GCN-MHSA: A novel malicious traffic detection method based on graph convolutional neural network and multi-head self-attention mechanism},
      author={Chen, Jinfu and Xie, Haodi and Cai, Saihua and Song, Luo and Geng, Bo and Guo, Wuhao},
      journal={Computers & Security},
      volume={147},
      pages={104083},
      year={2024},
      publisher={Elsevier}
    }
    """
    def __init__(self, params: GCN_MHSAParams):
        super().__init__(params)

        self.head_num = params.args.head_num
        assert params.args.feature_num % self.head_num == 0, "qkv_dim must be divisible by head_num"
        self.head_dim = params.args.feature_num // self.head_num

        self.gc1 = GraphConvolution(params.args.feature_num, params.args.feature_num, params.args.head_num)
        self.gc2 = GraphConvolution(params.args.feature_num, params.args.feature_num, params.args.head_num)
        self.linear = nn.Linear(params.args.feature_num, params.args.class_num)

    def forward(self, x):
        hidden = self.gc1.forward(x)
        hidden = self.gc2.forward(hidden)
        if self.params.args.learning_mode:
            hidden = hidden[self.current_indices]
        output = self.linear.forward(hidden)

        return output


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_num: int):
        super().__init__()
        assert input_dim % head_num == 0, "input_dim must be divisible by head_num!"

        self.head_num = head_num
        self.head_dim = input_dim // self.head_num

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: [[float]]) -> [[float]]:
        N = x.shape[0]
        # (N, E)
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        # (H, N, E)
        q = q.view(N, self.head_num, self.head_dim).transpose(0, 1)
        k = k.view(N, self.head_num, self.head_dim).transpose(0, 1)
        v = v.view(N, self.head_num, self.head_dim).transpose(0, 1)

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (H, N, N)
        attn = torch.softmax(attn_weights, dim=-1)

        # 1. 加单位矩阵（添加自环）
        I = torch.eye(N, device=attn.device).expand(self.head_num, N, N)
        A = attn + I  # (H, N, N)

        # 2. 计算度向量（每个矩阵一行求和）
        D = A.sum(dim=-1)  # (H, N)

        # 3. 计算 D^{-1/2}，并处理 inf
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # 防止除以0

        # 4. 构造归一化因子 D^{-1/2} 的批量对角形式（使用广播技巧）
        D_left = D_inv_sqrt.unsqueeze(2)  # (H, N, 1)
        D_right = D_inv_sqrt.unsqueeze(1)  # (H, 1, N)

        # 5. 批量归一化：D^{-1/2} @ A @ D^{-1/2} 等价于逐元素乘
        A_norm = A * D_left * D_right  # (H, N, N)

        # (H, N, D)
        Z = torch.bmm(A_norm, v)
        # (N, H*D)
        Z2 = Z.transpose(0, 1).reshape(N, -1)

        D2 = D_inv_sqrt.sum(0)
        # (N, 1)
        D3 = D2.unsqueeze(1)

        hidden1 = D3 * Z2
        hidden2 = self.out_proj.forward(hidden1)
        hidden3 = f.relu(hidden2)

        return hidden3
