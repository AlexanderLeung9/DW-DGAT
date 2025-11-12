import torch
import torch.nn as nn
import torch.nn.functional as f
import arguments as ag
import networks as nw


class MV_GCNParams(nw.NetworkParams):
    """
    @inproceedings{zhang2018multi,
      title={Multi-view graph convolutional network and its applications on neuroimage analysis for parkinson’s disease},
      author={Zhang, Xi and He, Lifang and Chen, Kun and Luo, Yuan and Zhou, Jiayu and Wang, Fei},
      booktitle={AMIA Annual Symposium Proceedings},
      volume={2018},
      pages={1147},
      year={2018},
      organization={American Medical Informatics Association}
    }
    """
    def __init__(self, args: ag.MV_GCNArgs, shared_graphs: [[[float]]], perm: [int], F_in: int, M: int, F_outs: [int], pool_size: int, linear_dim: int):
        super().__init__(args)

        self.args = args
        self.shared_graphs = shared_graphs
        self.perm = perm
        self.F_in = F_in
        self.M = M
        self.F_outs = F_outs
        self.pool_size = pool_size
        self.linear_dim = linear_dim


class MV_GCN(nw.NetworkBase):
    def __init__(self, params: MV_GCNParams):
        super().__init__(params)

        self.shared_graphs = params.shared_graphs
        self.pool_size = params.pool_size
        self.k_order = params.args.k_order
        self.dropout_rate = params.args.dropout_rate

        self.Ws = nn.ModuleList([nn.ParameterList() for _ in range(2)])
        self.bs = nn.ModuleList([nn.ParameterList() for _ in range(2)])
        self.layer_num = len(params.shared_graphs)
        assert self.layer_num == 2

        for i in range(len(self.Ws)):
            for j in range(self.layer_num):
                F_in = params.F_in if j == 0 else params.F_outs[j-1]
                W = nn.Parameter(torch.FloatTensor(F_in * params.args.k_order, params.F_outs[j]).to(ag.Arguments.device))
                W.data.normal_(0, 0.1)
                Ws_i = self.Ws[i]
                if isinstance(Ws_i, nn.ParameterList):
                    Ws_i.append(W)
                else:
                    raise NotImplementedError("Dismiss a warning.")

                F_in = params.shared_graphs[j].shape[0]
                b = nn.Parameter(torch.ones(1, F_in, params.F_outs[j], dtype=torch.float).to(ag.Arguments.device) * 0.1)
                bs_i = self.bs[i]
                if isinstance(bs_i, nn.ParameterList):
                    bs_i.append(b)
                else:
                    raise NotImplementedError("Dismiss a warning.")

        self.linear1 = nn.Linear(params.shared_graphs[-1].shape[0] // params.pool_size, params.linear_dim)
        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear1.bias, 0.1)

        self.linear2 = nn.Linear(params.linear_dim, params.args.class_num)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear2.bias, 0.1)

    def forward(self, data: [[[[[float]]]]]) -> [[float]]:
        """
        :params data: shape=(pair_n, 2, view_n, M, F)
        """
        data = data.to(ag.Arguments.device)
        view_pool_0 = []
        view_pool_1 = []
        view_num = data.shape[2]
        B, M, F = 0, 0, 0
        for i in range(view_num):
            ys = self.__inference_single(data[:, :, i, :, :])
            for j, y in enumerate(ys):
                B, M, F = y.shape
                ys[j] = y.reshape(B, M*F)
            view_pool_0.append(ys[0])
            view_pool_1.append(ys[1])

        pool_vp_0 = MV_GCN.__view_pool(view_pool_0)
        pool_vp_1 = MV_GCN.__view_pool(view_pool_1)

        # Dot product layer
        x_0 = pool_vp_0.reshape(B * M, F)
        x_1 = pool_vp_1.reshape(B * M, F)
        x_0 = f.normalize(x_0, p=2, dim=1, eps=1e-12)
        x_1 = f.normalize(x_1, p=2, dim=1, eps=1e-12)

        x_2 = x_0 * x_1
        x_2 = torch.sum(x_2, dim=1)
        x_2 = x_2.reshape(B, M)

        x_3 = self.linear1.forward(x_2)
        x_3 = f.relu(x_3)
        if self.dropout_rate > 0:
            x_3 = f.dropout(x_3, self.dropout_rate, self.training)

        logits = self.linear2.forward(x_3)
        return logits

    def loss(self, logits: [[float]], targets: [int]):
        loss_value = f.cross_entropy(logits, targets)
        l_w = torch.norm(self.linear1.weight, p=2)
        l_b = torch.norm(self.linear1.bias, p=2)
        reg_loss = l_w + l_b
        loss_value += reg_loss
        return loss_value

    def __inference_single(self, view: [[[[float]]]]):
        """
        :params view: shape=(pair_n, 2, ROI_n, ROI_n)
        """
        output = []
        for i in range(view.shape[1]):
            x_i = view[:, i, :, :]
            for j in range(self.layer_num):
                x_i = self.__chebyshev5(x_i, i, j)
                x_i = self.__b2relu(x_i, i, j)
                x_i = self.__avg_pool(x_i)
            output.append(x_i)

        return output

    def __chebyshev5(self, x: [[[float]]], i: int, j: int):
        """
        :params x: shape=(batch_size, ROI_n, ROI_n)
        """
        B, M, F = x.shape
        x0 = x.permute(1, 2, 0).reshape(M, -1)
        x3 = torch.unsqueeze(x0, 0)

        x1 = torch.sparse.mm(self.shared_graphs[j], x0)
        x2 = torch.unsqueeze(x1, 0)
        x3 = torch.vstack((x3, x2))

        for k in range(2, self.k_order):
            x2 = 2 * torch.sparse.mm(self.shared_graphs[j], x1) - x0
            x4 = torch.unsqueeze(x2, 0)
            x3 = torch.vstack((x3, x4))
            x0, x1 = x1, x2

        x4 = x3.reshape(self.k_order, M, F, B).permute(3, 1, 2, 0).reshape(B*M, F*self.k_order)
        Ws_ij = self.Ws[i][j]
        if isinstance(Ws_ij, nn.Parameter):
            x5 = torch.mm(x4, Ws_ij)
        else:
            raise NotImplementedError("Dismiss a warning.")
        x5 = x5.reshape(B, M, -1)
        return x5

    def __b2relu(self, x: [[[float]]], i: int, j: int):
        x += self.bs[i][j]
        y = f.relu(x)
        return y

    def __avg_pool(self, x):
        """
        Average pooling of size p. Should be a power of 2.
        :params x: shape=(B, M, F_out)
        """
        if self.pool_size > 1:
            B, M, F_out = x.shape
            x1 = x.permute(0, 2, 1).reshape(-1, M)
            padding = (self.pool_size - 1) // 2
            x_padded = f.pad(x1, (padding, padding), mode="constant", value=0)
            output = f.avg_pool1d(x_padded, self.pool_size, self.pool_size, padding=0)
            output = output.reshape(B, F_out, M//self.pool_size)
            output = output.permute(0, 2, 1)
            return output
        else:
            return x

    @staticmethod
    def __view_pool(view_features, method='max'):
        """
        Max pooling of size p. Should be a power of 2.
        :params view_features: shape=(view_n, B, M*F)
        """

        view_features2 = [view_feature.unsqueeze(0) for view_feature in view_features]
        vp = torch.vstack(view_features2)

        if method == 'max':
            vp, _ = torch.max(vp, dim=0)
        elif method == 'mean':
            vp = torch.mean(vp, dim=0)
        return vp
