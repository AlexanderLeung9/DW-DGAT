import sys
import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import functools as ft
import timm.models.swin_transformer_v2_cr as tms
import timm.models.vision_transformer as tmv
import networks as nw
import arguments as ag
import datasets as ds
import enums as es
import utils.GraphUtils as gu
import utils.BDGraphUtils as bu


class DW_DGATParams(nw.NetworkParams):
    def __init__(self, args: ag.DW_DGATArgs, all_samples: [[ds.SampleBase]]):
        super().__init__(args)
        self.args = args
        self.all_samples = all_samples


class DW_DGAT(nw.AdversarialNetwork):
    def __init__(self, params: DW_DGATParams):
        super().__init__(params)
        self.args = params.args

        if params.args.single_graph_module == ag.ESingleGraphModule.none:
            self.singleGCN = None
            self.linear = None
            globalGCN_input_dim = params.args.feature_num
        else:
            self.singleGCN = SingleGraphModule(params.args, params.args.feature_num)
            self.linear = nn.Linear(self.singleGCN.output_dim, params.args.class_num)
            globalGCN_input_dim = self.singleGCN.output_dim

        if params.args.adj_graph_type == es.EAdjGraphType.NoGraph:
            if self.singleGCN is None:
                mlp_input_dim = params.args.feature_num
            else:
                mlp_input_dim = self.singleGCN.output_dim
            self.globalGCN = None
        else:
            mlp_input_dim = globalGCN_input_dim * 2
            self.globalGCN = GlobalGraphModule(params.args, params.all_samples, globalGCN_input_dim, mlp_input_dim, 2, -1, True)

        if params.args.single_graph_module != ag.ESingleGraphModule.none:
            if params.args.adj_graph_type == es.EAdjGraphType.NoGraph:
                self.classification = nn.Linear(mlp_input_dim, params.args.class_num)
            else:
                mlp_args = ag.MLPArgs(params.args.txt_indices)
                mlp_args.dropout_rate = params.args.dropout_rate
                mlp_args.layer_num = params.args.MLP_layer_num
                mlp_args.feature_num = mlp_input_dim
                mlp_args.output_dim = params.args.class_num
                mlp_params = nw.MLPParams(mlp_args)
                if params.args.cls_gen_train_ratio == 0:
                    mlp_params.init_params = True
                else:
                    mlp_params.init_params = False
                self.classification = nw.MLP(mlp_params)
        else:
            self.classification = nn.Linear(mlp_input_dim, params.args.class_num)
            self.classification.reset_parameters()

        self.hidden_output1 = None

    def forward(self, x: [[[float]]]) -> [[float]]:
        if self.args.single_graph_module != ag.ESingleGraphModule.none:
            hidden_output1 = self.singleGCN.forward(x)
            if self.globalGCN is not None:
                if not self.args.learning_mode:
                    self.hidden_output1 = self.linear.forward(hidden_output1)
                else:
                    self.hidden_output1 = self.linear.forward(hidden_output1)[self.current_indices]
        else:
            hidden_output1 = x

        if self.globalGCN is not None:
            if not self.args.learning_mode:
                hidden_output3 = self.globalGCN.forward(hidden_output1, self.current_batch_indices)
            else:
                hidden_output3 = self.globalGCN.forward(hidden_output1, self.current_indices)
        else:
            if not self.args.learning_mode:
                hidden_output3 = hidden_output1
            else:
                hidden_output3 = hidden_output1[self.current_indices]

        scores = self.classification.forward(hidden_output3)
        return scores

    def adversarial_loss(self, logits: [[float]], targets: [int], weight_ratios: [[float]] = None) -> torch.Tensor:
        labels1 = targets.unsqueeze(1)

        max_scores, _ = torch.max(logits, dim=1)
        logits -= max_scores.unsqueeze(1).expand_as(logits)
        probabilities = torch.softmax(logits, dim=1)
        selected_probs1 = torch.gather(probabilities, dim=1, index=labels1).squeeze()
        loss_values = torch.log(selected_probs1)

        if weight_ratios is not None:
            min_weights, _ = torch.min(weight_ratios, dim=1)
            weight_ratios -= min_weights.unsqueeze(1).expand_as(weight_ratios)
            # Larger changes into smaller and vice versa.
            weight_probabilities = torch.softmax(-weight_ratios, dim=1) + sys.float_info.epsilon
            min_probs, _ = weight_probabilities.min(dim=1)
            weight_probabilities1 = weight_probabilities / min_probs.unsqueeze(1).expand_as(weight_probabilities)
            selected_weights1 = torch.gather(weight_probabilities1, dim=1, index=labels1).squeeze()
            loss_values *= selected_weights1

        loss_value = -torch.sum(loss_values) / len(targets)

        if self.hidden_output1 is not None:
            loss1 = f.cross_entropy(self.hidden_output1, targets)
            final_loss = (loss_value + loss1) / 2
        else:
            final_loss = loss_value
        return final_loss


class SingleGraphModule(nn.Module):
    def __init__(self, args: ag.DW_DGATArgs, input_dim: int):
        super().__init__()
        self.args = args

        index1 = np.ceil(np.log2(input_dim))
        index2 = np.floor(np.log2(input_dim))
        index = index1 if index1 > index2 else index1 + 1
        hidden_dim1 = int(2 ** index)
        self.filter1 = nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, args.motif_len), stride=(1, args.motif_len))
        self.bn1 = nn.BatchNorm2d(hidden_dim1)

        hidden_dim2 = hidden_dim1 * 2
        self.filter2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(args.motif_len, 1), stride=(args.motif_len, 1))
        self.bn2 = nn.BatchNorm2d(hidden_dim2)

        # reduce the length of sequence for ViT encoder
        hidden_dim3 = hidden_dim2 * 2
        stride = int(np.ceil(args.motif_len / 2))
        self.filter3 = nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=(args.motif_len, args.motif_len), stride=stride)
        self.bn3 = nn.BatchNorm2d(hidden_dim3)

        if args.single_graph_module == ag.ESingleGraphModule.gap:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.linears = nn.ModuleList()
            self.bns = nn.ModuleList()

            dim2 = 0
            for i in range(args.MLP_layer_num):
                if args.adj_graph_type == es.EAdjGraphType.NoGraph:
                    dim1 = hidden_dim3 if i == 0 else hidden_dim3 // 2 ** i
                    dim2 = hidden_dim3 // 2 ** (i + 1)
                else:
                    dim1 = hidden_dim3 if i == 0 else hidden_dim3 * 2 ** i
                    dim2 = hidden_dim3 * 2 ** (i + 1)

                linear = nn.Linear(dim1, dim2)
                self.linears.append(linear)

                bn = nn.BatchNorm1d(dim2)
                self.bns.append(bn)

            self.output_dim = dim2
            self.apply(nw.NetworkBase.init_weights)

        elif args.single_graph_module == ag.ESingleGraphModule.res_net:
            self.res_net = nw.ResNet2D_2(34, 64)
            self.output_dim = 512

        elif args.single_graph_module == ag.ESingleGraphModule.swin_t:
            self.swin_t = tms.SwinTransformerV2Cr(
                img_size=(20, 20),
                patch_size=1,  # 每个元素是 Patch，不进一步划分
                window_size=5,  # 窗口大小 5×5
                shift_size=2,  # 滑动步长 2
                embed_dim=hidden_dim2,  # 初始通道数
                depths=(2, 2, 2),  # 3个 Stage，分辨率 20→10→5
                num_heads=(2, 4, 8),  # 多头注意力头数
                num_classes=0,  # 移除分类头
                mlp_ratio=4.0,  # FFN 扩展比例（默认值）
                drop_path_rate=0.1,  # 随机深度衰减率（可选）
                pretrained=False,  # 无预训练权重适配此配置
            )
            self.output_dim = hidden_dim2 * 2 * 2
        else:
            # vit_small
            self.output_dim = 384
            head_num = 6
            if args.single_graph_module == ag.ESingleGraphModule.vit_tiny:
                self.output_dim //= 2
                head_num //= 2
            # vit = timm.create_model(f"{args.single_GCN_type.name}_patch16_224")
            # self.output_dim = vit.embed_dim
            # seq_len = 36 if self.args.motif_nums == [11, 6, 3] else 90

            self.embed_layer = nn.Linear(input_dim, self.output_dim)
            # self.embed_layer.reset_parameters()
            depth = 12
            self.vit_blocks = nn.ModuleList()
            for _ in range(depth):
                block = MaskedBlock(
                    dim=self.output_dim,
                    num_heads=head_num,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_norm=False,
                    init_values=None,
                    norm_layer=ft.partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                )
                self.vit_blocks.append(block)
            self.vit_norm = nn.LayerNorm(self.output_dim, eps=1e-6)
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + ag.BDArguments.ROI_NUM, self.output_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))
            init.trunc_normal_(self.pos_embed, std=.02)
            init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x: [[[[float]]]]) -> [[float]]:
        if self.args.single_graph_module.name.startswith("vit_"):
            B, R, F_ = x.shape

            # (B, R, E)
            y1 = self.embed_layer.forward(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            # (B, 1+R, E)
            y2 = torch.cat((cls_tokens, y1), dim=1)
            # (B, 1+R, E)
            y = y2 + self.pos_embed

            # (B, 1+R, E)
            for block in self.vit_blocks:
                assert isinstance(block, MaskedBlock)
                if self.args.attn_mask:
                    attn_masks = SingleGraphModule.__generate_roi_mask(x)
                else:
                    attn_masks = None
                y = block.forward(y, attn_masks)
            y = self.vit_norm.forward(y)

            # (B, E)
            cls_token0 = y[:, 0]
            return cls_token0

        logits = self.filter1(x)
        logits = self.bn1(logits)
        logits = f.leaky_relu(logits)

        logits = self.filter2(logits)
        logits = self.bn2(logits)
        logits = f.leaky_relu(logits)

        if self.args.single_graph_module == ag.ESingleGraphModule.swin_t:
            logits = self.swin_t.stages.forward(logits)
            y = self.swin_t.forward_head(logits)
            return y
        elif self.args.single_graph_module == ag.ESingleGraphModule.res_net:
            y = self.res_net.forward(logits)
            return y
        else:
            logits = self.filter3(logits)
            logits = self.bn3(logits)
            logits = f.leaky_relu(logits)

            logits2 = self.gap.forward(logits)
            y = logits2.squeeze()

            for i in range(self.args.MLP_layer_num):
                y = f.dropout(y, self.args.dropout_rate, self.training)

                linear = self.linears[i]
                assert isinstance(linear, nn.Linear)
                y = linear.forward(y)

                bn = self.bns[i]
                assert isinstance(bn, nn.BatchNorm1d)
                y = bn.forward(y)

                y = f.relu(y)
            return y

    @staticmethod
    def __generate_roi_mask(x):
        B, R, M = x.shape
        # (B, R)
        roi_valid = x.abs().sum(dim=-1) > 1e-6
        attn_mask = torch.zeros(B, 1+R, 1+R, dtype=torch.float32, device=x.device)
        invalid_roi = torch.ones(B, 1+R, dtype=torch.bool, device=x.device)
        invalid_roi[:, 1:] = ~roi_valid
        attn_mask[invalid_roi.unsqueeze(2) | invalid_roi.unsqueeze(1)] = float('-inf')
        attn_mask = attn_mask.unsqueeze(1)  # 添加 head 维度

        return attn_mask


class MaskedBlock(tmv.Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: t.Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = tmv.Mlp,
    ) -> None:
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_drop=proj_drop,
                         attn_drop=attn_drop, init_values=init_values, drop_path=drop_path,
                         act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer)
        self.attn = MaskedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop,
                                    proj_drop=proj_drop, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor, mask: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn.forward(self.norm1(x), mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskedAttention(tmv.Attention):
    def forward(self, x: torch.Tensor, mask: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = f.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalGraphModule(nn.Module):
    def __init__(self, args: ag.DW_DGATArgs, all_samples: [[ds.SampleBase]], input_dim: int, output_dim: int, layer_num: int, masked_label: int, gc_type: bool):
        super().__init__()

        self.args = args
        self.all_samples = all_samples
        self.masked_label = masked_label
        self.gc_type = gc_type
        self.adjacency_graph = None

        dim2 = 0
        self.gcs = nn.ModuleList()
        for i in range(layer_num):
            dim1 = input_dim if i == 0 else input_dim * 2 ** i
            dim2 = input_dim * 2 ** (i + 1)
            if self.gc_type:
                gc = MHSAGraphConvolution(dim1, dim2, args.GGA_head_num, args.dropout_rate)
            else:
                gc = ConventionalGraphConvolution(dim1, dim2)
            self.gcs.append(gc)

        self.fc = nn.Linear(dim2, output_dim)

        # self.apply(nw.NetworkBase.init_params)

    def forward(self, nodes: [[float]], current_batch_indices: [int]) -> [[float]]:
        adjacency_graph = self._construct_adjacency_graph(nodes, current_batch_indices)

        hidden = f.dropout(nodes, self.args.dropout_rate, self.training)
        for gc in self.gcs:
            if self.gc_type:
                assert isinstance(gc, MHSAGraphConvolution)
                hidden = gc.forward(hidden, adjacency_graph)
            else:
                assert isinstance(gc, ConventionalGraphConvolution)
                hidden = gc.forward(hidden, adjacency_graph)

        output = self.fc.forward(hidden)
        if self.args.learning_mode:
            current_indices = current_batch_indices
            output = output[current_indices]
        return output

    def _construct_adjacency_graph(self, nodes: [[float]], current_batch_indices: [int]) -> torch.Tensor:
        if not self.args.learning_mode:
            samples = self.all_samples[current_batch_indices]
            adjacency_graph = self.__build_adjacency_graph(samples, nodes)
        else:
            samples = self.all_samples
            if self.adjacency_graph is None:
                self.adjacency_graph = self.__build_adjacency_graph(samples, nodes)
            adjacency_graph = self.adjacency_graph

        if self.masked_label >= 0:
            graph_masks = bu.build_masked_graph(samples, self.masked_label)
            graph_masks = torch.from_numpy(graph_masks).to(adjacency_graph.device)
            adjacency_graph = graph_masks * adjacency_graph

        return adjacency_graph

    def __build_adjacency_graph(self, samples: [ds.SampleBase], nodes: [[float]]) -> torch.Tensor:
        if self.args.adj_graph_type == es.EAdjGraphType.Phenotype:
            adjacent_graph = bu.build_phenotype_graph(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Euclidean:
            adjacent_graph = gu.build_feature_graph(nodes.detach().cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Unweighted:
            adjacent_graph = bu.build_RA_GCN_graph1(samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.CityBlock:
            adjacent_graph = bu.build_RA_GCN_graph2(nodes.detach().cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Relationship:
            adjacent_graph = bu.build_relationship_graph(samples, np.float32)
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        if self.args.learning_mode and nodes.shape[0] > 1:
            gu.draw_heatmap(adjacent_graph, self.args.adj_graph_type.name, self.args)
        adjacent_graph, _ = gu.normalize_adjacent_graph(adjacent_graph, "gcn")
        adjacent_graph = torch.from_numpy(adjacent_graph).to(ag.Arguments.device)

        return adjacent_graph


class ConventionalGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: [[float]], adjacent_graph: [[float]]) -> [[float]]:
        hidden = torch.mm(x, self.weight)
        hidden = torch.mm(adjacent_graph, hidden)
        hidden = self.norm(hidden)
        hidden = f.relu(hidden, inplace=True)
        return hidden


class MHSAGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_num: int, dropout_rate: float):
        super().__init__()

        # self.MHSA = nn.MultiheadAttention(input_dim, head_num, kdim=output_dim, vdim=output_dim, batch_first=True)
        # self.MHSA = tmv.Attention(input_dim, head_num)
        self.MHSA = CustomMHSA(input_dim, output_dim, head_num, dropout_rate)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: [[float]], adjacent_graph: [[int]]) -> [[float]]:
        N_N_rows = adjacent_graph.unsqueeze(2) * x.unsqueeze(0)
        attn_output = self.MHSA.forward(N_N_rows)

        hidden = attn_output.sum(dim=1)
        hidden = self.norm(hidden)
        hidden = f.gelu(hidden)
        return hidden


class CustomMHSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_num, dropout_rate: float):
        super().__init__()

        assert input_dim % head_num == 0, "input_dim must be divisible by head_num!"
        self.head_num = head_num
        self.head_dim = input_dim // head_num
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(input_dim, output_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.head_num, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DW_DGATWeightGenerator(nw.AdversarialNetwork):
    def __init__(self, params: DW_DGATParams):
        super().__init__(params)

        self.args = params.args
        self.all_samples = params.all_samples
        self.class_indices: [[int]] = [[] for _ in range(self.args.class_num)]

        self.singleGCN = SingleGraphModule(params.args, params.args.feature_num)
        globalGCN_input_dim = self.singleGCN.output_dim

        self.globalGCNs = nn.ModuleList()
        self.globalGCN_output_dim = globalGCN_input_dim * 2

        for i in range(self.args.class_num):
            globalGCN = GlobalGraphModule(params.args, params.all_samples, globalGCN_input_dim, self.globalGCN_output_dim, 2, i, True)
            self.globalGCNs.append(globalGCN)

        mlp_args = ag.MLPArgs(params.args.txt_indices)
        mlp_args.dropout_rate = params.args.dropout_rate
        mlp_args.layer_num = params.args.MLP_layer_num
        mlp_args.feature_num = self.globalGCN_output_dim
        mlp_args.output_dim = params.args.class_num
        mlp_params = nw.MLPParams(mlp_args)
        mlp_params.init_params = False
        self.classification = nw.MLP(mlp_params)

        self.apply(self.__init_params)

    @staticmethod
    def __init_params(m: nn.Module):
        if isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm3d):
            m.reset_parameters()
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()
        elif isinstance(m, nn.Conv1d):
            m.reset_parameters()
        elif isinstance(m, nn.Conv2d):
            m.reset_parameters()
        elif isinstance(m, nn.Conv3d):
            m.reset_parameters()

    @property
    def current_indices(self) -> [int]:
        return super().current_indices

    @current_indices.setter
    def current_indices(self, value: [int]):
        """
        The DW_DGANWeightGenerator is used only in the training phase.
        """
        self._current_indices = value

        self.class_indices.clear()
        for i in range(self.args.class_num):
            self.class_indices.append([])

        for index in self._current_indices:
            label = self.all_samples[index].label
            self.class_indices[label].append(index)

    def forward(self, x: [[[float]]]) -> [[float]]:
        if self.args.single_graph_module != ag.ESingleGraphModule.none:
            hidden_output1 = self.singleGCN.forward(x)
        else:
            hidden_output1 = x

        if not self.args.learning_mode:
            N = x.shape[0]
        else:
            N = len(self.current_indices)

        hidden_output3 = torch.zeros((self.args.class_num, N, self.globalGCN_output_dim), dtype=torch.float).to(ag.Arguments.device)
        for i, globalGCN in enumerate(self.globalGCNs):
            assert isinstance(globalGCN, GlobalGraphModule)
            if not self.args.learning_mode:
                hidden_output3[i, ...] = globalGCN.forward(hidden_output1, self.current_batch_indices)
            else:
                hidden_output3[i, ...] = globalGCN.forward(hidden_output1, self.current_indices)

            if not self.args.learning_mode:
                other_classes = np.setdiff1d(self.current_batch_indices, self.class_indices[i])
                other_indices = np.where(np.isin(self.current_batch_indices, other_classes))[0]
            else:
                other_classes = np.setdiff1d(self.current_indices, self.class_indices[i])
                other_indices = np.where(np.isin(self.current_indices, other_classes))[0]
            hidden_output3[i][other_indices] = 0

        hidden_output4 = torch.sum(hidden_output3, dim=0)
        scores = self.classification.forward(hidden_output4)
        return scores

    def adversarial_loss(self, weight_ratios: [[float]], targets: [int], logits: [[float]]) -> torch.Tensor:
        labels1 = targets.unsqueeze(1)

        max_scores, _ = torch.max(logits, dim=1)
        logits -= max_scores.unsqueeze(1).expand_as(logits)
        probabilities = torch.softmax(logits, dim=1)
        selected_probs1 = torch.gather(probabilities, dim=1, index=labels1).squeeze()
        loss_values = torch.log(selected_probs1)

        min_weights, _ = torch.min(weight_ratios, dim=1)
        weight_ratios -= min_weights.unsqueeze(1).expand_as(weight_ratios)
        # Larger values change into smaller and vice versa.
        weight_probabilities = torch.softmax(-weight_ratios, dim=1) + sys.float_info.epsilon
        min_probs, _ = weight_probabilities.min(dim=1)
        weight_probabilities1 = weight_probabilities / min_probs.unsqueeze(1).expand_as(weight_probabilities)
        selected_weights1 = torch.gather(weight_probabilities1, dim=1, index=labels1).squeeze()
        loss_values *= selected_weights1

        # nan_count = torch.isnan(loss_values).sum()
        # if nan_count > 0:
        #     print(nan_count)
        # nan_count = torch.isnan(weight_probabilities).sum()
        # if nan_count > 0:
        #     print(nan_count)
        loss_value1 = -torch.sum(loss_values) / len(targets)
        loss_value4 = torch.log(weight_probabilities)
        # nan_count = torch.isnan(loss_value4).sum()
        # if nan_count > 0:
        #     print(nan_count)
        # if (weight_probabilities == 0).any():
        #     print("(weight_probabilities == 0).any()")
        # if torch.isinf(loss_value4).any():
        #     print("torch.isinf(loss_value4).any()")
        loss_value2 = self.args.free_coefficient * torch.sum(weight_probabilities * loss_value4)
        # if torch.isnan(loss_value2):
        #     print(loss_value2)
        number = len(targets) * self.args.class_num
        loss_value3 = loss_value2 / number
        # if torch.isnan(loss_value3):
        #     print(loss_value3)
        final_loss = loss_value1 - loss_value3
        return final_loss
