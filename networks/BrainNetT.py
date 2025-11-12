import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
import networks as nw
import arguments as ag
from .ptdec import DEC


class BrainNetT(nw.NetworkBase):
    """
    @article{kan2022brain,
      title={Brain network transformer},
      author={Kan, Xuan and Dai, Wei and Cui, Hejie and Zhang, Zilong and Guo, Ying and Yang, Carl},
      journal={Advances in Neural Information Processing Systems},
      volume={35},
      pages={25586--25599},
      year={2022}
    }
    """
    def __init__(self, params: nw.NetworkParams):
        super().__init__(params)
        assert isinstance(params.args, ag.BrainNetTransformerArgs)

        self.attention_list = nn.ModuleList()
        self.assignments = []
        forward_dim = ag.BDArguments.ROI_NUM

        self.pos_encoding = params.args.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(ag.BDArguments.ROI_NUM, params.args.pos_embed_dim), requires_grad=True)
            forward_dim = ag.BDArguments.ROI_NUM + params.args.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = params.args.sizes
        sizes[0] = ag.BDArguments.ROI_NUM
        in_sizes = [ag.BDArguments.ROI_NUM] + sizes[:-1]
        do_pooling = params.args.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=params.args.orthogonal,
                                    freeze_center=params.args.freeze_center,
                                    project_assignment=params.args.project_assignment,
                                    head_num=params.args.head_num)
            )

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, params.args.class_num),
        )

    def forward(self, node_feature: torch.Tensor):
        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        self.assignments = []

        for attn in self.attention_list:
            node_feature, assignment = attn(node_feature)
            self.assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [attn.get_attention_weights() for attn in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, head_num: int = 4):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, head_num=head_num,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class InterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, head_num, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, head_num, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: t.Optional[torch.Tensor] = None

    def _sa_block(self, x: torch.Tensor, attn_mask: t.Optional[torch.Tensor], key_padding_mask: t.Optional[torch.Tensor], is_causal: bool = False)\
            -> torch.Tensor:
        x, weights = self.self_attn.forward(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> t.Optional[torch.Tensor]:
        return self.attention_weights
