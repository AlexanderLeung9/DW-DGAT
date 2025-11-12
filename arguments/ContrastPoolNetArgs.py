import arguments as ag
import enums as es


class ContrastPoolNetArgs(ag.BDArguments):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 1000
        self.optimizer = "Adam"
        self.learning_rate = 1e-2
        self.batch_size = 4
        self.txt_indices = [0, 23, 24, 25]
        self.networks_merge_mode = es.ENetworksMergeMode.Stack
        self.pool_ratio: float = 0.5
        self.input_dim: int = 3
        self.hidden_dim: int = 86
        self.layer_num: int = 2
        self.dropout_rate: float = 0.0
        self.batch_norm: bool = True
        self.residual: bool = True
        self.aggregator_type: str = "maxpool"
        self.lambda1: float = 0.001
        self.learnable_q: bool = False
        self.edge_feat: bool = False
        self.max_num_node: int = ag.BDArguments.ROI_NUM
        self.init_lr: float = 1e-2
        self.weight_decay: float = 0.0
        self.lr_reduce_factor: float = 0.5
        self.lr_schedule_patience: int = 25
