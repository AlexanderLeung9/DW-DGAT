import typing as t
import enums as es
import arguments as ag
import optimizers as om
import schedulers as sd


class BrainNetTransformerArgs(ag.BDArguments):
    def __init__(self):
        super().__init__()

        self.optimizer = om.AdamParams(1.0e-4)
        self.optimizer.weight_decay = 1.0e-4
        self.stop_value = 250
        self.scheduler = sd.CosineAnnealingLRParams(1.0e-5)
        self.batch_size = 16
        self.txt_indices = [0]
        self.networks_merge_mode = es.ENetworksMergeMode.Stack

        self.pos_encoding: t.Optional[str] = None
        self.pos_embed_dim: int = 360
        self.sizes: [int] = [360, 100]
        self.pooling: [bool] = [False, True]
        self.orthogonal: bool = True
        self.freeze_center: bool = True
        self.project_assignment: bool = True
        self.head_num: int = 5
