import arguments as ag
import optimizers as om


class NetworkGATArgs(ag.LG_GNNArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.005)
        self.optimizer.weight_decay = 0.0005
        self.batch_size = 64
        self.learning_mode = False

        self.dropout_rate: float = 0.1
        self.head_num: int = 8
        self.hidden_dim: int = 64
