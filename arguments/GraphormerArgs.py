import enums as es
import arguments as ag
import optimizers as om


class GraphormerArgs(ag.LG_GNNArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 250
        self.optimizer = om.AdamParams(0.001)
        self.optimizer.weight_decay = 0.0005
        self.batch_size: int = 64
        self.learning_mode = False

        self.dropout_rate: float = 0.5
        self.layer_num = 3
        self.input_node_dim = 3
        self.node_dim = 128
        self.input_edge_dim = 1
        self.edge_dim = 128
        self.output_dim = 2
        self.head_num = 4
        self.ff_dim = 256
        self.max_in_degree = 5
        self.max_out_degree = 5
        self.max_path_distance = 5
