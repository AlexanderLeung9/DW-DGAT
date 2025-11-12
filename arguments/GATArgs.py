import enums as es
import arguments as ag
import optimizers as om


class GATArgs(ag.GCNArgs):
    def __init__(self, txt_indices: list[int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices, adj_graph_type)

        self.optimizer = om.AdamParams(0.005)
        self.optimizer.weight_decay = 0.0005
        # self.lowest_accuracy = 0.3
        self.dropout_rate: float = 0.1
        self.head_num: int = 8
        self.hidden_dim: int = 64
