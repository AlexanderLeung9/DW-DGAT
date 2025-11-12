import enums as es
import arguments as ag
import optimizers as om


class GCNArgs(ag.MLPArgs):
    def __init__(self, txt_indices: list[int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices)

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.001)
        self.optimizer.weight_decay = 0.0005
        self.dropout_rate: float = 0.5
        self.hidden_dim = 16
        self.adj_graph_type: es.EAdjGraphType = adj_graph_type
