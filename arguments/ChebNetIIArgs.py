import enums as es
import arguments as ag
import optimizers as om


class ChebNetIIArgs(ag.GCNArgs):
    def __init__(self, txt_indices: list[int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices, adj_graph_type)

        self.optimizer = om.AdamParams(0.01)
        self.optimizer.weight_decay = 0.0005
        self.dropout_rate: float = 0.5
        self.k_order = 4
        self.hidden_dim = 64
