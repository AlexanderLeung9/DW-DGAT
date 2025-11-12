import enums as es
import arguments as ag
import optimizers as om
import schedulers as sd


class MV_GCNArgs(ag.BDArguments):
    def __init__(self, adj_graph_type: es.EAdjGraphType):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.005)
        self.scheduler = sd.ExponentialLRParams()
        self.learning_mode = True
        self.txt_indices = [0, 1, 2, 29]
        self.networks_merge_mode = es.ENetworksMergeMode.Stack
        self.k_order: int = 3
        self.kNN: int = 10
        self.dropout_rate: float = 0.0
        self.adj_graph_type: es.EAdjGraphType = adj_graph_type
