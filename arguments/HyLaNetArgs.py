import enums as es
import arguments as ag
import optimizers as om


class HyLaNetArgs(ag.GCNArgs):
    def __init__(self, txt_indices: [int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices, adj_graph_type)

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.1)
        self.k_order = 2
        self.he_dim = 2
        self.hyla_dim = 100
        self.sparse = True
        self.lambda_scale = 0.07
