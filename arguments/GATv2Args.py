import enums as es
import arguments as ag


class GATv2Args(ag.GATArgs):
    def __init__(self, txt_indices: list[int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices, adj_graph_type)
        pass
