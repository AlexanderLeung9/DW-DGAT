import enums as es
import arguments as ag
import optimizers as om


class RA_GCNArgs(ag.GCNArgs):
    def __init__(self, txt_indices: list[int], adj_graph_type: es.EAdjGraphType):
        super().__init__(txt_indices, adj_graph_type)
        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.001)
        self.NaN_error_continue = True
        self.lowest_accuracy = 0.1
        self.learning_mode = True
        self.gen_lr_rate: float = 0.01
        self.dropout_rate: float = 0.5
        self.free_coefficient = 0.5
        self.cls_gen_train_ratio = 1
        self.hidden_dim = 4
        self.VAE_hidden_dim: int = 8
        self.VAE_batch_size: int = 32
        self.VAE_num_epochs: int = 200

    def initialize(self):
        super().initialize()

        metric_num = len(self.txt_indices)
        if metric_num <= 3:
            self.networks_merge_mode = es.ENetworksMergeMode.UpperTriangle
            self.feature_num = ((0 + ag.BDArguments.ROI_NUM-1) * ag.BDArguments.ROI_NUM // 2) * metric_num
        else:
            self.networks_merge_mode = es.ENetworksMergeMode.L1Norm
            self.feature_num = (metric_num - 1) * ag.BDArguments.ROI_NUM

        strClasses = ",".join(str(tClass) for tClass in self.classes)
        self.data_file_name = f"{self.net_name}-{self.business.name}-{strClasses}-m{len(self.txt_indices)}.dat"
