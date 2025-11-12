import enums as es
import arguments as ag
import optimizers as om


class MLPArgs(ag.BDArguments):
    def __init__(self, txt_indices: list[int]):
        super().__init__()

        self.stop_value: int = 150
        self.batch_size = 64
        self.optimizer = om.AdamParams(0.001)

        self.txt_indices = txt_indices

        self.dropout_rate: float = 0
        self.expand_OR_contract: bool = True
        self.layer_num: int = 3
        self.output_dim: int = 0

    def initialize(self):
        super().initialize()

        metric_num = len(self.txt_indices)
        if metric_num <= 3:
            self.networks_merge_mode = es.ENetworksMergeMode.UpperTriangle
            self.feature_num = ((0 + ag.BDArguments.ROI_NUM - 1) * ag.BDArguments.ROI_NUM // 2) * metric_num
        else:
            self.networks_merge_mode = es.ENetworksMergeMode.L1Norm
            self.feature_num = (metric_num - 1) * ag.BDArguments.ROI_NUM

        if self.output_dim == 0:
            self.output_dim = self.class_num
