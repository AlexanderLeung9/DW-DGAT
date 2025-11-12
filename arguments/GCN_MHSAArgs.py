import enums as es
import arguments as ag
import optimizers as om


class GCN_MHSAArgs(ag.BDArguments):
    def __init__(self, txt_indices: list[int], networks_merge_mode: es.ENetworksMergeMode):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.01)
        self.NaN_error_continue = True
        self.lowest_accuracy = 0.2
        self.batch_size = 64

        self.txt_indices = txt_indices
        self.networks_merge_mode = networks_merge_mode
        
        self.dropout_rate: float = 0.5
        self.head_num: int = ag.BDArguments.ROI_NUM // 10
        
    def initialize(self):
        super().initialize()

        metric_num = len(self.txt_indices)
        if metric_num > 3:
            assert self.networks_merge_mode == es.ENetworksMergeMode.L1Norm
            self.feature_num = (metric_num - 1) * ag.BDArguments.ROI_NUM
        else:
            if self.networks_merge_mode == es.ENetworksMergeMode.L1Norm:
                self.feature_num = metric_num * ag.BDArguments.ROI_NUM
            elif self.networks_merge_mode == es.ENetworksMergeMode.UpperTriangle:
                self.feature_num = ((0 + ag.BDArguments.ROI_NUM - 1) * ag.BDArguments.ROI_NUM // 2) * metric_num
            else:
                self.feature_num = ag.BDArguments.ROI_NUM * metric_num
