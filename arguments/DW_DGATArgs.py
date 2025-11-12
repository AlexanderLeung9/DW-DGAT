import typing as t
import enum as e
import enums as es
import arguments as ag
import optimizers as om


class ESingleGraphModule(e.Enum):
    none = 0,
    gap = 1,
    res_net = 2,
    vit_tiny = 3,
    vit_small = 4,
    swin_t = 5


class DW_DGATArgs(ag.BDArguments):
    def __init__(self, txt_indices: list[int]):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.001)
        self.lowest_accuracy = 0.3
        self.load_OR_save_data = None
        self.batch_size = 64
        self.verbose_logs = False
        self.learning_mode = False
        self.center_num: int = 60
        self.motif_nums: list[int] = []
        self.dropout_rate: float = 0.5
        self.motif_len: int = 3
        self.GGA_head_num: int = 0
        self.free_coefficient: float = 0.5
        self.MLP_layer_num: int = 2
        self.attn_mask: bool = False
        self.related_num: int = 0
        self.diseased_ROIs: list[int] = []

        self.txt_indices = txt_indices
        self.single_graph_module: ESingleGraphModule = ESingleGraphModule.vit_small
        self.adj_graph_type: es.EAdjGraphType = es.EAdjGraphType.Phenotype
        self.cls_gen_train_ratio = 1

    def initialize(self):
        super().initialize()
        metric_num = len(self.txt_indices)
        if metric_num <= 3:
            self.networks_merge_mode = es.ENetworksMergeMode.UpperTriangle
        else:
            self.networks_merge_mode = es.ENetworksMergeMode.L1Norm

        if self.single_graph_module != ESingleGraphModule.none:
            # data fusion + similarity
            self.feature_num = metric_num
            # only data fusion
            # self.feature_num = metric_num - 1

            if self.single_graph_module == ESingleGraphModule.swin_t:
                self.motif_nums = [11, 6, 3]
            else:
                self.motif_nums = [12, 6, 3]

            # Use data fusion.
            strClasses = ",".join(str(tClass) for tClass in self.classes)
            strMotifNums = ",".join(str(motif_num) for motif_num in self.motif_nums)

            if self.single_graph_module.name.startswith("vit_"):
                self.center_num = 45
                pooling = "zero" if self.feature_num == metric_num - 1 else "sim"
                self.data_file_name = f"{self.net_name}-{self.business.name}-{strClasses}-c{self.center_num}-{pooling}-m{metric_num}.dat"
            else:
                self.data_file_name = f"{self.net_name}-{self.business.name}-{strClasses}-c{self.center_num}-{strMotifNums}-m{metric_num}.dat"

            if self.business == es.EBusiness.PD:
                self.GGA_head_num = 6
            else:
                self.GGA_head_num = 8
        else:
            # upper triangle
            if metric_num <= 3:
                self.feature_num = ((0 + ag.BDArguments.ROI_NUM-1) * ag.BDArguments.ROI_NUM // 2) * metric_num
            # only data fusion
            else:
                self.feature_num = (metric_num - 1) * ag.BDArguments.ROI_NUM

            self.GGA_head_num = ag.BDArguments.ROI_NUM // 10

        if self.related_num > 0:
            assert self.related_num == 3, "3 is better than 2."
            if self.business == es.EBusiness.PD:
                self.diseased_ROIs = [1, 2, 7, 8, 19, 20, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 57, 58, 71, 72, 73, 74, 75, 76, 77, 78]
            else:
                self.diseased_ROIs = [13, 14, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 49, 50, 51, 52, 53, 54, 55, 56, 65, 66, 67, 68, 69, 70, 83, 84, 85, 86]
