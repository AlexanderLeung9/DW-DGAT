import enums as es
import arguments as ag
import optimizers as om


class MA_GCNNArgs(ag.BDArguments):
    def __init__(self, txt_indices: list[int]):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.SGDParams(0.001, 0.9)
        self.optimizer.weight_decay = 0.01
        self.dropout_rate: float = 0.5
        # "we adjust the batch size from 45 to 450 to get the best accuracy."
        self.batch_size = 64
        self.load_OR_save_data = None
        self.lowest_accuracy = 0.1
        self.txt_indices = txt_indices
        self.center_num = 60
        self.slot_nums = [12, 6, 3]
        self.motif_len: int = 3
        self.motif_similarity: es.EMotifSimilarity = es.EMotifSimilarity.Euclidean

    def initialize(self):
        super().initialize()

        metrics_num = len(self.txt_indices)
        if metrics_num <= 3:
            self.feature_num = metrics_num * ag.BDArguments.ROI_NUM
        else:
            self.feature_num = metrics_num - 1

        strClasses = ",".join(str(tClass) for tClass in self.classes)
        strSlotNums = ",".join(str(tSlotNum) for tSlotNum in self.slot_nums)
        self.data_file_name = f"{self.net_name}-{self.business.name}-{strClasses}-c{self.center_num}-{strSlotNums}-m{len(self.txt_indices)}.dat"
