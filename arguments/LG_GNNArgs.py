import enums as es
import arguments as ag
import optimizers as om


class LG_GNNArgs(ag.BDArguments):
    """
    《Classification of Brain Disorders in rs-fMRI via Local-to-Global Graph Neural Networks》
    """
    def __init__(self):
        super().__init__()

        self.stop_value: int = 500
        self.optimizer = om.AdamParams(0.01)
        self.learning_mode = True

        self.txt_indices: [int] = [0, 23, 24, 25]
        self.feature_num = len(self.txt_indices) - 1
        self.networks_merge_mode = es.ENetworksMergeMode.Stack

        self.dropout_rate: float = 0.3
        self.k_order: int = 3
