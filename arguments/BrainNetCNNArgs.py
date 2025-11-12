import enums as es
import arguments as ag
import optimizers as om


class BrainNetCNNArgs(ag.BDArguments):
    def __init__(self):
        super().__init__()

        self.optimizer = om.SGDParams(0.01, 0.9)
        self.optimizer.weight_decay = 0.0005
        self.optimizer.nesterov = True
        self.batch_size = 14
        self.stop_value = 200
        self.NaN_error_continue = True
        self.txt_indices = [0]
        self.networks_merge_mode = es.ENetworksMergeMode.Stack
        self.feature_num = len(self.txt_indices)
