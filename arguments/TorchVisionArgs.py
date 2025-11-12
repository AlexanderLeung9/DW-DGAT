import typing as t
import enums as es
import arguments as ag
import optimizers as om


class TorchVisionArgs(ag.BDArguments):
    def __init__(self, net_name: t.Optional[str] = None):
        super().__init__()

        if net_name is not None:
            self._net_name = net_name
        self.optimizer = om.AdamParams(0.001)
        self.batch_size = 64
        self.pretrained_weights_folder = None

        self.txt_indices = [0, 1, 2]
        self.feature_num = len(self.txt_indices)
        self.networks_merge_mode = es.ENetworksMergeMode.Stack

    def initialize(self):
        super().initialize()

        if self._net_name != "":
            if self._net_name.startswith("VGGNet"):
                self.stop_value = 250
            elif self._net_name.startswith("ResNet"):
                self.stop_value = 150
            elif self._net_name.startswith("DenseNet"):
                self.stop_value = 150

            self.net_state_file = ""
