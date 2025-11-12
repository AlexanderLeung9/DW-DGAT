import arguments as ag
import optimizers as om


class SwinTransformerArgs(ag.TorchVisionArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 250
        self.optimizer = om.AdamParams(0.001)
        self.lowest_accuracy = 0.2
