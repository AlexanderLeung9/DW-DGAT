import arguments as ag
import optimizers as om


class ViTArgs(ag.TorchVisionArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 500
