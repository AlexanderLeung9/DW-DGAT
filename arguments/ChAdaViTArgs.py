import arguments as ag
import optimizers as om


class ChAdaViTArgs(ag.TorchVisionArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 100
        self.optimizer = om.AdamParams(0.0001)
        self.lowest_accuracy = 0.2
        self.batch_size: int = 16
        self.dropout_rate: float = 0.0

        self.embed_dim = 192
        self.patch_size = 6
        self.depth = 12
        self.num_heads = 12
        self.drop_path_rate = 0.
        self.max_number_channels = 3
        self.return_all_tokens = False
