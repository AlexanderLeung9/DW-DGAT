import arguments as ag
import optimizers as om
import schedulers as sc


class BrainGNNArgs(ag.LG_GNNArgs):
    def __init__(self):
        super().__init__()

        self.stop_value: int = 100
        self.optimizer = om.AdamParams(0.01)
        self.optimizer.weight_decay = 5e-3
        self.scheduler = sc.StepLRParams()
        self.scheduler.step_size = 20
        self.scheduler.gamma = 0.5
        self.batch_size = 100
        self.learning_mode = False

        self.pooling_ratio: float = 0.5
        self.layer_num: int = 2
        self.lamb0: float = 1
        self.lamb1: float = 0
        self.lamb2: float = 0
        self.lamb3: float = 0.1
        self.lamb4: float = 0.1
        self.lamb5: float = 0.1
