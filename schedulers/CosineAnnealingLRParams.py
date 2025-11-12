import schedulers as sd


class CosineAnnealingLRParams(sd.SchedulerParams):
    def __init__(self, eta_min: float):
        super().__init__()

        self.T_max: int = 0
        self.eta_min: float = eta_min
