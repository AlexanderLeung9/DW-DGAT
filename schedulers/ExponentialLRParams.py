import schedulers as sd


class ExponentialLRParams(sd.SchedulerParams):
    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        description = f"name=ExponentialLR, gamma={self.gamma}"
        return description
