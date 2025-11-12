import schedulers as sd


class StepLRParams(sd.SchedulerParams):
    def __init__(self):
        super().__init__()
        self.step_size: int = 0

    def __str__(self) -> str:
        description = f"name=StepLR, gamma={self.gamma}, step_size={self.step_size}"
        return description
