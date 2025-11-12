import schedulers as sd


class MultiStepLRParams(sd.SchedulerParams):
    def __init__(self):
        super().__init__()
        self.milestones: [int] = []

    def __str__(self) -> str:
        description = f"name=MultiStepLR, gamma={self.gamma}, milestones={self.milestones}"
        return description
