import schedulers as sd


class LambdaLRParams(sd.SchedulerParams):
    def __init__(self):
        super().__init__()
        self.lr_lambda = None

    def __str__(self) -> str:
        description = f"name=LambdaLR, gamma={self.gamma}, lr_lambda={self.lr_lambda}"
        return description
