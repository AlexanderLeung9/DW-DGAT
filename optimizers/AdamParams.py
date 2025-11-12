import optimizers as om


class AdamParams(om.OptimizerParams):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)
        self.betas: tuple[float, float] = (0.9, 0.999)

    def __str__(self) -> str:
        description = f"name=Adam, learning_rate={self.learning_rate}, weight_decay={self.weight_decay}, betas={self.betas}"
        return description
