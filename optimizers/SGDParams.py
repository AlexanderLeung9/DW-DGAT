import optimizers as om


class SGDParams(om.OptimizerParams):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__(learning_rate)
        self.momentum: float = momentum
        self.nesterov: bool = False

    def __str__(self) -> str:
        description = f"name=SGD, learning_rate={self.learning_rate}, weight_decay={self.weight_decay}, momentum={self.momentum}, nesterov={self.nesterov}"
        return description
