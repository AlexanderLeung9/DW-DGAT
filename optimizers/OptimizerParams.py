class OptimizerParams(object):
    def __init__(self, learning_rate: float):
        self.learning_rate: float = learning_rate
        # For L2 norm regularization.
        self.weight_decay: float = 0
