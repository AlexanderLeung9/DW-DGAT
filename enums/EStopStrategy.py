import enum as e
import numpy as np


class EStopStrategy(e.Enum):
    FixedEpochs = 0
    MinLoss = 1
    EarlyStop = 2


class EarlyStopping(object):
    def __init__(self, loss_OR_accuracy: bool, patience: int, min_delta: float = 0.0):
        """
        :param loss_OR_accuracy: False: loss; True: accuracy
        :param patience: tolerated number of epochs
        :param min_delta: minimum change in loss or accuracy
        """
        self.standard: bool = loss_OR_accuracy
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_value: float = np.inf if not loss_OR_accuracy else 0.0
