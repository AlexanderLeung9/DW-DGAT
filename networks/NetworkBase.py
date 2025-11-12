import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import typing as t
import arguments as ag
from abc import abstractmethod, ABC


class NetworkParams(object):
    def __init__(self, args: ag.Arguments):
        self.args = args


class NetworkBase(nn.Module):
    def __init__(self, params: t.Optional[NetworkParams]):
        super().__init__()
        self.params = params
        self._current_indices: [int] = np.array([], dtype=np.int32)
        self.batch_index: int = -1

    @staticmethod
    def init_params(m: nn.Module):
        if isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm3d):
            m.reset_parameters()
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()
        elif isinstance(m, nn.Conv1d):
            m.reset_parameters()
        elif isinstance(m, nn.Conv2d):
            m.reset_parameters()
        elif isinstance(m, nn.Conv3d):
            m.reset_parameters()
        elif isinstance(m, nn.Linear):
            m.reset_parameters()

    # Applicable to load partial parameters.
    def load_parameters(self, state_no: int):
        pass

    # Applicable to save partial parameters.
    def save_parameters(self, state_no: int):
        pass

    @property
    def current_indices(self) -> [int]:
        return self._current_indices

    @current_indices.setter
    def current_indices(self, value: [int]):
        self._current_indices = value

    @property
    def current_batch_indices(self) -> [int]:
        start_index = self.args.batch_size * self.batch_index
        end_index = self.args.batch_size * (self.batch_index + 1)
        if end_index > len(self.current_indices):
            end_index = len(self.current_indices)
        current_batch = self.current_indices[start_index:end_index]
        return current_batch

    def loss(self, scores: [[float]], labels: [int]) -> torch.Tensor:
        assert isinstance(self, NetworkBase)
        return f.cross_entropy(scores, labels)


class AdversarialNetwork(NetworkBase, ABC):
    @abstractmethod
    def adversarial_loss(self, my_logits: [[float]], labels: [int], opponent_logits: [[float]]) -> torch.Tensor:
        raise NotImplementedError
