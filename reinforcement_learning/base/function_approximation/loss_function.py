from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .approximator import Approximator
from ..experience_acquisition.experience_batch import ExperienceBatch


class LossFunction(nn.Module, ABC):
    def __init__(self, approximator: Approximator, target_approximator: Optional[Approximator], expected_horizon: int):
        super().__init__()
        self.approximator = approximator
        self.target_approximator = target_approximator
        self.expected_horizon = expected_horizon

    @abstractmethod
    def forward(self, batch: ExperienceBatch) -> torch.Tensor:
        pass

    def update(self) -> None:
        # Copy approximator to target approximator
        if self.target_approximator is not None:
            self.target_approximator.load_state_dict(self.approximator.state_dict())
