from __future__ import annotations

from abc import ABC

import torch
import torch.nn as nn


class Approximator(nn.Module, ABC):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model: nn.Module

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(state_tensor)

    def update(self, approximator: Approximator):
        self.load_state_dict(approximator.state_dict())
