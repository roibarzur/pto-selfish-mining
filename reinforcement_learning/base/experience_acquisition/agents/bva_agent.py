from abc import ABC
from typing import Optional

import torch

from .caching_agent import CachingAgent
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator


class BVAAgent(CachingAgent, ABC):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, use_cache: bool = True):
        super().__init__(approximator, simulator, use_cache)
        self.base_value_approximation = 0

    def update(self, approximator: Optional[Approximator] = None, base_value_approximation: Optional[float] = None,
               **kwargs) -> None:
        super().update(approximator, **kwargs)

        if base_value_approximation is not None:
            self.base_value_approximation = base_value_approximation

    def reduce_to_v_table(self) -> torch.Tensor:
        v_table = super().reduce_to_v_table()
        if hasattr(self, 'nn_factor'):
            v_table *= self.nn_factor

        v_table += self.base_value_approximation

        return v_table
