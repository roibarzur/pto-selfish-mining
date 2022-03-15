from abc import ABC, abstractmethod
from typing import Optional

import torch

from blockchain_mdps import BlockchainModel
from ..agents.planning_agent import PlanningAgent
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator


class CachingAgent(PlanningAgent, ABC):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, use_cache: bool = True):
        super().__init__(approximator, simulator)
        self.state_value_cache = {}
        self.use_cache = use_cache

    def get_state_evaluation(self, state: BlockchainModel.State, exploring: bool) -> torch.Tensor:
        if state not in self.state_value_cache or not self.use_cache:
            state_tensor = torch.tensor(state, device=self.simulator.device, dtype=torch.float)
            state_eval = self.evaluate_state(state_tensor, exploring)

            if self.use_cache:
                self.state_value_cache[state] = state_eval
            else:
                return state_eval

        return self.state_value_cache[state]

    @abstractmethod
    def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
        pass

    def update(self, approximator: Optional[Approximator] = None, **kwargs) -> None:
        super().update(approximator, **kwargs)
        self.state_value_cache = {}
