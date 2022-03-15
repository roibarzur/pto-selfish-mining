from typing import Tuple, Optional, Dict

import numpy as np
import torch

from blockchain_mdps import BlockchainModel
from ..base.blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ..base.experience_acquisition.exploaration_mechanisms.epsilon_greedy_exploration import EpsilonGreedyExploration
from ..base.experience_acquisition.exploaration_mechanisms.state_dependant_boltzmann_exploration import \
    StateDependantBoltzmannExploration
from ..base.experience_acquisition.agents.bva_agent import BVAAgent
from ..base.function_approximation.approximator import Approximator


class LSMDQNAgent(BVAAgent):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, starting_epsilon: float = 0.5,
                 epsilon_step: float = 0, use_boltzmann: bool = False, boltzmann_temperature: float = 1, depth: int = 1,
                 ground_initial_state: bool = True, value_clip: float = 0):
        super().__init__(approximator, simulator)
        if use_boltzmann:
            self.exploration_mechanism = StateDependantBoltzmannExploration(boltzmann_temperature)
        else:
            self.exploration_mechanism = EpsilonGreedyExploration(starting_epsilon, epsilon_step)

        self.depth = depth
        assert self.depth > 0

        self.ground_initial_state = ground_initial_state
        self.value_clip = value_clip
        assert self.value_clip >= 0

        self.deep_state_value_cache = {}

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'exploration_mechanism': self.exploration_mechanism, 'depth': self.depth,
             'ground_initial_state': self.ground_initial_state, 'value_clip': self.value_clip}
        return str(d)

    def plan_action(self, explore: bool = True) -> Tuple[int, torch.Tensor]:
        state_tuple = self.simulator.torch_to_tuple(self.current_state)
        _, action_values = self.state_value(state_tuple, self.depth, explore)
        action = self.invoke_exploration_mechanism(self.exploration_mechanism, action_values, explore)
        target_value = action_values[action]

        if self.value_clip > 0:
            target_value = np.clip(target_value, -self.value_clip, self.value_clip)

        return action, torch.tensor([target_value], device=self.simulator.device, dtype=torch.float)

    def action_value(self, state: BlockchainModel.State, action: int, depth: int, exploring: bool) -> float:
        transition_values = self.simulator.get_state_transition_values(state, action)
        total_value = 0
        for next_state in transition_values.probabilities.keys():
            value, _ = self.state_value(next_state, depth - 1, exploring)

            # Add transition reward
            value += transition_values.rewards[next_state]

            # Remove difficulty contribution
            value -= self.base_value_approximation * transition_values.difficulty_contributions[next_state]

            # Multiply by transition probability
            value *= transition_values.probabilities[next_state]

            total_value += value

        return total_value

    def state_value(self, state: BlockchainModel.State, depth: int, exploring: bool
                    ) -> Tuple[float, Optional[Dict[int, float]]]:
        if depth == 0 or self.ground_initial_state and exploring and self.simulator.is_initial_state(state)\
                and depth < self.depth:
            return self.get_state_evaluation(state, exploring).item(), None

        if (state, depth) in self.deep_state_value_cache:
            action_values = self.deep_state_value_cache[state, depth]
        else:
            action_values = dict((action, self.action_value(state, action, depth, exploring))
                                 for action in self.simulator.get_state_legal_actions(state))
            self.deep_state_value_cache[state, depth] = action_values

        value = max(action_values.values())

        return value, action_values

    def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
        if self.ground_initial_state and exploring and self.simulator.is_initial_state(state):
            value = 0
        else:
            legal_actions_tensor = self.simulator.get_state_legal_actions_tensor(state)

            with torch.no_grad():
                q_values = self.approximator(state)
                legal_q_values = q_values.masked_fill_(mask=~legal_actions_tensor, value=float('-inf'))
                value = legal_q_values.max()

        if self.value_clip > 0:
            value = torch.clamp(value, -self.value_clip, self.value_clip)

        return value

    def update(self, approximator: Optional[Approximator] = None, base_value_approximation: Optional[float] = None,
               **kwargs) -> None:
        super().update(approximator, base_value_approximation, **kwargs)
        self.deep_state_value_cache = {}
