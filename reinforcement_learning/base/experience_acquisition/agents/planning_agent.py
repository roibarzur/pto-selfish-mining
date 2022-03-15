from abc import ABC, abstractmethod
from typing import Tuple

import torch

from .agent import Agent
from ..experience import Experience
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator


class PlanningAgent(Agent, ABC):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator):
        super().__init__(approximator, simulator)
        self.chosen_action = int(0)

    def choose_action(self, explore: bool = True) -> int:
        return self.chosen_action

    @abstractmethod
    def plan_action(self, explore: bool = True) -> Tuple[int, torch.Tensor]:
        pass

    def step(self, explore: bool = True) -> Experience:
        action, target_value = self.plan_action(explore=explore)
        self.chosen_action = action

        exp = super().step(explore=explore)
        exp.target_value = target_value

        return exp

    def reduce_to_v_table(self) -> torch.Tensor:
        v_table = torch.zeros(self.simulator.num_of_states)
        for state_index, state in enumerate(self.simulator.enumerate_states()):
            self.reset(state)

            try:
                _, target = self.plan_action(explore=False)
                target = target[0].item()
            except ValueError:
                target = float('-inf')

            v_table[state_index] = target

        return v_table

    def reduce_to_q_table(self) -> torch.Tensor:
        v_table = self.reduce_to_v_table()

        q_table = torch.zeros((self.simulator.num_of_states, self.simulator.num_of_actions), dtype=torch.float)
        for state_index, state in enumerate(self.simulator.enumerate_states()):
            q_values = torch.zeros((self.simulator.num_of_actions,), dtype=torch.float)

            for action in self.simulator.get_state_legal_actions(state):
                transition_values = self.simulator.get_state_transition_values(state, action)
                total_value = 0

                for next_state in transition_values.probabilities.keys():
                    value = v_table[next_state]

                    # Discount by difficulty contribution
                    difficulty_contribution = transition_values.difficulty_contributions[next_state]
                    value *= (1 - 1 / self.simulator.expected_horizon) ** difficulty_contribution

                    # Add transition reward
                    value += transition_values.rewards[next_state]

                    # Multiply by transition probability
                    value *= transition_values.probabilities[next_state]

                    total_value += value

                q_values[action] = total_value

            legal_actions = self.simulator.get_state_legal_actions_tensor(state)
            legal_q_values = q_values.masked_fill_(mask=~legal_actions, value=float('-inf'))
            q_table[state_index, :] = legal_q_values

        return q_table
