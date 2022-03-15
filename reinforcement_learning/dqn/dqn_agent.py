import torch

from ..base.blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ..base.experience_acquisition.agents.agent import Agent
from ..base.experience_acquisition.exploaration_mechanisms.epsilon_greedy_exploration import EpsilonGreedyExploration
from ..base.function_approximation.approximator import Approximator


class DQNAgent(Agent):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, starting_epsilon: float,
                 epsilon_step: float):
        super().__init__(approximator, simulator)
        self.exploration_mechanism = EpsilonGreedyExploration(starting_epsilon=starting_epsilon,
                                                              epsilon_step=epsilon_step)

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'exploration_mechanism': self.exploration_mechanism}
        return str(d)

    def choose_action(self, explore: bool = True) -> int:
        with torch.no_grad():
            q_values = self.approximator(self.current_state)
            q_values.masked_fill_(mask=~self.legal_actions, value=float('-inf'))

        return self.invoke_exploration_mechanism(self.exploration_mechanism, q_values, explore)
