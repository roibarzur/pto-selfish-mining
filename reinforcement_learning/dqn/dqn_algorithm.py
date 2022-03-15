from .dqn_agent import DQNAgent
from .dqn_loss_function import DQNLossFunction
from ..base.function_approximation.mlp_approximator import MLPApproximator
from ..base.training.rl_algorithm import RLAlgorithm


class DQNAlgorithm(RLAlgorithm):
    def create_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [256, 256])

        return MLPApproximator(self.device, self.simulator.state_space_dim,
                               self.simulator.num_of_actions, hidden_layers_sizes=hidden_layers_sizes)

    def create_agent(self) -> DQNAgent:
        starting_epsilon = self.creation_args.get('starting_epsilon', 0.5)
        epsilon_step = self.creation_args.get('epsilon_step', 0.5 / 1000000)
        return DQNAgent(self.create_approximator(), self.simulator, starting_epsilon=starting_epsilon,
                        epsilon_step=epsilon_step)

    def create_loss_fn(self) -> DQNLossFunction:
        return DQNLossFunction(self.approximator, self.create_approximator(), self.simulator.expected_horizon)
