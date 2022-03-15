from .lddqn_agent import LDDQNAgent
from ..base.function_approximation.huber_loss_function import HuberLossFunction
from ..base.function_approximation.mlp_approximator import MLPApproximator
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.utility.parameter_schedule import ParameterSchedule


class LDDQNAlgorithm(RLAlgorithm):
    def create_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [256, 256])
        dropout = self.creation_args.get('dropout', 0)
        use_normalization = self.creation_args.get('use_normalization', [False])
        use_norm_bias = self.creation_args.get('use_norm_bias', [False])
        return MLPApproximator(self.device, self.simulator.state_space_dim, self.simulator.num_of_actions,
                               hidden_layers_sizes, dropout, normalize_outputs=use_normalization,
                               use_norm_bias=use_norm_bias)

    def create_agent(self) -> LDDQNAgent:
        starting_epsilon = self.creation_args.get('starting_epsilon', 0.5)
        epsilon_step = self.creation_args.get('epsilon_step', 0.5 / 10000000)
        use_boltzmann = self.creation_args.get('use_boltzmann', False)
        boltzmann_temperature = self.creation_args.get('boltzmann_temperature', 1)
        depth = self.creation_args.get('depth', 1)
        use_base_approximation = self.creation_args.get('use_base_approximation', True)
        ground_initial_state = self.creation_args.get('ground_initial_state', True)
        value_clip = self.creation_args.get('value_clip', 0)
        nn_factor = self.creation_args.get('nn_factor', 1 / self.simulator.expected_horizon)
        return LDDQNAgent(self.create_approximator(), self.create_approximator(), self.simulator,
                          starting_epsilon=starting_epsilon, epsilon_step=epsilon_step, use_boltzmann=use_boltzmann,
                          boltzmann_temperature=boltzmann_temperature, depth=depth,
                          use_base_approximation=use_base_approximation, ground_initial_state=ground_initial_state,
                          value_clip=value_clip, nn_factor=nn_factor)

    def create_loss_fn(self) -> HuberLossFunction:
        diff_penalty_start = self.creation_args.get('diff_penalty_start', 0)
        diff_penalty_step = self.creation_args.get('diff_penalty_step', 0)
        huber_beta = self.creation_args.get('huber_beta', float('inf'))
        return HuberLossFunction(self.approximator, self.simulator,
                                 target_l2_penalty_schedule=ParameterSchedule(diff_penalty_start, diff_penalty_step),
                                 beta=huber_beta)
