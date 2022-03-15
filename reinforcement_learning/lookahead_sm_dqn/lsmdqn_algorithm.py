from .lsmdqn_agent import LSMDQNAgent
from ..base.function_approximation.huber_loss_function import HuberLossFunction
from ..base.function_approximation.mlp_approximator import MLPApproximator
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.utility.parameter_schedule import ParameterSchedule


class LSMDQNAlgorithm(RLAlgorithm):
    def create_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [256, 256])
        dropout = self.creation_args.get('dropout', 0)
        return MLPApproximator(self.device, self.simulator.state_space_dim, self.simulator.num_of_actions,
                               hidden_layers_sizes, dropout)

    def create_agent(self) -> LSMDQNAgent:
        starting_epsilon = self.creation_args.get('starting_epsilon', 0.5)
        epsilon_step = self.creation_args.get('epsilon_step', 0.5 / 10000000)
        use_boltzmann = self.creation_args.get('use_boltzmann', False)
        boltzmann_temperature = self.creation_args.get('boltzmann_temperature', 1)
        depth = self.creation_args.get('depth', 1)
        ground_initial_state = self.creation_args.get('ground_initial_state', True)
        value_clip = self.creation_args.get('value_clip', 0)
        return LSMDQNAgent(self.create_approximator(), self.simulator, starting_epsilon=starting_epsilon,
                           epsilon_step=epsilon_step, use_boltzmann=use_boltzmann,
                           boltzmann_temperature=boltzmann_temperature, depth=depth,
                           ground_initial_state=ground_initial_state, value_clip=value_clip)

    def create_loss_fn(self) -> HuberLossFunction:
        diff_penalty_start = self.creation_args.get('diff_penalty_start', 0)
        diff_penalty_step = self.creation_args.get('diff_penalty_step', 0)
        return HuberLossFunction(self.approximator, self.simulator,
                                 target_l2_penalty_schedule=ParameterSchedule(diff_penalty_start, diff_penalty_step))
