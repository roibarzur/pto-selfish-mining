from blockchain_mdps import BlockchainModel
from .mcts_agent import MCTSAgent
from .mcts_approximator import MCTSApproximator
from .mcts_loss_function import MCTSLossFunction
from ..base.function_approximation.approximator import Approximator
from ..base.function_approximation.table_approximator import TableApproximator
from ..base.training.rl_algorithm import RLAlgorithm


class MCTSAlgorithm(RLAlgorithm):
    def __init__(self, blockchain_model: BlockchainModel, **creation_args) -> None:
        self.blockchain_model = blockchain_model
        super().__init__(**creation_args)

    def create_approximator(self) -> Approximator:
        use_table = self.creation_args.get('use_table', False)
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [256, 256])
        dropout = self.creation_args.get('dropout', 0)
        q_table = self.creation_args.get('q_table')
        bias_in_last_layer = self.creation_args.get('bias_in_last_layer', True)
        use_normalization = self.creation_args.get('use_normalization', False)
        use_norm_bias = self.creation_args.get('use_norm_bias', False)
        momentum = self.creation_args.get('momentum', 0.99)
        if use_table:
            return TableApproximator(self.device, self.blockchain_model, self.simulator, weights=q_table)
        else:
            return MCTSApproximator(self.device, self.simulator.state_space_dim, self.simulator.num_of_actions,
                                    hidden_layers_sizes, dropout, bias_in_last_layer, use_normalization, use_norm_bias,
                                    momentum)

    def create_agent(self) -> MCTSAgent:
        starting_epsilon = self.creation_args.get('starting_epsilon', 0.2)
        epsilon_step = self.creation_args.get('epsilon_step', 0.2 / 10000000)
        use_boltzmann = self.creation_args.get('use_boltzmann', False)
        boltzmann_temperature = self.creation_args.get('boltzmann_temperature', 1)
        target_pi_temperature = self.creation_args.get('target_pi_temperature', 1)
        puct_const = self.creation_args.get('puct_const', 1)
        use_action_prior = self.creation_args.get('use_action_prior', True)
        depth = self.creation_args.get('depth', 250)
        mc_simulations = self.creation_args.get('mc_simulations', 100)
        warm_up_simulations = self.creation_args.get('warm_up_simulations', 0)
        use_base_approximation = self.creation_args.get('use_base_approximation', True)
        ground_initial_state = self.creation_args.get('ground_initial_state', True)
        use_cached_values = self.creation_args.get('use_cached_values', True)
        value_clip = self.creation_args.get('value_clip', 0)
        nn_factor = self.creation_args.get('nn_factor', 1 / self.simulator.expected_horizon)
        prune_tree_rate = self.creation_args.get('prune_tree_rate', 250)
        root_dirichlet_noise = self.creation_args.get('root_dirichlet_noise', 0.5)
        return MCTSAgent(
            self.create_approximator(),
            self.simulator,
            starting_epsilon=starting_epsilon,
            epsilon_step=epsilon_step,
            use_boltzmann=use_boltzmann,
            boltzmann_temperature=boltzmann_temperature,
            target_pi_temperature=target_pi_temperature,
            puct_const=puct_const,
            use_action_prior=use_action_prior,
            depth=depth,
            mc_simulations=mc_simulations,
            warm_up_simulations=warm_up_simulations,
            use_base_approximation=use_base_approximation,
            ground_initial_state=ground_initial_state,
            use_cached_values=use_cached_values,
            value_clip=value_clip,
            nn_factor=nn_factor,
            prune_tree_rate=prune_tree_rate,
            root_dirichlet_noise=root_dirichlet_noise
        )

    def create_loss_fn(self) -> MCTSLossFunction:
        cross_entropy_loss_weight = self.creation_args.get('cross_entropy_loss_weight', 1)
        use_action_prior = self.creation_args.get('use_action_prior', True)
        huber_beta = self.creation_args.get('huber_beta', float('inf'))
        normalize_target_values = self.creation_args.get('normalize_target_values', False)
        normalization_momentum = self.creation_args.get('normalization_momentum', 0.9)
        return MCTSLossFunction(self.approximator, self.simulator, cross_entropy_loss_weight, use_action_prior,
                                huber_beta, normalize_target_values, normalization_momentum)

    # def create_optimizer(self) -> torch.optim.Optimizer:
    #    return torch.optim.SGD(self.approximator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
