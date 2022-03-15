import numpy as np
import torch
import torch.nn as nn

from ..base.blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ..base.experience_acquisition.experience_batch import ExperienceBatch
from ..base.function_approximation.approximator import Approximator
from ..base.function_approximation.loss_function import LossFunction


# noinspection PyAbstractClass
class MCTSLossFunction(LossFunction):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, cross_entropy_loss_weight: float,
                 use_action_prior: bool, beta: float, normalize_target_values: bool, normalization_momentum: float = 0):
        super().__init__(approximator, None, simulator.expected_horizon)
        self.simulator = simulator
        self.cross_entropy_loss_weight = cross_entropy_loss_weight
        self.use_action_prior = use_action_prior
        self.beta = beta
        self.normalize_target_values = normalize_target_values
        self.normalization_momentum = normalization_momentum
        self.target_values_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch: ExperienceBatch, **kwargs) -> torch.Tensor:
        all_predicted_values = self.approximator(batch.prev_states)
        predicted_q_values = torch.gather(all_predicted_values, dim=1, index=batch.actions.unsqueeze(1)).squeeze()

        q_values = batch.target_values[:, 0]

        if self.normalize_target_values:
            self.target_values_mean *= self.normalization_momentum
            # weighted_q_values = q_values * batch.prev_difficulty_contributions \
            #                     / torch.sum(batch.prev_difficulty_contributions)
            # weighted_q_values = q_values * batch.difficulty_contributions / torch.sum(batch.difficulty_contributions)
            weighted_q_values = q_values
            self.target_values_mean += (1 - self.normalization_momentum) * np.float32(weighted_q_values.mean().item())
            q_values -= self.target_values_mean

        pi_values = batch.target_values[:, 1:]

        # loss = nn.functional.smooth_l1_loss(predicted_q_values, q_values, beta=self.beta)
        loss = nn.functional.mse_loss(predicted_q_values, q_values)

        if self.use_action_prior:
            num_of_actions = all_predicted_values.shape[1] // 2
            predicted_pi_values = nn.functional.log_softmax(all_predicted_values[:, num_of_actions:], dim=1)
            loss -= self.cross_entropy_loss_weight * torch.mul(predicted_pi_values, pi_values).sum(dim=1).mean()

        return loss
