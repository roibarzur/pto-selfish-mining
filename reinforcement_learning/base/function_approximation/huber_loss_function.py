import torch
import torch.nn as nn

from .approximator import Approximator
from .loss_function import LossFunction
from ..blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ..experience_acquisition.experience_batch import ExperienceBatch
from ..utility.parameter_schedule import ParameterSchedule


# noinspection PyAbstractClass
class HuberLossFunction(LossFunction):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator,
                 target_l2_penalty_schedule: ParameterSchedule = None, beta: float = 1,
                 normalize_target_values: bool = False, normalization_momentum: float = 0):
        super().__init__(approximator, None, simulator.expected_horizon)
        self.simulator = simulator
        self.target_l2_penalty_schedule = target_l2_penalty_schedule or ParameterSchedule(0, 0)
        self.beta = beta
        self.normalize_target_values = normalize_target_values
        self.normalization_momentum = normalization_momentum
        self.target_values_mean = 0

    def forward(self, batch: ExperienceBatch, **kwargs) -> torch.Tensor:
        target_values = batch.target_values
        if self.normalize_target_values:
            self.target_values_mean *= self.normalization_momentum
            self.target_values_mean += (1 - self.normalization_momentum) * target_values.mean().item()
            target_values -= self.target_values_mean

        all_predicted_q_values = self.approximator(batch.prev_states)
        predicted_q_values = torch.gather(all_predicted_q_values, dim=1, index=batch.actions.unsqueeze(1)).squeeze()

        target_l2_penalty = self.target_l2_penalty_schedule.get_parameter()
        target_l2_loss = torch.square(predicted_q_values).mean()

        # return nn.functional.smooth_l1_loss(predicted_q_values, batch.target_values.squeeze(), beta=self.beta) \
        return nn.functional.mse_loss(predicted_q_values, target_values.squeeze()) \
               + target_l2_penalty * target_l2_loss
