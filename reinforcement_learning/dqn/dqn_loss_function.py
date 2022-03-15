import torch
import torch.nn as nn

from ..base.experience_acquisition.experience_batch import ExperienceBatch
from ..base.function_approximation.loss_function import LossFunction


# noinspection PyAbstractClass
class DQNLossFunction(LossFunction):
    def forward(self, batch: ExperienceBatch) -> torch.Tensor:
        all_predicted_q_values = self.approximator(batch.prev_states)
        predicted_q_values = torch.gather(all_predicted_q_values, dim=1, index=batch.actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_states_q_values = self.target_approximator(batch.next_states)
            legal_next_states_q_values = next_states_q_values.masked_fill_(mask=~batch.legal_actions_list,
                                                                           value=float('-inf'))
            best_next_q_value = legal_next_states_q_values.max(dim=1).values

            # If the simulator is done then no next q value
            best_next_q_value = torch.mul(best_next_q_value, ~batch.is_done_list)

            effective_discount_factor = torch.pow(
                torch.tensor([1 - 1 / self.expected_horizon], device=best_next_q_value.device),
                batch.difficulty_contributions)

            target_q_values = batch.rewards + torch.mul(best_next_q_value, effective_discount_factor)

        return nn.functional.mse_loss(predicted_q_values, target_q_values)
