from __future__ import annotations

from typing import List, NamedTuple

import torch

from .experience import Experience


class ExperienceBatch(NamedTuple):
    prev_states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    difficulty_contributions: torch.Tensor
    prev_difficulty_contributions: torch.Tensor
    is_done_list: torch.Tensor
    legal_actions_list: torch.Tensor
    target_values: torch.Tensor

    @staticmethod
    def from_experience_list(experience_list: List[Experience], device: torch.device = torch.device('cpu')
                             ) -> ExperienceBatch:
        prev_states = torch.stack([s.prev_state for s in experience_list]).to(device)
        actions = torch.tensor([s.action for s in experience_list], device=device, dtype=torch.long)
        next_states = torch.stack([s.next_state for s in experience_list]).to(device)
        rewards = torch.tensor([s.reward for s in experience_list], device=device, dtype=torch.float)
        difficulty_contributions = torch.tensor([s.difficulty_contribution for s in experience_list], device=device,
                                                dtype=torch.float)
        prev_difficulty_contributions = torch.tensor([s.prev_difficulty_contribution for s in experience_list],
                                                     device=device, dtype=torch.float)
        is_done_list = torch.tensor([s.is_done for s in experience_list], device=device, dtype=torch.bool)
        legal_actions_list = torch.stack([s.legal_actions for s in experience_list]).to(device)

        if all(s.target_value is not None for s in experience_list):
            target_values = torch.stack([s.target_value for s in experience_list]).to(device)
        else:
            target_values = None

        return ExperienceBatch(prev_states=prev_states, actions=actions, next_states=next_states, rewards=rewards,
                               difficulty_contributions=difficulty_contributions,
                               prev_difficulty_contributions=prev_difficulty_contributions, is_done_list=is_done_list,
                               legal_actions_list=legal_actions_list, target_values=target_values)

    def combine(self, other: ExperienceBatch) -> ExperienceBatch:
        prev_states = torch.cat([self.prev_states, other.prev_states])
        actions = torch.cat([self.actions, other.actions])
        next_states = torch.cat([self.next_states, other.next_states])
        rewards = torch.cat([self.rewards, other.rewards])
        difficulty_contributions = torch.cat([self.difficulty_contributions, other.difficulty_contributions])
        prev_difficulty_contributions = torch.cat(
            [self.prev_difficulty_contributions, other.prev_difficulty_contributions])
        is_done_list = torch.cat([self.is_done_list, other.is_done_list])
        legal_actions_list = torch.cat([self.legal_actions_list, other.legal_actions_list])
        target_values = torch.cat([self.target_values, other.target_values])

        return ExperienceBatch(prev_states=prev_states, actions=actions, next_states=next_states, rewards=rewards,
                               difficulty_contributions=difficulty_contributions,
                               prev_difficulty_contributions=prev_difficulty_contributions, is_done_list=is_done_list,
                               legal_actions_list=legal_actions_list, target_values=target_values)

    def __len__(self) -> int:
        return len(self.prev_states)
