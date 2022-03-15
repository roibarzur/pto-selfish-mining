from __future__ import annotations

from typing import Optional

import torch


class Experience:
    def __init__(self, prev_state: Optional[torch.Tensor], action: Optional[int], next_state: Optional[torch.Tensor],
                 reward: Optional[float], difficulty_contribution: Optional[float],
                 prev_difficulty_contribution: Optional[float], is_done: Optional[bool],
                 legal_actions: Optional[torch.Tensor], target_value: Optional[torch.Tensor], info: Optional[dict]):
        self.prev_state = prev_state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.difficulty_contribution = difficulty_contribution
        self.prev_difficulty_contribution = prev_difficulty_contribution
        self.is_done = is_done
        self.legal_actions = legal_actions
        self.target_value = target_value
        self.info = info

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if state['prev_state'] is not None:
            state['prev_state'] = state['prev_state'].numpy()

        if state['next_state'] is not None:
            state['next_state'] = state['next_state'].numpy()

        if state['legal_actions'] is not None:
            state['legal_actions'] = state['legal_actions'].numpy()

        if state['target_value'] is not None:
            state['target_value'] = state['target_value'].numpy()

        return state

    def __setstate__(self, state: dict) -> None:
        state = state.copy()

        if state['prev_state'] is not None:
            state['prev_state'] = torch.tensor(state['prev_state'])

        if state['next_state'] is not None:
            state['next_state'] = torch.tensor(state['next_state'])

        if state['legal_actions'] is not None:
            state['legal_actions'] = torch.tensor(state['legal_actions'])

        if state['target_value'] is not None:
            state['target_value'] = torch.tensor(state['target_value'])

        self.__dict__ = state

    @staticmethod
    def create_dummy() -> Experience:
        return Experience(prev_state=None, action=None, next_state=None, reward=None, difficulty_contribution=None,
                          prev_difficulty_contribution=None, is_done=None, legal_actions=None, target_value=None,
                          info=None)
