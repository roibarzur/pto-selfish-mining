import numpy as np
from typing import Dict


class MDPMatrix:
    """Matrix for an MDP"""
    
    def __init__(self, num_of_actions: int, num_of_states: int):
        self.num_of_actions = num_of_actions
        self.num_of_states = num_of_states

        self.M = self._build_mat()

    def _build_mat(self) -> np.array:
        return np.zeros((self.num_of_actions, self.num_of_states, self.num_of_states))

    def get_val(self, action: int, from_state: int, to_state: int) -> float:
        return self.M[action, from_state, to_state]

    def get_induced(self, policy: tuple) -> np.array:
        policy = list(policy)
        induced = np.zeros((self.num_of_states, self.num_of_states))
        for i in range(self.num_of_states):
            induced[i, :] = self.M[policy[i], i, :]

        return induced

    def set_batch(self, action: int, from_state: int, transition_values: Dict[int, float]) -> None:
        for to_state, value in transition_values.items():
            self.set(action, from_state, to_state, value)

    def set(self, action: int, from_state: int, to_state: int, value: float) -> None:
        if value != 0:
            self.M[action, from_state, to_state] += value

    def reset(self, action: int, from_state: int, to_state: float) -> None:
        self.M[action, from_state, to_state] = 0

    def get_data(self) -> np.array:
        return self.M
