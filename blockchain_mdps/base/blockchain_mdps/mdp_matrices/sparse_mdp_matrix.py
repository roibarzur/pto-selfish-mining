from scipy.sparse import csr_matrix, lil_matrix
from typing import List

import numpy as np

from .mdp_matrix import MDPMatrix


class SparseMDPMatrix(MDPMatrix):
    """Sparse matrix for an MDP"""
    
    def __init__(self, num_of_actions: int, num_of_states: int):
        super().__init__(num_of_actions, num_of_states)

    def _build_mat(self) -> List[lil_matrix]:
        return [lil_matrix((self.num_of_states, self.num_of_states), dtype=np.float64) for _ in
                range(self.num_of_actions)]

    def get_val(self, action: int, from_state: int, to_state: int) -> float:
        return self.M[action][from_state, to_state]

    def get_induced(self, policy: List[int]) -> csr_matrix:
        policy = np.array(policy)
        induced = csr_matrix((self.num_of_states, self.num_of_states))
        for action in range(self.num_of_actions):
            induced += self.M[action].multiply(np.expand_dims(policy == action, axis=1))

        return induced

    def set(self, action: int, from_state: int, to_state: int, value: float) -> None:
        self.M[action][from_state, to_state] += value

    def reset(self, action: int, from_state: int, to_state: int) -> None:
        self.M[action][from_state, to_state] = 0

    def get_data(self) -> List[lil_matrix]:
        return self.M.copy()
