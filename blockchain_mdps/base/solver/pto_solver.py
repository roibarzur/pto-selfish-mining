from typing import List, Tuple, Union

import mdptoolbox.mdp as mdptoolbox
import numpy as np
from scipy.sparse import csr_matrix, find, spmatrix

from .blockchain_mdp_solver import BlockchainMDPSolver
from ..blockchain_mdps.blockchain_mdp import BlockchainMDP
from ..blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP
from ..blockchain_model import BlockchainModel


class PTOSolver(BlockchainMDPSolver):
    def __init__(self, model: BlockchainModel, expected_horizon: int = 10 ** 5, use_sparse: bool = True):
        self.expected_horizon = expected_horizon
        self.use_sparse = use_sparse
        mdp = SparseBlockchainMDP(model) if self.use_sparse else BlockchainMDP(model)
        super().__init__(model, mdp)

    def calc_opt_policy(self, discount: int = 1, epsilon: float = 1e-5, max_iter: int = 100000, skip_check: bool = True,
                        verbose: bool = False) -> Tuple[BlockchainModel.Policy, float, int, np.array]:
        self.mdp.build_mdp(check_valid=not skip_check)
        p_mat, r_mat = self.get_pt_mdp()
        vi = mdptoolbox.PolicyIteration(p_mat, r_mat, discount=discount, epsilon=epsilon, max_iter=max_iter,
                                        skip_check=skip_check)
        if verbose:
            vi.setVerbose()
        vi.run()

        return vi.policy, vi.V[self.mdp.initial_state_index] / self.expected_horizon, vi.iter, vi.V

    def get_pt_mdp(self) -> Tuple[Union[np.array, List[spmatrix]], Union[np.array, List[spmatrix]]]:
        if self.use_sparse:
            return self.get_pt_mdp_sparse()
        else:
            return self.get_pt_mdp_dense()

    def get_pt_mdp_dense(self) -> Tuple[np.array, np.array]:
        p_mat = np.multiply(np.power(1 - 1 / self.expected_horizon, self.mdp.D.get_data()), self.mdp.P.get_data())
        p_mat[:, :, self.mdp.final_state_index] = 0
        return p_mat, self.mdp.R.get_data()

    def get_pt_mdp_sparse(self) -> Tuple[List[spmatrix], List[spmatrix]]:
        p_mats = self.mdp.P.get_data()
        r_mats = self.mdp.R.get_data()
        d_mats = self.mdp.D.get_data()

        for action in range(self.mdp.num_of_actions):
            row_indices, col_indices, transition_probabilities = find(p_mats[action])
            difficulty_contributions = np.power(1 - 1 / self.expected_horizon,
                                                d_mats[action][row_indices, col_indices].toarray().squeeze())
            pt_transition_probabilities = np.multiply(transition_probabilities, difficulty_contributions)

            # Because the value of the final state is 0
            pt_transition_probabilities[col_indices == self.mdp.final_state_index] = 0

            p_mats[action] = csr_matrix((pt_transition_probabilities, (row_indices, col_indices)),
                                        shape=(self.mdp.num_of_states, self.mdp.num_of_states))

            r_mats[action] = r_mats[action].tocsr(copy=True)

        return p_mats, r_mats
