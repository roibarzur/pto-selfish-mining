from typing import List, Tuple

import mdptoolbox.mdp as mdptoolbox
import numpy as np
from scipy.sparse import spmatrix

from .blockchain_mdp_solver import BlockchainMDPSolver
from ..blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP
from ... import BlockchainModel


class OSMSolver(BlockchainMDPSolver):
    def __init__(self, model: BlockchainModel):
        super().__init__(model, SparseBlockchainMDP(model))

    def calc_opt_policy(self, discount: int = 1, epsilon: float = 1e-5, max_iter: int = 100000, skip_check: bool = True,
                        verbose: bool = False) -> Tuple[BlockchainModel.Policy, float, int, np.array]:
        self.mdp.build_mdp(check_valid=not skip_check)

        vi = None
        low = 0
        high = 1
        rho = 0.5

        iterations = 0

        while high - low >= epsilon / 8:
            rho = (low + high) / 2
            p_mat, r_mat = self.get_rho_mdp(rho)

            # noinspection PyTypeChecker
            vi = mdptoolbox.PolicyIteration(p_mat, r_mat, discount=discount, epsilon=epsilon, max_iter=max_iter,
                                            skip_check=skip_check)

            if verbose:
                vi.setVerbose()
            vi.run()

            iterations += vi.iter

            r = vi.V[self.mdp.initial_state_index]

            if r > 0:
                low = rho
            else:
                high = rho

        return vi.policy, rho, iterations, vi.V

    def get_rho_mdp(self, rho: float) -> Tuple[List[spmatrix], np.array]:
        p_mats = self.mdp.P.get_data()
        r_mats = self.mdp.R.get_data()
        d_mats = self.mdp.D.get_data()

        r_mat = np.zeros((self.mdp.num_of_states, self.mdp.num_of_actions))

        for action in range(self.mdp.num_of_actions):
            # Calculate the expected reward first
            reward_mat = (1 - rho) * r_mats[action] - rho * (d_mats[action] - r_mats[action])
            r_mat[:, action] = p_mats[action].multiply(reward_mat).sum(1).A.reshape(self.mdp.num_of_states)

            # Because the value of the final state is 0
            p_mats[action] = p_mats[action].tocsr(copy=True)
            p_mats[action][:, self.mdp.initial_state_index] = 0
            p_mats[action][:, self.mdp.final_state_index] = 0

        return p_mats, r_mat
