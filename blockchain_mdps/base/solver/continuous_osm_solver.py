from typing import Tuple

import mdptoolbox.mdp as mdptoolbox
import numpy as np

from .osm_solver import OSMSolver
from ... import BlockchainModel


class ContinuousOSMSolver(OSMSolver):
    def calc_opt_policy(self, discount: int = 1, epsilon: float = 1e-5, max_iter: int = 100000, skip_check: bool = True,
                        verbose: bool = False) -> Tuple[BlockchainModel.Policy, float, int, np.array]:
        self.mdp.build_mdp(check_valid=not skip_check)

        vi = None
        policy = None
        low = 0
        high = 1
        rho = 0.5

        iterations = 0

        while high - low >= epsilon / 8:
            rho = (low + high) / 2
            p_mat, r_mat = self.get_rho_mdp(rho)

            # noinspection PyTypeChecker
            vi = mdptoolbox.PolicyIteration(p_mat, r_mat, discount=discount, epsilon=epsilon, max_iter=max_iter,
                                            skip_check=skip_check, policy0=policy)

            if verbose:
                vi.setVerbose()
            vi.run()

            iterations += vi.iter
            policy = vi.policy

            r = vi.V[self.mdp.initial_state_index]

            if r > 0:
                low = rho
            else:
                high = rho

        return vi.policy, rho, iterations, vi.V
