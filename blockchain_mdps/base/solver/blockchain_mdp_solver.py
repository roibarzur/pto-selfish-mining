from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np

from ... import BlockchainModel
from ..blockchain_mdps.sparse_blockchain_mdp import BlockchainMDP


class BlockchainMDPSolver(ABC):
    def __init__(self, model: BlockchainModel, mdp: BlockchainMDP):
        self.model = model
        self.mdp = mdp

    @abstractmethod
    def calc_opt_policy(self, discount: int = 1, epsilon: float = 1e-5, max_iter: int = 100000, skip_check: bool = True,
                        verbose: bool = False) -> Tuple[BlockchainModel.Policy, float, int, np.array]:
        pass
