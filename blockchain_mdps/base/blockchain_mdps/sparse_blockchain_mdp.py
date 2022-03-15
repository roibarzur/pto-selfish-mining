from abc import ABC

import numpy as np
import networkx as nx
from scipy.sparse import find, spmatrix
from scipy.sparse.linalg import eigs

from blockchain_mdps.base.blockchain_model import BlockchainModel
from blockchain_mdps.base.blockchain_mdps.mdp_matrices.sparse_mdp_matrix import SparseMDPMatrix
from .blockchain_mdp import BlockchainMDP


class SparseBlockchainMDP(BlockchainMDP, ABC):
    """A base class for an MDP which uses sparse matrices"""

    def init_matrix(self) -> SparseMDPMatrix:
        return SparseMDPMatrix(self.num_of_actions, self.num_of_states)

    @staticmethod
    def policy_induced_chain_to_graph(policy_induced_chain: spmatrix) -> nx.DiGraph:
        return nx.from_scipy_sparse_matrix(policy_induced_chain, create_using=nx.DiGraph())

    @staticmethod
    def calc_steady_distribution(policy_induced_chain: spmatrix) -> np.array:
        _, v = eigs(policy_induced_chain.T, k=1, which='LR')
        steady_prob = np.real_if_close(v).flatten()
        steady_prob = steady_prob / np.sum(steady_prob)
        return steady_prob

    @staticmethod
    def dot(a: spmatrix, b: spmatrix) -> spmatrix:
        return a.dot(b)

    @staticmethod
    def multiply(a: spmatrix, b: spmatrix) -> spmatrix:
        return a.multiply(b)

    def test_state_transition(self, policy_induced_chain: spmatrix, state_index: int) -> BlockchainModel.State:
        (_, J, V) = find(policy_induced_chain[state_index, :])
        next_state_dist = V
        next_state_dist_acc = np.add.accumulate(next_state_dist)
        chosen_index = np.argmax(next_state_dist_acc > np.random.random())
        next_state_index = J[chosen_index]
        return self.model.state_space.index_to_element(next_state_index)
