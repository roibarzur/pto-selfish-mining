from abc import ABC

from scipy.sparse import find

from .blockchain_mdp import *
from mdp_matrices.sparse_mdp_matrix import *


class SparseBlockchainMDP(BlockchainMDP, ABC):
    """A base class for an MDP which uses sparse matrices"""

    def init_matrix(self):
        return SparseMDPMatrix(self.num_of_actions, self.num_of_states)

    def get_pt_mdp(self, expected_horizon):
        p_mats = self.P.get_data()
        r_mats = self.R.get_data()
        d_mats = self.D.get_data()

        for action in range(self.num_of_actions):
            row_indices, col_indices, transition_probabilities = find(p_mats[action])
            difficulty_contributions = np.power(1 - 1 / expected_horizon,
                                                d_mats[action][row_indices, col_indices].toarray().squeeze())
            pt_transition_probabilities = np.multiply(transition_probabilities, difficulty_contributions)

            # Because the value of the final state is 0
            pt_transition_probabilities[col_indices == self.final_state] = 0

            p_mats[action] = csr_matrix((pt_transition_probabilities, (row_indices, col_indices)),
                                        shape=(self.num_of_states, self.num_of_states))

            r_mats[action] = r_mats[action].tocsr(copy=True)

        return p_mats, r_mats

    def policy_induced_chain_to_graph(self, policy_induced_chain):
        return nx.from_scipy_sparse_matrix(policy_induced_chain, create_using=nx.DiGraph())

    def test_state_transition(self, policy_induced_chain, state_index):
        (_, J, V) = find(policy_induced_chain[state_index, :])
        next_state_dist = V
        next_state_dist_acc = np.add.accumulate(next_state_dist)
        chosen_index = np.argmax(next_state_dist_acc > np.random.random())
        return J[chosen_index]

    def dot(self, a, b):
        return a.dot(b)

    def multiply(self, a, b):
        return a.multiply(b)
