from scipy.sparse import csr_matrix, find
from blockchain_mdp import *
from sparse_mdp_matrix import *

class SparseBlockchainMDP(BlockchainMDP):
    """A base class for an MDP which uses sparse matrices"""

    def init_matrix(self):
        return SparseMDPMatrix(self.num_of_actions, self.num_of_states)

    def policy_induced_chain_to_graph(self, policy_induced_chain):
        return nx.from_scipy_sparse_matrix(policy_induced_chain, create_using=nx.DiGraph())

    def test_state_transition(self, policy_induced_chain, state_index):
        (_, J, V) = find(policy_induced_chain[state_index, :])
        next_state_dist = V
        next_state_dist_acc = np.add.accumulate(next_state_dist)
        chosen_indice = np.argmax(next_state_dist_acc > np.random.random())
        return J[chosen_indice]

    def dot(self, a, b):
        return a.dot(b)

    def multiply(self, a, b):
        return a.multiply(b)
