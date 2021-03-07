from scipy.sparse import csr_matrix, lil_matrix

from .mdp_matrix import *


class SparseMDPMatrix(MDPMatrix):
    """Sparse matrix for an MDP"""
    
    def __init__(self, num_of_actions, num_of_states):
        self.finalized = False
        super(SparseMDPMatrix, self).__init__(num_of_actions, num_of_states)

    def _build_mat(self):
        return [lil_matrix((self.num_of_states, self.num_of_states), dtype=np.float64) for _ in
                range(self.num_of_actions)]

    def get_val(self, action, from_state, to_state):
        return self.M[action][from_state, to_state]

    def get_induced(self, policy):
        policy = np.array(policy)
        induced = csr_matrix((self.num_of_states, self.num_of_states))
        for action in range(self.num_of_actions):
            induced += self.M[action].multiply(np.expand_dims(policy == action, axis=1))

        return induced

    def set(self, action, from_state, to_state, value):
        self.M[action][from_state, to_state] += value

    def reset(self, action, from_state, to_state):
        self.M[action][from_state, to_state] = 0

    def get_data(self):
        return self.M.copy()
