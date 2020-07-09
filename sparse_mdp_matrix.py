import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from mdp_matrix import *

class SparseMDPMatrix(MDPMatrix):
    """Sparse matrix for an MDP"""
    
    def __init__(self, num_of_actions, num_of_states):
        self.num_of_actions = num_of_actions
        self.num_of_states = num_of_states

        self.finalized = False

        self.M = [lil_matrix((self.num_of_states, self.num_of_states), dtype = np.float64) for i in range(self.num_of_actions)]

    def get_val(self, action, from_state, to_state):
        if not self.finalized:
            self.finalize()
        
        return self.M[action][from_state, to_state]

    def get_row(self, action, state):
        raise PendingDeprecationWarning
        if not self.finalized:
            self.finalize()

        return self.M[action][state, :]

    def get_row_as_array(self, action, state):
        raise PendingDeprecationWarning
        return self.get_row(action, state).toarray()

    def get_induced(self, policy):
        policy = np.array(policy)
        induced = csr_matrix((self.num_of_states, self.num_of_states))
        for action in range(self.num_of_actions):
            induced += self.M[action].multiply(np.expand_dims(policy == action, axis=1))

        return induced

    def set(self, action, from_state, to_state, value):
        if self.finalized:
            raise AssertionError

        self.M[action][from_state, to_state] += value

    def reset(self, action, from_state, to_state):
        if self.finalized:
            raise AssertionError

        self.M[action][from_state, to_state] = 0

    def finalize(self):
        self.M = [self.M[i].tocsr(copy=True) for i in range(self.num_of_actions)]

        self.finalized = True

    def to_raw_data(self):
        if not self.finalized:
            self.finalize()

        return self.M

    def dot(a, b):
        raise PendingDeprecationWarning
        return np.dot(a, b)