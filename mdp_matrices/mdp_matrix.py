import numpy as np


class MDPMatrix:
    """Matrix for an MDP"""
    
    def __init__(self, num_of_actions, num_of_states):
        self.num_of_actions = num_of_actions
        self.num_of_states = num_of_states

        self.M = self._build_mat()

    def _build_mat(self):
        return np.zeros((self.num_of_actions, self.num_of_states, self.num_of_states))

    def get_val(self, action, from_state, to_state):
        return self.M[action, from_state, to_state]

    def get_induced(self, policy):
        induced = np.zeros((self.num_of_states, self.num_of_states))
        for i in range(self.num_of_states):
            induced[i, :] = self.M[policy[i], i, :]

        return induced

    def set(self, action, from_state, to_state, value):
        self.M[action, from_state, to_state] += value

    def reset(self, action, from_state, to_state):
        self.M[action, from_state, to_state] = 0

    def get_data(self):
        return self.M
