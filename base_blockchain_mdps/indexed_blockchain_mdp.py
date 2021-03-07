from .sparse_blockchain_mdp import *


class IndexedBlockchainMDP(SparseBlockchainMDP, ABC):
    def __init__(self, alpha, gamma, max_fork):
        super(IndexedBlockchainMDP, self).__init__(alpha, gamma, max_fork)
        self.base_sizes = self.calc_state_space_base_sizes()
        self.state_space_dim = len(self.base_sizes)

    def calc_state_space_base_sizes(self):
        return [self._max_fork_b, self._max_fork_b, self.num_of_fork_states]

    def state_space_transform(self, state):
        (a, h, fork) = state
        return a, h, self.Fork(fork)

    def calc_num_of_states(self):
        return np.prod(self.calc_state_space_base_sizes()) + 1

    def calc_initial_state(self):
        return self.state_to_index(self.state_space_transform((0,) * len(self.calc_state_space_base_sizes())))

    def state_to_index(self, state):
        try:
            base_sizes = self.base_sizes
            state_space_dim = self.state_space_dim
        except AttributeError:
            base_sizes = self.calc_state_space_base_sizes()
            state_space_dim = len(base_sizes)

        if state == 0:
            return self.final_state

        index = 0
        for i in reversed(range(state_space_dim)):
            if state[i] < 0 or state[i] >= base_sizes[i]:
                print(state)
                print(i)
                print(base_sizes[i])
                raise ValueError

            index += state[i]

            if i > 0:
                index *= base_sizes[i - 1]

        return index + 1

    def index_to_state(self, index):
        if index < 0 or index >= self.num_of_states:
            raise ValueError
        elif index == self.final_state:
            return 0
        else:
            index = index - 1
            state = []
            for i in range(self.state_space_dim):
                state.append(int(index % self.base_sizes[i]))
                index /= self.base_sizes[i]

            return self.state_space_transform(tuple(state))
