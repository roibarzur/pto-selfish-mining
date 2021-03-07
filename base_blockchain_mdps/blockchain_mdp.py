from enum import IntEnum

import mdptoolbox.mdp as mdptoolbox
import networkx as nx
from scipy import stats
from scipy.sparse.linalg import eigs

from mdp_matrices.mdp_matrix import *


class BlockchainMDP:
    def __init__(self, alpha, gamma, max_fork):
        self.alpha = alpha
        self.gamma = gamma

        self.max_fork = max_fork
        self._max_fork_b = self.max_fork + 1

        self.Fork = self.create_int_enum('Fork', self.get_fork_states())
        self.num_of_fork_states = len(list(self.Fork))

        self.Action = self.create_int_enum('Action', self.get_actions())
        self.num_of_actions = len(list(self.Action))

        self.num_of_states = self.calc_num_of_states()

        self.error_penalty = -100000 * self.max_fork

        self.initial_state = self.calc_initial_state()
        self.final_state = 0

        self.P = self.init_matrix()
        self.R = self.init_matrix()
        self.D = self.init_matrix()

        self._built = False
        self.opt_policy = None

        self._cumulative_d = 0

    def init_matrix(self):
        return MDPMatrix(self.num_of_actions, self.num_of_states)

    @staticmethod
    def create_int_enum(enum_name, names):
        d = zip(names, range(len(names)))
        return IntEnum(enum_name, d)

    def get_actions(self):
        return ['Illegal', 'Adopt', 'Override', 'Match', 'Wait']

    def get_fork_states(self):
        return ['Irrelevant', 'Relevant', 'Active']

    def calc_num_of_states(self):
        return self.num_of_fork_states * self._max_fork_b ** 2 + 1

    def calc_initial_state(self):
        return self.state_to_index((0, 0, self.Fork.Irrelevant))

    def state_to_index(self, state):
        if state == 0:
            return self.final_state

        (a, h, fork) = state
        if a > self.max_fork or a < 0 or h > self.max_fork or h < 0 or not isinstance(fork, self.Fork):
            raise ValueError
        else:
            return self._max_fork_b ** 2 * fork + self._max_fork_b * h + a + 1

    def index_to_state(self, index):
        if index < 0 or index >= self.num_of_states:
            raise ValueError
        elif index == self.final_state:
            return 0
        else:
            return (int((index - 1) % self._max_fork_b),
                    int(((index - 1) / self._max_fork_b) % self._max_fork_b),
                    self.Fork(int((index - 1) / self._max_fork_b ** 2)))

    def print_states(self):
        for i in range(self.num_of_states):
            state = self.index_to_state(i)
            print(self.state_to_index(state), state)

    def build_mdp(self):
        if self._built:
            return

        for i in range(self.num_of_states):
            self.set_single_state_transitions(i)

        self._built = True

    def set_single_state_transitions(self, state_index):
        raise NotImplementedError

    def get_pt_mdp(self, expected_horizon):
        p_mat = np.multiply(np.power(1 - 1 / expected_horizon, self.D.get_data()), self.P.get_data())
        p_mat[:, :, self.final_state] = 0
        return p_mat, self.R.get_data()

    def calc_opt_policy(self, expected_horizon=10 ** 5, discount=1, epsilon=1e-5, max_iter=100000, skip_check=True,
                        verbose=False):
        self.build_mdp()

        p_mat, r_mat = self.get_pt_mdp(expected_horizon)

        vi = mdptoolbox.PolicyIteration(p_mat, r_mat, discount=discount, epsilon=epsilon, max_iter=max_iter,
                                        skip_check=skip_check)
        if verbose:
            vi.setVerbose()
        vi.run()

        self.opt_policy = vi.policy

        return self.opt_policy, vi.V[self.initial_state] / expected_horizon, vi.iter

    def get_policy_induced_chain(self, policy=None):
        if policy is None:
            policy = self.opt_policy

        self.build_mdp()

        return self.P.get_induced(policy)

    def policy_induced_chain_to_graph(self, policy_induced_chain):
        return nx.from_numpy_matrix(policy_induced_chain, create_using=nx.DiGraph())

    def find_reachable_states(self, policy=None):
        if policy is None:
            policy = self.opt_policy

        policy_induced_chain = self.get_policy_induced_chain(policy)
        reachable_states = np.zeros(self.num_of_states, dtype=bool)
        g = self.policy_induced_chain_to_graph(policy_induced_chain)
        for suc in nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(g, self.initial_state):
            reachable_states[suc] = True

        reachable_states[self.initial_state] = True

        return reachable_states

    def print_policy(self, policy=None, reachable_states=None, print_size=8, pre_aux_obj=None, post_aux_obj=None):
        if policy is None:
            policy = self.opt_policy

        if reachable_states is None:
            reachable_states = self.find_reachable_states(policy)

        print_size = min(self.max_fork + 1, print_size)

        policy_table = np.zeros((print_size, print_size), dtype=object)

        for a in range(print_size):
            for h in range(print_size):
                s = ""
                for f in range(self.num_of_fork_states):
                    state = (a, h, self.Fork(f))
                    if pre_aux_obj is not None:
                        state = pre_aux_obj + state
                    if post_aux_obj is not None:
                        state = state + post_aux_obj
                    state_index = self.state_to_index(state)
                    action = self.Action(policy[state_index])
                    if not reachable_states[state_index]:
                        ch = '*'
                    elif action is self.Action.Illegal:
                        ch = '-'
                    else:
                        ch = action.name[0].lower()
                    s += ch

                policy_table[a, h] = s

        print(policy_table)

    def test_policy_start_aux(self):
        self._cumulative_d = 0

    def test_policy_step_aux(self, action, state_index, next_state_index):
        self._cumulative_d += self.D.get_val(action, state_index, next_state_index)

    def policy_reward_modifier(self, reward):
        return reward / self._cumulative_d

    def get_policy_induced_chain_state_reward(self, policy, policy_induced_chain):
        state_reward = np.array(self.multiply(self.R.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()
        blocks_advanced = np.array(
            self.multiply(self.D.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()

        return state_reward, blocks_advanced

    def calc_policy_revenue(self, policy=None):
        if policy is None:
            policy = self.opt_policy

        policy_induced_chain = self.get_policy_induced_chain(policy)

        w, v = eigs(np.transpose(policy_induced_chain), k=1, which='LR')
        steady_prob = np.real_if_close(v).flatten()
        steady_prob = steady_prob / np.sum(steady_prob)
        print(steady_prob.shape)

        state_reward, blocks_advanced = self.get_policy_induced_chain_state_reward(policy, policy_induced_chain)

        return self.dot(state_reward, steady_prob) / self.dot(blocks_advanced, steady_prob)

    def test_policy(self, policy=None, episode_length=10000, times=1000, confidence=0.99, verbose=False):
        if policy is None:
            policy = self.opt_policy

        policy_induced_chain = self.get_policy_induced_chain(policy)

        rewards = np.zeros(times, dtype=np.float64)
        steps = np.zeros(times, dtype=int)
        for i in range(times):
            if verbose and i % (times / 20) == 0:
                print('{0}%'.format(int(i / times * 100)))

            state_index = self.initial_state
            reward = 0

            self.test_policy_start_aux()

            actual_steps = 0

            for step in range(episode_length):
                actual_steps += 1
                if state_index == self.final_state:
                    break

                state = self.index_to_state(state_index)
                action = self.Action(policy[state_index])

                next_state_index = self.test_state_transition(policy_induced_chain, state_index)

                step_reward = self.R.get_val(action, state_index, next_state_index)
                if step_reward < 0:
                    print('Error', state, action)

                self.test_policy_step_aux(action, state_index, next_state_index)

                reward += step_reward
                state_index = next_state_index

            rewards[i] = self.policy_reward_modifier(reward)
            steps[i] = actual_steps

        std_err = stats.sem(rewards)
        h = std_err * stats.t.ppf((1 + confidence) / 2, times - 1)
        print('Average reward: {:0.5f}\u00B1{:0.5f} ({:0.5f} - {:0.5f})'.format(rewards.mean(), h, rewards.mean() - h,
                                                                                rewards.mean() + h))
        print('Average length ', int(steps.mean()))

    def test_state_transition(self, policy_induced_chain, state_index):
        next_state_dist = policy_induced_chain[state_index, :]
        next_state_dist_acc = np.add.accumulate(next_state_dist)
        return np.argmax(next_state_dist_acc > np.random.random())

    def dot(self, a, b):
        return np.dot(a, b)

    def multiply(self, a, b):
        return np.multiply(a, b)


if __name__ == '__main__':
    print('blockchain_mdp module test')

    mdp = BlockchainMDP(0.35, 0.5, 3)
    mdp.print_states()
