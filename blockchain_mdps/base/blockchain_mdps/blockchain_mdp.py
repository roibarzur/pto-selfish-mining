import math
from typing import Optional, Tuple, Dict

import networkx as nx
import numpy as np
from scipy import stats
from scipy.linalg import null_space

from blockchain_mdps.base.blockchain_mdps.mdp_matrices.mdp_matrix import MDPMatrix
from blockchain_mdps.base.blockchain_model import BlockchainModel


class BlockchainMDP:
    def __init__(self, model: BlockchainModel):
        self.model = model
        self.num_of_states = self.model.state_space.size
        self.num_of_actions = self.model.action_space.size
        self.initial_state_index = self.model.state_space.element_to_index(self.model.initial_state)
        self.final_state_index = self.model.state_space.element_to_index(self.model.final_state)

        self.P: Optional[MDPMatrix] = None
        self.R: Optional[MDPMatrix] = None
        self.D: Optional[MDPMatrix] = None

        self._built = False

    def init_matrix(self) -> MDPMatrix:
        return MDPMatrix(self.num_of_actions, self.num_of_states)

    def build_mdp(self, check_valid: bool = False) -> None:
        if self._built:
            return

        self.P = self.init_matrix()
        self.R = self.init_matrix()
        self.D = self.init_matrix()

        for state_index in range(self.num_of_states):
            for action_index in range(self.num_of_actions):
                transition_values = self.model.get_state_transitions(
                    self.model.state_space.index_to_element(state_index),
                    self.model.action_space.index_to_element(action_index), check_valid=check_valid)

                if check_valid and not math.isclose(sum(transition_values.probabilities.values()), 1, abs_tol=1e-5):
                    raise AssertionError('Probabilities don\'t sum to 1')

                if check_valid and any(d < 0 for d in transition_values.difficulty_contributions.values()):
                    raise AssertionError('Negative difficulty contribution')

                self.P.set_batch(action_index, state_index,
                                 self.translate_dict_keys_from_tuple_to_int(transition_values.probabilities))
                self.R.set_batch(action_index, state_index,
                                 self.translate_dict_keys_from_tuple_to_int(transition_values.rewards))
                self.D.set_batch(action_index, state_index,
                                 self.translate_dict_keys_from_tuple_to_int(transition_values.difficulty_contributions))

        self._built = True

    def translate_dict_keys_from_tuple_to_int(self, dictionary: Dict[tuple, float]) -> Dict[int, float]:
        return {self.model.state_space.element_to_index(elem): value for elem, value in dictionary.items()}

    def get_policy_induced_chain(self, policy: BlockchainModel.Policy) -> np.array:
        self.build_mdp()

        return self.P.get_induced(policy)

    @staticmethod
    def policy_induced_chain_to_graph(policy_induced_chain: np.array) -> nx.DiGraph:
        return nx.from_numpy_matrix(policy_induced_chain, create_using=nx.DiGraph())

    def find_reachable_states(self, policy: BlockchainModel.Policy) -> np.array:
        policy_induced_chain = self.get_policy_induced_chain(policy)
        reachable_states = np.zeros(self.num_of_states, dtype=bool)
        g = self.policy_induced_chain_to_graph(policy_induced_chain)
        for suc in nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(g, self.initial_state_index):
            reachable_states[suc] = True

        reachable_states[self.initial_state_index] = True

        return reachable_states

    def get_policy_induced_chain_state_reward(self, policy: BlockchainModel.Policy, policy_induced_chain: np.array
                                              ) -> Tuple[np.array, np.array]:
        state_reward = np.array(self.multiply(self.R.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()
        blocks_advanced = np.array(
            self.multiply(self.D.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()

        return state_reward, blocks_advanced

    def calc_policy_revenue(self, policy: BlockchainModel.Policy) -> float:
        policy_induced_chain = self.get_policy_induced_chain(policy)
        policy_induced_chain[self.final_state_index, self.final_state_index] = 0

        steady_prob = self.calc_steady_distribution(policy_induced_chain)

        state_reward, blocks_advanced = self.get_policy_induced_chain_state_reward(policy, policy_induced_chain)

        return self.dot(state_reward, steady_prob) / self.dot(blocks_advanced, steady_prob)

    @staticmethod
    def calc_steady_distribution(policy_induced_chain: np.array) -> np.array:
        v = null_space(np.eye(policy_induced_chain.shape[0]) - np.transpose(policy_induced_chain)).flatten()
        steady_prob = np.real_if_close(v)
        steady_prob = steady_prob / np.sum(steady_prob)
        return steady_prob

    @staticmethod
    def dot(a: np.array, b: np.array) -> np.array:
        return np.dot(a, b)

    @staticmethod
    def multiply(a: np.array, b: np.array) -> np.array:
        return np.multiply(a, b)

    def test_state_transition(self, policy_induced_chain: np.array, state_index: int) -> BlockchainModel.State:
        next_state_dist = policy_induced_chain[state_index, :]
        next_state_dist_acc = np.add.accumulate(next_state_dist)
        next_state_index = np.argmax(next_state_dist_acc > np.random.random()).item()
        return self.model.state_space.index_to_element(next_state_index)

    def test_policy(self, policy: BlockchainModel.Policy, episode_length: int = 10000, times: int = 1000,
                    confidence: float = 0.99, verbose: bool = False) -> None:

        policy_induced_chain = self.get_policy_induced_chain(policy)

        rewards = np.zeros(times, dtype=np.float64)
        steps = np.zeros(times, dtype=int)
        for i in range(times):
            if verbose and i % (times / 20) == 0:
                print('{0}%'.format(int(i / times * 100)))

            state = self.model.initial_state
            reward = 0
            difficulty_contribution = 0

            actual_steps = 0

            for step in range(episode_length):
                actual_steps += 1
                if state == self.model.final_state:
                    break

                state_index = self.model.state_space.element_to_index(state)
                action = self.model.action_space.index_to_element(policy[state_index])

                next_state = self.test_state_transition(policy_induced_chain, state_index)
                next_state_index = self.model.state_space.element_to_index(next_state)

                step_reward = self.R.get_val(action, state_index, next_state_index)
                step_difficulty_contribution = self.D.get_val(action, state_index, next_state_index)

                reward += step_reward
                difficulty_contribution += step_difficulty_contribution
                state = next_state

            rewards[i] = reward / difficulty_contribution
            steps[i] = actual_steps

        std_err = stats.sem(rewards)
        h = std_err * stats.t.ppf((1 + confidence) / 2, times - 1)
        print('Average reward: {:0.5f}\u00B1{:0.5f} ({:0.5f} - {:0.5f})'.format(rewards.mean(), h, rewards.mean() - h,
                                                                                rewards.mean() + h))
        print('Average length ', int(steps.mean()))
