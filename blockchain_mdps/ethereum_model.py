import sys
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class EthereumModel(BlockchainModel):
    def __init__(self, alpha: float, max_fork: int):
        self.alpha = alpha
        self.max_fork = max_fork

        self.gamma = 0.5
        self.uncle_rewards = [0, 7.0 / 8, 6.0 / 8, 5.0 / 8, 4.0 / 8, 3.0 / 8, 2.0 / 8]
        self.nephew_reward = 1.0 / 32
        self._uncle_dist_b = len(self.uncle_rewards)
        self._honest_uncles_b = 2 ** (self._uncle_dist_b - 1)

        self.Fork = self.create_int_enum('Fork', ['Relevant', 'Active'])
        self.Action = self.create_int_enum('Action',
                                           ['Illegal', 'AdoptReveal', 'Forget', 'Override', 'Match', 'Wait', 'Reveal'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork)

    def get_state_space(self) -> Space:
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork, 2,
                                                         self._honest_uncles_b, self._uncle_dist_b)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Relevant, 0, 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Relevant, 0, 0, 0

    # noinspection DuplicatedCode
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, au, hu, r = state

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action is self.Action.AdoptReveal:
            if h > 0:
                num_of_uncles_included = min(2 * h, bin(hu).count('1')) + int(a > 0)
                new_hu = ((int(bin(hu).replace('1', '0', 2 * h), 2)) << h) % self._honest_uncles_b

                r_included = bool(2 * h - int(au > 0) - bin(hu).count('1') > 0) and a > 0 and r > 0
                r_include_distance = max(r, 1 + int((int(au > 0) + bin(hu).count('1')) / 2))

                new_au = int(a > 0 and not r_included)
                if 0 < h < self._uncle_dist_b:
                    new_au_reward = new_au * self.uncle_rewards[h]
                else:
                    new_au_reward = 0

                if au > 0 and bin(hu).count('1') >= 2:
                    au_not_included_penalty = -1.0 / 8
                else:
                    au_not_included_penalty = 0

                attacker_block = 1, 0, self.Fork.Relevant, new_au, new_hu, 0
                transitions.add(attacker_block, probability=self.alpha,
                                reward=au_not_included_penalty + new_au_reward
                                       + int(r_included) * self.uncle_rewards[r_include_distance],
                                difficulty_contribution=h + num_of_uncles_included)

                honest_block = 0, 1, self.Fork.Relevant, new_au, new_hu, 0
                transitions.add(honest_block, probability=1 - self.alpha,
                                reward=au_not_included_penalty + new_au_reward
                                       + int(r_included) * self.uncle_rewards[r_include_distance],
                                difficulty_contribution=h + num_of_uncles_included)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Forget:
            if h > 0 and r == 0:
                num_of_uncles_included = min(2 * h, bin(hu).count('1'))
                new_hu = ((int(bin(hu).replace('1', '0', 2 * h), 2)) << h) % self._honest_uncles_b

                if au > 0 and bin(hu).count('1') >= 2:
                    au_not_included_penalty = -1.0 / 8
                else:
                    au_not_included_penalty = 0

                attacker_block = 1, 0, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(attacker_block, probability=self.alpha, reward=au_not_included_penalty,
                                difficulty_contribution=h + num_of_uncles_included)

                honest_block = 0, 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_block, probability=1 - self.alpha, reward=au_not_included_penalty,
                                difficulty_contribution=h + num_of_uncles_included)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Override:
            if a > h:
                if 0 < h < self._uncle_dist_b:
                    new_hu = ((hu << (h + 1)) + 2 ** h) % self._honest_uncles_b
                else:
                    new_hu = (hu << (h + 1)) % self._honest_uncles_b

                attacker_block = a - h, 0, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(attacker_block, probability=self.alpha, reward=h + 1 + au * self.nephew_reward,
                                difficulty_contribution=h + 1)

                honest_block = a - h - 1, 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_block, probability=1 - self.alpha, reward=h + 1 + au * self.nephew_reward,
                                difficulty_contribution=h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Match:
            if 0 < h <= a < self.max_fork and fork is self.Fork.Relevant:
                if h < self._uncle_dist_b:
                    new_hu = ((hu << h) + 2 ** (h - 1)) % self._honest_uncles_b
                else:
                    new_hu = (hu << h) % self._honest_uncles_b

                if r > 0:
                    new_r = r
                elif h < self._uncle_dist_b:
                    new_r = h
                else:
                    new_r = 0

                attacker_block = a + 1, h, self.Fork.Active, au, hu, new_r
                transitions.add(attacker_block, probability=self.alpha)

                honest_support_block = a - h, 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_support_block, probability=self.gamma * (1 - self.alpha),
                                reward=h + au * self.nephew_reward, difficulty_contribution=h)

                honest_adversary_block = a, h + 1, self.Fork.Relevant, au, hu, new_r
                transitions.add(honest_adversary_block, probability=(1 - self.gamma) * (1 - self.alpha))
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = a + 1, h, self.Fork.Relevant, au, hu, r
                transitions.add(attacker_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant, au, hu, r
                transitions.add(honest_block, probability=1 - self.alpha)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                if h < self._uncle_dist_b:
                    new_hu = ((hu << h) + 2 ** (h - 1)) % self._honest_uncles_b
                else:
                    new_hu = (hu << h) % self._honest_uncles_b

                attacker_block = a + 1, h, self.Fork.Active, au, hu, r
                transitions.add(attacker_block, probability=self.alpha)

                honest_support_block = a - h, 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_support_block, probability=self.gamma * (1 - self.alpha),
                                reward=h + au * self.nephew_reward, difficulty_contribution=h)

                honest_adversary_block = a, h + 1, self.Fork.Relevant, au, hu, r
                transitions.add(honest_adversary_block, probability=(1 - self.gamma) * (1 - self.alpha))
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Reveal:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork and h < self._uncle_dist_b \
                    and r == 0 and a > 0 and h > 1:
                attacker_block = a + 1, h, self.Fork.Relevant, au, hu, h
                transitions.add(attacker_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant, au, hu, h
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def build_test_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(1, self.state_space.size):
            a, h, fork, au, hu, r = self.state_space.index_to_element(i)

            if h > a > 1 and r == 0:
                action = self.Action.AdoptReveal
            elif h > a:
                action = self.Action.Forget
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Override
            elif h == a and a >= 1:
                action = self.Action.Match
            elif a > 0 and r == 0:
                action = self.Action.Reveal
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(1, self.state_space.size):
            a, h, fork, au, hu, r = self.state_space.index_to_element(i)

            if h > 0:
                action = self.Action.AdoptReveal
            elif a > 0:
                action = self.Action.Override
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('ethereum_mdp module test')
    from blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP

    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = EthereumModel(0.35, max_fork=5)
    print(mdp.state_space.size)
    p = mdp.build_honest_policy()

    solver = SparseBlockchainMDP(mdp)
    mdp.print_policy(p, solver.find_reachable_states(p))
    print(solver.calc_policy_revenue(p))
