import sys

from base_blockchain_mdps.indexed_blockchain_mdp import *


class EthereumMDP(IndexedBlockchainMDP):
    def __init__(self, alpha, gamma=0.5, max_fork=20):
        self.uncle_rewards = [0, 7.0 / 8, 6.0 / 8, 5.0 / 8, 4.0 / 8, 3.0 / 8, 2.0 / 8]
        self.nephew_reward = 1.0 / 32
        self._uncle_dist_b = len(self.uncle_rewards)
        self._honest_uncles_b = 2 ** (self._uncle_dist_b - 1)
        super(EthereumMDP, self).__init__(alpha, gamma, max_fork)

    def calc_state_space_base_sizes(self):
        return [self._max_fork_b, self._max_fork_b, self.num_of_fork_states, 2, self._honest_uncles_b,
                self._uncle_dist_b]

    def state_space_transform(self, state):
        (a, h, fork, au, hu, r) = state
        return a, h, self.Fork(fork), au, hu, r

    def get_actions(self):
        return ['Illegal', 'AdoptReveal', 'Forget', 'Override', 'Match', 'Wait', 'Reveal']

    def get_fork_states(self):
        return ['Relevant', 'Active']

    def set_single_state_transitions(self, state_index):
        if state_index == self.final_state:
            for i in range(self.num_of_actions):
                self.P.set(i, state_index, self.final_state, 1)
        else:
            (a, h, fork, au, hu, r) = self.index_to_state(state_index)

            # Illegal
            self.P.set(self.Action.Illegal, state_index, self.final_state, 1)
            self.R.set(self.Action.Illegal, state_index, self.final_state, self.error_penalty / 2)

            # AdoptReveal
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

                attacker_block = self.state_to_index((1, 0, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, attacker_block, self.alpha)
                self.R.set(self.Action.AdoptReveal, state_index, attacker_block, au_not_included_penalty + new_au_reward
                           + int(r_included) * self.uncle_rewards[r_include_distance])
                self.D.set(self.Action.AdoptReveal, state_index, attacker_block, h + num_of_uncles_included)

                honest_block = self.state_to_index((0, 1, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, honest_block, 1 - self.alpha)
                self.R.set(self.Action.AdoptReveal, state_index, honest_block, au_not_included_penalty + new_au_reward +
                           int(r_included) * self.uncle_rewards[r_include_distance])
                self.D.set(self.Action.AdoptReveal, state_index, honest_block, h + num_of_uncles_included)
            else:
                self.P.set(self.Action.AdoptReveal, state_index, self.final_state, 1)
                self.R.set(self.Action.AdoptReveal, state_index, self.final_state, self.error_penalty)

            # Forget
            if h > 0 and r == 0:
                num_of_uncles_included = min(2 * h, bin(hu).count('1'))
                new_hu = ((int(bin(hu).replace('1', '0', 2 * h), 2)) << h) % self._honest_uncles_b

                if au > 0 and bin(hu).count('1') >= 2:
                    au_not_included_penalty = -1.0 / 8
                else:
                    au_not_included_penalty = 0

                attacker_block = self.state_to_index((1, 0, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, attacker_block, self.alpha)
                self.R.set(self.Action.Forget, state_index, attacker_block, au_not_included_penalty)
                self.D.set(self.Action.Forget, state_index, attacker_block, h + num_of_uncles_included)

                honest_block = self.state_to_index((0, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, honest_block, 1 - self.alpha)
                self.R.set(self.Action.Forget, state_index, honest_block, au_not_included_penalty)
                self.D.set(self.Action.Forget, state_index, honest_block, h + num_of_uncles_included)
            else:
                self.P.set(self.Action.Forget, state_index, self.final_state, 1)
                self.R.set(self.Action.Forget, state_index, self.final_state, self.error_penalty)

            # Override
            if a > h:
                if 0 < h < self._uncle_dist_b:
                    new_hu = ((hu << (h + 1)) + 2 ** h) % self._honest_uncles_b
                else:
                    new_hu = (hu << (h + 1)) % self._honest_uncles_b

                attacker_block = self.state_to_index((a - h, 0, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Override, state_index, attacker_block, self.alpha)
                self.R.set(self.Action.Override, state_index, attacker_block, h + 1 + au * self.nephew_reward)
                self.D.set(self.Action.Override, state_index, attacker_block, h + 1)

                honest_block = self.state_to_index((a - h - 1, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Override, state_index, honest_block, 1 - self.alpha)
                self.R.set(self.Action.Override, state_index, honest_block, h + 1 + au * self.nephew_reward)
                self.D.set(self.Action.Override, state_index, honest_block, h + 1)
            else:
                self.P.set(self.Action.Override, state_index, self.final_state, 1)
                self.R.set(self.Action.Override, state_index, self.final_state, self.error_penalty)

            # Match
            # noinspection DuplicatedCode
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

                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active, au, hu, new_r))
                self.P.set(self.Action.Match, state_index, attacker_block, self.alpha)

                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Match, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Match, state_index, honest_support_block, h + au * self.nephew_reward)
                self.D.set(self.Action.Match, state_index, honest_support_block, h)

                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant, au, hu, new_r))
                self.P.set(self.Action.Match, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Match, state_index, self.final_state, 1)
                self.R.set(self.Action.Match, state_index, self.final_state, self.error_penalty)

            # Wait
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)

                honest_block = self.state_to_index((a, h + 1, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, honest_block, 1 - self.alpha)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                if h < self._uncle_dist_b:
                    new_hu = ((hu << h) + 2 ** (h - 1)) % self._honest_uncles_b
                else:
                    new_hu = (hu << h) % self._honest_uncles_b

                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active, au, hu, r))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)

                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Wait, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Wait, state_index, honest_support_block, h + au * self.nephew_reward)
                self.D.set(self.Action.Wait, state_index, honest_support_block, h)

                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Wait, state_index, self.final_state, 1)
                self.R.set(self.Action.Wait, state_index, self.final_state, self.error_penalty)

            # Reveal
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork and h < self._uncle_dist_b \
                    and r == 0 and a > 0 and h > 1:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Relevant, au, hu, h))
                self.P.set(self.Action.Reveal, state_index, attacker_block, self.alpha)

                honest_block = self.state_to_index((a, h + 1, self.Fork.Relevant, au, hu, h))
                self.P.set(self.Action.Reveal, state_index, honest_block, 1 - self.alpha)
            else:
                self.P.set(self.Action.Reveal, state_index, self.final_state, 1)
                self.R.set(self.Action.Reveal, state_index, self.final_state, self.error_penalty)

    def build_test_policy(self):
        policy = np.zeros(self.num_of_states, dtype=int)

        for i in range(1, self.num_of_states):
            (a, h, fork, au, hu, r) = self.index_to_state(i)

            if h > a > 1 and r == 0:
                action = self.Action.AdoptReveal
            elif h > a:
                action = self.Action.Forget
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Override
            elif h == a and a >= 1:
                action = self.Action.Match
            elif r == 0:
                action = self.Action.Reveal
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)

    def build_honest_policy(self):
        policy = np.zeros(self.num_of_states, dtype=int)

        for i in range(1, self.num_of_states):
            (a, h, fork, au, hu, r) = self.index_to_state(i)

            if h > 0:
                action = self.Action.AdoptReveal
            elif a > 0:
                action = self.Action.Override
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)

    def print_policy(self, policy=None, reachable_states=None, print_size=8, pre_aux_obj=None, post_aux_obj=None):
        if policy is None:
            policy = self.opt_policy

        if reachable_states is None:
            reachable_states = self.find_reachable_states(policy)

        print_size = min(self.max_fork + 1, print_size)

        # For longer versions swap with comments
        for au in range(1):  # range(2):
            for hu in range(1):  # range(self._honest_uncles_b):
                for r in range(1):  # range(self._uncle_dist_b):
                    print(au, hu, r)
                    super(EthereumMDP, self).print_policy(policy, reachable_states, print_size,
                                                          post_aux_obj=(au, hu, r))


if __name__ == '__main__':
    print('ethereum_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = EthereumMDP(0.257, max_fork=10)
    print(mdp.num_of_states)
    p = mdp.build_test_policy()
    mdp.print_policy(p)
    print(mdp.calc_policy_revenue(p))
