import sys

from base_blockchain_mdps.indexed_blockchain_mdp import *


class BitcoinMDP(IndexedBlockchainMDP):

    def __init__(self, alpha, gamma, max_fork):
        super(BitcoinMDP, self).__init__(alpha, gamma, max_fork)

    def get_actions(self):
        return ['Illegal', 'Adopt', 'Override', 'Match', 'Wait']

    def set_single_state_transitions(self, state_index):
        if state_index == self.final_state:
            for i in range(self.num_of_actions):
                self.P.set(i, state_index, self.final_state, 1)
        else:
            (a, h, fork) = self.index_to_state(state_index)

            # Illegal
            self.P.set(self.Action.Illegal, state_index, self.final_state, 1)
            self.R.set(self.Action.Illegal, state_index, self.final_state, self.error_penalty / 2)

            # Adopt
            if h > 0:
                attacker_block = self.state_to_index((1, 0, self.Fork.Irrelevant))
                self.P.set(self.Action.Adopt, state_index, attacker_block, self.alpha)
                self.D.set(self.Action.Adopt, state_index, attacker_block, h)

                honest_block = self.state_to_index((0, 1, self.Fork.Irrelevant))
                self.P.set(self.Action.Adopt, state_index, honest_block, (1 - self.alpha))
                self.D.set(self.Action.Adopt, state_index, honest_block, h)
            else:
                self.P.set(self.Action.Adopt, state_index, self.final_state, 1)
                self.R.set(self.Action.Adopt, state_index, self.final_state, self.error_penalty)

            # Override
            if a > h:
                attacker_block = self.state_to_index((a - h, 0, self.Fork.Irrelevant))
                self.P.set(self.Action.Override, state_index, attacker_block, self.alpha)
                self.R.set(self.Action.Override, state_index, attacker_block, h + 1)
                self.D.set(self.Action.Override, state_index, attacker_block, h + 1)

                honest_block = self.state_to_index((a - h - 1, 1, self.Fork.Relevant))
                self.P.set(self.Action.Override, state_index, honest_block, (1 - self.alpha))
                self.R.set(self.Action.Override, state_index, honest_block, h + 1)
                self.D.set(self.Action.Override, state_index, honest_block, h + 1)
            else:
                self.P.set(self.Action.Override, state_index, self.final_state, 1)
                self.R.set(self.Action.Override, state_index, self.final_state, self.error_penalty)

            # Match
            if 0 < h <= a < self.max_fork and fork is self.Fork.Relevant:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active))
                self.P.set(self.Action.Match, state_index, attacker_block, self.alpha)

                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Match, state_index, honest_support_block, h)
                self.D.set(self.Action.Match, state_index, honest_support_block, h)

                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Match, state_index, self.final_state, 1)
                self.R.set(self.Action.Match, state_index, self.final_state, self.error_penalty)

            # Wait
            # noinspection DuplicatedCode
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Irrelevant))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)

                honest_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_block, 1 - self.alpha)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)

                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Wait, state_index, honest_support_block, h)
                self.D.set(self.Action.Wait, state_index, honest_support_block, h)

                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Wait, state_index, self.final_state, 1)
                self.R.set(self.Action.Wait, state_index, self.final_state, self.error_penalty)

    def is_policy_honest(self, policy=None):
        if policy is None:
            policy = self.opt_policy

        return policy[self.state_to_index((0, 0, self.Fork.Irrelevant))] == self.Action.Wait \
            and policy[self.state_to_index((1, 0, self.Fork.Irrelevant))] == self.Action.Override \
            and policy[self.state_to_index((0, 1, self.Fork.Irrelevant))] == self.Action.Adopt \
            and policy[self.state_to_index((0, 1, self.Fork.Relevant))] == self.Action.Adopt

    def build_sm1_policy(self):
        policy = np.zeros(self.num_of_states, dtype=int)

        for i in range(1, self.num_of_states):
            (a, h, fork) = self.index_to_state(i)

            if h > a:
                action = self.Action.Adopt
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Override
            elif (h == 1 and a == 1) or fork is self.Fork.Relevant:
                action = self.Action.Match
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('bitcoin_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = BitcoinMDP(0.35, 0.5, 75)
    print(mdp.num_of_states)
    p = mdp.build_sm1_policy()
    mdp.print_policy(p)
    print(mdp.calc_policy_revenue(p))
