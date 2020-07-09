from indexed_blockchain_mdp import *
from ethereum.infinite_blockchain_mdp import *

class SectionCountingBlockchainMDP(IndexedBlockchainMDP):
    """An MDP with a known finite section horizon, known section count and uncle rewards"""

    def __init__(self, alpha, gamma=0.5, max_fork=20, num_of_sections=1, expected_section_length=10**5):
        self.num_of_sections = num_of_sections
        self._num_of_sections_b = self.num_of_sections
        self.expected_section_length = expected_section_length
        self.section_end_prob = 1 / expected_section_length
        self.uncle_rewards = [0, 7.0 / 8, 6.0 / 8, 5.0 / 8, 4.0 / 8, 3.0 / 8, 2.0 / 8]
        self.nephew_reward = 1.0 / 32
        self._uncle_dist_b = len(self.uncle_rewards)
        self._honest_uncles_b = 2 ** (self._uncle_dist_b - 1)
        super(SectionCountingBlockchainMDP, self).__init__(alpha, 0.5, max_fork)

    def calc_state_space_base_sizes(self):
        return [self._num_of_sections_b, self._max_fork_b, self._max_fork_b, self.num_of_fork_states, 2, self._honest_uncles_b, self._uncle_dist_b]
    
    def state_space_transform(self, state):
        (c, a, h, fork, au, hu, r) = state
        return (c, a, h, self.Fork(fork), au, hu, r)
    
    def state_to_index(self, state):
        if state != 0 and state[0] == self.num_of_sections:
            return self.final_state
        
        return super(SectionCountingBlockchainMDP, self).state_to_index(state)
    
    def get_actions(self):
        return ['Illegal', 'AdoptReveal', 'Forget', 'Override', 'Match', 'Wait', 'Reveal']
    
    def get_fork_states(self):
        return ['Relevant', 'Active']

    def build_MDP(self):
        super(SectionCountingBlockchainMDP, self).build_MDP()

        for action in list(self.Action):
            for state_index in range(self.num_of_states):
                self.P.reset(action, state_index, self.final_state)

    def set_single_state_transitions(self, state_index):
        if state_index == self.final_state:
            for i in range(self.num_of_actions):
                self.P.set(i, state_index, self.final_state, 1)
        else:
            (c, a, h, fork, au, hu, r) = self.index_to_state(state_index)
            
            #Illegal
            self.P.set(self.Action.Illegal, state_index, self.final_state, 1)
            self.R.set(self.Action.Illegal, state_index, self.final_state, self.error_penalty / 2)
            
            #AdoptReveal
            num_of_uncles_included = min(2 * h, bin(hu).count('1')) + int(a > 0)
            if c + int((h + num_of_uncles_included) / self.expected_section_length) + 1 <= self.num_of_sections and h > 0:
                sure_sections = int((h + num_of_uncles_included) / self.expected_section_length)
                rand_blocks = (h + num_of_uncles_included) % self.expected_section_length
                num_of_uncles_included = min(2 * h, bin(hu).count('1'))

                new_hu = ((int(bin(hu).replace('1', '0', 2 * h), 2)) << h) % self._honest_uncles_b

                r_included = bool(2 * h - int(au > 0) - bin(hu).count('1') > 0) and a > 0 and r > 0
                r_include_distance = max(r, 1 + int((int(au > 0) + bin(hu).count('1')) / 2))
                
                new_au = int(a > 0 and not r_included)
                if h > 0 and h < self._uncle_dist_b:
                    new_au_reward = new_au * self.uncle_rewards[h]
                else:
                    new_au_reward = 0

                if au > 0 and bin(hu).count('1') >= 2:
                    au_not_included_penalty = -1.0 / 8
                else:
                    au_not_included_penalty = 0

                attacker_block_1 = self.state_to_index((c + sure_sections, 1, 0, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, attacker_block_1, self.alpha * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.AdoptReveal, state_index, attacker_block_1, au_not_included_penalty + new_au_reward + int(r_included) * self.uncle_rewards[r_include_distance])

                attacker_block_2 = self.state_to_index((c + sure_sections + 1, 1, 0, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, attacker_block_2, self.alpha * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.AdoptReveal, state_index, attacker_block_2, au_not_included_penalty + new_au_reward + int(r_included) * self.uncle_rewards[r_include_distance])

                honest_block_1 = self.state_to_index((c + sure_sections, 0, 1, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, honest_block_1, (1 - self.alpha) * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.AdoptReveal, state_index, honest_block_1, au_not_included_penalty + new_au_reward + int(r_included) * self.uncle_rewards[r_include_distance])

                honest_block_2 = self.state_to_index((c + sure_sections + 1, 0, 1, self.Fork.Relevant, new_au, new_hu, 0))
                self.P.set(self.Action.AdoptReveal, state_index, honest_block_2, (1 - self.alpha) * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.AdoptReveal, state_index, honest_block_2, au_not_included_penalty + new_au_reward + int(r_included) * self.uncle_rewards[r_include_distance])
            else:
                self.P.set(self.Action.AdoptReveal, state_index, self.final_state, 1)
                self.R.set(self.Action.AdoptReveal, state_index, self.final_state, self.error_penalty)

            #Forget
            num_of_uncles_included = min(2 * h, bin(hu).count('1'))
            if c + int((h + num_of_uncles_included) / self.expected_section_length) + 1 <= self.num_of_sections and h > 0 and r == 0:
                sure_sections = int((h + num_of_uncles_included) / self.expected_section_length)
                rand_blocks = (h + num_of_uncles_included) % self.expected_section_length
                new_hu = ((int(bin(hu).replace('1', '0', 2 * h), 2)) << h) % self._honest_uncles_b

                if au > 0 and bin(hu).count('1') >= 2:
                    au_not_included_penalty = -1.0 / 8
                else:
                    au_not_included_penalty = 0

                attacker_block_1 = self.state_to_index((c + sure_sections, 1, 0, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, attacker_block_1, self.alpha * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Forget, state_index, attacker_block_1, au_not_included_penalty)

                attacker_block_2 = self.state_to_index((c + sure_sections + 1, 1, 0, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, attacker_block_2, self.alpha * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Forget, state_index, attacker_block_2, au_not_included_penalty)

                honest_block_1 = self.state_to_index((c + sure_sections, 0, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, honest_block_1, (1 - self.alpha) * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Forget, state_index, honest_block_1, au_not_included_penalty)

                honest_block_2 = self.state_to_index((c + sure_sections + 1, 0, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Forget, state_index, honest_block_2, (1 - self.alpha) * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Forget, state_index, honest_block_2, au_not_included_penalty)
            else:
                self.P.set(self.Action.Forget, state_index, self.final_state, 1)
                self.R.set(self.Action.Forget, state_index, self.final_state, self.error_penalty)

            #Override
            if a > h and c + int((h + 1) / self.expected_section_length) + 1 <= self.num_of_sections:
                sure_sections = int((h + 1) / self.expected_section_length)
                rand_blocks = (h + 1) % self.expected_section_length
                if h > 0 and h < self._uncle_dist_b:
                    new_hu = ((hu << (h + 1)) + 2 ** h) % self._honest_uncles_b
                else:
                    new_hu = (hu << (h + 1)) % self._honest_uncles_b

                attacker_block_1 = self.state_to_index((c + sure_sections, a - h, 0, self.Fork.Relevant, 0, new_hu , 0))
                self.P.set(self.Action.Override, state_index, attacker_block_1, self.alpha * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Override, state_index, attacker_block_1, h + 1 + au * self.nephew_reward)

                attacker_block_2 = self.state_to_index((c + sure_sections + 1, a - h, 0, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Override, state_index, attacker_block_2, self.alpha * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Override, state_index, attacker_block_2, h + 1 + au * self.nephew_reward)

                honest_block_1 = self.state_to_index((c + sure_sections, a - h - 1, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Override, state_index, honest_block_1, (1 - self.alpha) * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Override, state_index, honest_block_1, h + 1 + au * self.nephew_reward)

                honest_block_2 = self.state_to_index((c + sure_sections + 1, a - h - 1, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Override, state_index, honest_block_2, (1 - self.alpha) * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Override, state_index, honest_block_2, h + 1 + au * self.nephew_reward)
            else:
                self.P.set(self.Action.Override, state_index, self.final_state, 1)
                self.R.set(self.Action.Override, state_index, self.final_state, self.error_penalty)
                
            #Match
            if a >= h and fork is self.Fork.Relevant and h > 0 and a < self.max_fork and c + int(h / self.expected_section_length) + 1 <= self.num_of_sections:
                sure_sections = int(h / self.expected_section_length)
                rand_blocks = h % self.expected_section_length
                if h < self._uncle_dist_b:
                    new_hu = ((hu << h) + 2 ** (h - 1)) % self._honest_uncles_b
                else:
                    new_hu = (hu << h) % self._honest_uncles_b

                if r > 0:
                    new_r = r
                elif h > 0 and h < self._uncle_dist_b:
                    new_r = h
                else:
                    new_r = 0

                attacker_block = self.state_to_index((c, a + 1, h, self.Fork.Active, au, hu, new_r))
                self.P.set(self.Action.Match, state_index, attacker_block, self.alpha)
            
                honest_support_block_1 = self.state_to_index((c + sure_sections, a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Match, state_index, honest_support_block_1, self.gamma * (1 - self.alpha) * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Match, state_index, honest_support_block_1, h + au * self.nephew_reward)

                honest_support_block_2 = self.state_to_index((c + sure_sections + 1, a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Match, state_index, honest_support_block_2, self.gamma * (1 - self.alpha) * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Match, state_index, honest_support_block_2, h + au * self.nephew_reward)
            
                honest_adversary_block = self.state_to_index((c, a, h + 1, self.Fork.Relevant, au, hu, new_r))
                self.P.set(self.Action.Match, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Match, state_index, self.final_state, 1)
                self.R.set(self.Action.Match, state_index, self.final_state, self.error_penalty)

            #Wait
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = self.state_to_index((c, a + 1, h, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)
            
                honest_block = self.state_to_index((c, a, h + 1, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, honest_block, 1 - self.alpha)
            elif fork is self.Fork.Active and a >= h and h > 0 and a < self.max_fork and c + int(h / self.expected_section_length) + 1 <= self.num_of_sections:
                sure_sections = int(h / self.expected_section_length)
                rand_blocks = h % self.expected_section_length
                if h < self._uncle_dist_b:
                    new_hu = ((hu << h) + 2 ** (h - 1)) % self._honest_uncles_b
                else:
                    new_hu = (hu << h) % self._honest_uncles_b

                attacker_block = self.state_to_index((c, a + 1, h, self.Fork.Active, au, hu, r))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)
            
                honest_support_block_1 = self.state_to_index((c + sure_sections, a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Wait, state_index, honest_support_block_1, self.gamma * (1 - self.alpha) * (1 - self.section_end_prob) ** rand_blocks)
                self.R.set(self.Action.Wait, state_index, honest_support_block_1, h + au * self.nephew_reward)
            
                honest_support_block_2 = self.state_to_index((c + sure_sections + 1, a - h, 1, self.Fork.Relevant, 0, new_hu, 0))
                self.P.set(self.Action.Wait, state_index, honest_support_block_2, self.gamma * (1 - self.alpha) * (1 - (1 - self.section_end_prob) ** rand_blocks))
                self.R.set(self.Action.Wait, state_index, honest_support_block_2, h + au * self.nephew_reward)

                honest_adversary_block = self.state_to_index((c, a, h + 1, self.Fork.Relevant, au, hu, r))
                self.P.set(self.Action.Wait, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Wait, state_index, self.final_state, 1)
                self.R.set(self.Action.Wait, state_index, self.final_state, self.error_penalty)

            #Reveal
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork and h < self._uncle_dist_b and r == 0 and a > 0 and h > 1:
                attacker_block = self.state_to_index((c, a + 1, h, self.Fork.Relevant, au, hu, h))
                self.P.set(self.Action.Reveal, state_index, attacker_block, self.alpha)
            
                honest_block = self.state_to_index((c, a, h + 1, self.Fork.Relevant, au, hu, h))
                self.P.set(self.Action.Reveal, state_index, honest_block, 1 - self.alpha)
            else:
                self.P.set(self.Action.Reveal, state_index, self.final_state, 1)
                self.R.set(self.Action.Reveal, state_index, self.final_state, self.error_penalty)

    def translate_to_infinite_policy(self, policy = None, c = 0):
        if policy is None:
            policy = self.opt_policy

        infinite_mdp = InfiniteBlockchainMDP(self.alpha, self.max_fork)
        infinite_policy = np.zeros(infinite_mdp.num_of_states, dtype=int)
        
        for i in range(1, infinite_mdp.num_of_states):
            (a, h, fork, au, hu, r) = infinite_mdp.index_to_state(i)
            infinite_policy[i] = policy[self.state_to_index((c, a, h, fork, au, hu, r))]

        return tuple(infinite_policy)

if (__name__ == '__main__'):
    print('ethereum.section_counting_blockchain_mdp module test')

    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    mdp = SectionCountingBlockchainMDP(0.45, 2, 2, 100)
    print(mdp.num_of_states)
    mdp.calc_opt_policy()
    #mdp.print_policy()
