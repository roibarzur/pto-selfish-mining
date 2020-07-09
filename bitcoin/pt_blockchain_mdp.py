from sparse_blockchain_mdp import *

class RandomStopBlockchainMDP(SparseBlockchainMDP):
    """An MDP where each block agreed by all has a chance to end the process"""

    def __init__(self, alpha, gamma, max_fork, expected_horizon):
        super(RandomStopBlockchainMDP, self).__init__(alpha, gamma, max_fork)
        self.expected_horizon = expected_horizon
        self.block_end_prob = 1.0 / expected_horizon
        
    def get_actions(self):
        return ['Illegal', 'Adopt', 'Override', 'Match', 'Wait']

    def set_single_state_transitions(self, state_index):
        if state_index == self.final_state:
            for i in range(self.num_of_actions):
                self.P.set(i, state_index, self.final_state, 1)
        else:
            (a, h, fork) = self.index_to_state(state_index)
            
            #Illegal
            self.P.set(self.Action.Illegal, state_index, self.final_state, 1)
            self.R.set(self.Action.Illegal, state_index, self.final_state, self.error_penalty / 2)

            #Adopt
            if h > 0:
                attacker_block = self.state_to_index((1, 0, self.Fork.Irrelevant))
                self.P.set(self.Action.Adopt, state_index, attacker_block, self.alpha * (1 - self.block_end_prob) ** h)

                honest_block = self.state_to_index((0, 1, self.Fork.Irrelevant))
                self.P.set(self.Action.Adopt, state_index, honest_block, (1 - self.alpha) * (1 - self.block_end_prob) ** h)

                self.P.set(self.Action.Adopt, state_index, self.final_state, 1 - (1 - self.block_end_prob) ** h)
            else:
                self.P.set(self.Action.Adopt, state_index, self.final_state, 1)
                self.R.set(self.Action.Adopt, state_index, self.final_state, self.error_penalty)

            #Override
            if a > h:
                attacker_block = self.state_to_index((a - h, 0, self.Fork.Irrelevant))
                self.P.set(self.Action.Override, state_index, attacker_block, self.alpha * (1 - self.block_end_prob) ** (h + 1))
                self.R.set(self.Action.Override, state_index, attacker_block, h + 1)

                honest_block = self.state_to_index((a - h - 1, 1, self.Fork.Relevant))
                self.P.set(self.Action.Override, state_index, honest_block, (1 - self.alpha) * (1 - self.block_end_prob) ** (h + 1))
                self.R.set(self.Action.Override, state_index, honest_block, h + 1)

                self.P.set(self.Action.Override, state_index, self.final_state, 1 - (1 - self.block_end_prob) ** (h + 1))
            else:
                self.P.set(self.Action.Override, state_index, self.final_state, 1)
                self.R.set(self.Action.Override, state_index, self.final_state, self.error_penalty)
                
            #Match
            if a >= h and fork is self.Fork.Relevant and h > 0 and a < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active))
                self.P.set(self.Action.Match, state_index, attacker_block, self.alpha)
            
                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_support_block, self.gamma * (1 - self.alpha) * (1 - self.block_end_prob) ** h)
                self.R.set(self.Action.Match, state_index, honest_support_block, h)
            
                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
                
                self.P.set(self.Action.Match, state_index, self.final_state, self.gamma * (1 - self.alpha) * (1 - (1 - self.block_end_prob) ** h))
            else:
                self.P.set(self.Action.Match, state_index, self.final_state, 1)
                self.R.set(self.Action.Match, state_index, self.final_state, self.error_penalty)

            #Wait
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Irrelevant))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)
            
                honest_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_block, 1 - self.alpha)
            elif fork is self.Fork.Active and a >= h and h > 0 and a < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active))
                self.P.set(self.Action.Wait, state_index, attacker_block, self.alpha)
            
                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_support_block, self.gamma * (1 - self.alpha) * (1 - self.block_end_prob) ** h)
                self.R.set(self.Action.Wait, state_index, honest_support_block, h)
            
                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))

                self.P.set(self.Action.Wait, state_index, self.final_state, self.gamma * (1 - self.alpha) * (1 - (1 - self.block_end_prob) ** h))
            else:
                self.P.set(self.Action.Wait, state_index, self.final_state, 1)
                self.R.set(self.Action.Wait, state_index, self.final_state, self.error_penalty)
                
    def policy_reward_modifier(self, reward):
        return reward / self.expected_horizon

if (__name__ == '__main__'):
    print('bitcoin.random_stop_blockchain_mdp module test')

    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    mdp = RandomStopBlockchainMDP(0.45, 0.0, 75, 1000)
    print(mdp.num_of_states)
    mdp.calc_opt_policy()
    mdp.print_policy()
    mdp.test_policy(verbose = True)
