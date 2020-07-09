from sparse_blockchain_mdp import *
from indexed_blockchain_mdp import *
from scipy.sparse.linalg import eigs

class InfiniteBlockchainMDP(IndexedBlockchainMDP):
    """An infinte MDP for policy evaluation"""

    def __init__(self, alpha, gamma, max_fork):
        super(InfiniteBlockchainMDP, self).__init__(alpha, gamma, max_fork)
        self.R2 = self.init_matrix()

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
                self.P.set(self.Action.Adopt, state_index, attacker_block, self.alpha)
                self.R2.set(self.Action.Adopt, state_index, attacker_block, h)

                honest_block = self.state_to_index((0, 1, self.Fork.Irrelevant))
                self.P.set(self.Action.Adopt, state_index, honest_block, (1 - self.alpha))
                self.R2.set(self.Action.Adopt, state_index, honest_block, h)
            else:
                self.P.set(self.Action.Adopt, state_index, self.final_state, 1)
                self.R.set(self.Action.Adopt, state_index, self.final_state, self.error_penalty)

            #Override
            if a > h:
                attacker_block = self.state_to_index((a - h, 0, self.Fork.Irrelevant))
                self.P.set(self.Action.Override, state_index, attacker_block, self.alpha)
                self.R.set(self.Action.Override, state_index, attacker_block, h + 1)

                honest_block = self.state_to_index((a - h - 1, 1, self.Fork.Relevant))
                self.P.set(self.Action.Override, state_index, honest_block, (1 - self.alpha))
                self.R.set(self.Action.Override, state_index, honest_block, h + 1)
            else:
                self.P.set(self.Action.Override, state_index, self.final_state, 1)
                self.R.set(self.Action.Override, state_index, self.final_state, self.error_penalty)
                
            #Match
            if a >= h and fork is self.Fork.Relevant and h > 0 and a < self.max_fork:
                attacker_block = self.state_to_index((a + 1, h, self.Fork.Active))
                self.P.set(self.Action.Match, state_index, attacker_block, self.alpha)
            
                honest_support_block = self.state_to_index((a - h, 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Match, state_index, honest_support_block, h)
            
                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Match, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
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
                self.P.set(self.Action.Wait, state_index, honest_support_block, self.gamma * (1 - self.alpha))
                self.R.set(self.Action.Wait, state_index, honest_support_block, h)
            
                honest_adversary_block = self.state_to_index((a, h + 1, self.Fork.Relevant))
                self.P.set(self.Action.Wait, state_index, honest_adversary_block, (1 - self.gamma) * (1 - self.alpha))
            else:
                self.P.set(self.Action.Wait, state_index, self.final_state, 1)
                self.R.set(self.Action.Wait, state_index, self.final_state, self.error_penalty)

    def test_policy_start_aux(self):
        self._honest_reward = 0

    def test_policy_step_aux(self, action, state_index, next_state_index):
        self._honest_reward += self.R2.get_val(action, state_index, next_state_index)
                
    def policy_reward_modifier(self, reward):
        return reward / (reward + self._honest_reward)

    def get_policy_induced_chain_state_reward(self, policy, policy_induced_chain):
        state_reward = np.array(self.multiply(self.R.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()
        state_honest_reward = np.array(self.multiply(self.R2.get_induced(policy), policy_induced_chain).sum(axis=1)).flatten()

        return state_reward, state_honest_reward

    def is_policy_honest(self, policy = None):
        if policy is None:
            policy = self.opt_policy

        return policy[self.state_to_index((0, 0, self.Fork.Irrelevant))] == self.Action.Wait \
            and policy[self.state_to_index((1, 0, self.Fork.Irrelevant))] == self.Action.Override \
            and policy[self.state_to_index((0, 1, self.Fork.Irrelevant))] == self.Action.Adopt \
            and policy[self.state_to_index((0, 1, self.Fork.Relevant))] == self.Action.Adopt

    def calc_policy_revenue(self, policy = None):
        if policy is None:
            policy = self.opt_policy

        if self.is_policy_honest(policy):
            return -1

        policy_induced_chain = self.get_policy_induced_chain(policy)

        P = np.transpose(policy_induced_chain)
        _, v = eigs(P, k=1, which='LR', maxiter = 100 * self.num_of_states)
        steady_prob = np.real_if_close(v).flatten()
        steady_prob[np.abs(steady_prob) < 1e-16] = 0
        #Refine
        steady_prob = P.dot(P.dot(P.dot(steady_prob)))
        steady_prob = steady_prob / np.sum(steady_prob)
        print(steady_prob.shape, np.sum((P.dot(steady_prob) - steady_prob) ** 2))

        state_reward, state_honest_reward = self.get_policy_induced_chain_state_reward(policy, policy_induced_chain)
        
        r1 = self.dot(state_reward, steady_prob.transpose())
        r2 = self.dot(state_honest_reward, steady_prob.transpose())

        return r1 / (r1 + r2)

    def build_SM1_policy(self):
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

if (__name__ == '__main__'):
    print('bitcoin.infinite_blockchain_mdp module test')

    np.set_printoptions(threshold=np.nan, linewidth=np.nan)

    mdp = InfiniteBlockchainMDP(0.35, 0.5, 75)
    print(mdp.num_of_states)
    policy = mdp.build_SM1_policy()
    mdp.print_policy(policy)
    print(mdp.calc_policy_revenue(policy))
    #mdp.test_policy(policy, verbose = True)
