from __future__ import annotations

import math
import sys

import numpy as np
from tabulate import tabulate

from blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP
from .base.base_space.dict_space import DictSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions
from .bitcoin_model import BitcoinModel

debug = 0
print_string = 0


def print_dbg(s):
    if debug:
        print(f'DEBUG {s}')


class BinaryList(list):
    def __str__(self):
        s = ''
        for i in self:
            s += f'{i}'
        return '[' + s + ']'
        # if print_string or s == '':
        #     return '[' + s + ']'
        # return f'[{int(s, 2)}]'

    def __hash__(self):
        return self.__str__().__hash__()


class mySet(set):
    def __hash__(self):
        return self.__str__().__hash__()

    def __str__(self):
        if len(self) == 0:
            return '{}'
        if len(self) == 1:
            for i in self:
                return f'{i}'
        s = '{'
        for i in self:
            s += f'{i},'
        return s[:-1] + '}'


class StateElement:
    def __init__(self, a: int, h: int, L: BinaryList, T_a: int, T_h: int, fork, pool: int, max_pool: int, max_lead=100):
        self.a = a
        self.h = h
        self.L = BinaryList(L)
        self.T_a = T_a
        self.T_h = T_h
        self.pool = pool
        self.fork = fork
        self.max_pool = max_pool
        self.max_lead = max_lead

    def __str__(self):
        return f'a={self.a}, h={self.h}, L={self.L}, T_a={self.T_a}, T_h={self.T_h}, fork={self.fork.name}, pool={self.pool}'

    def __hash__(self):
        return f'{self}'.__hash__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def to_tuple(self):
        return self
        # return self.a, self.h, self.L, self.T_a, self.T_h, self.pool, self.fork

    def is_legal(self):
        honest_fee = self.T_h
        attacker_fee = self.T_a + sum(self.L)
        legal = honest_fee <= self.pool and attacker_fee <= self.pool and len(self.L) == max(0, self.a - self.h)
        legal = legal and self.pool <= self.max_pool
        legal = legal and (self.a - self.h <= self.max_lead)
        print_dbg(f'{self} is {legal}')
        return legal

    def check_state(self, state: StateElement, func_name: str):

        print_dbg(f'{func_name}:: {self} --> {state}')
        if state.T_a + sum(state.L) > state.pool:
            raise Exception(f'!ERROR!! {func_name}:: {state} \t attacker got more fees that can have')
        if state.T_h > state.pool:
            raise Exception(f'!!ERROR!! {func_name}::{state} \t honest got more fees that can have')
        if len(state.L) != max(0, state.a - state.h):
            raise Exception(f'!!ERROR!! {func_name}::{state} \t the length of the list is not correct')
        if state.pool > self.pool:
            raise Exception(f'{func_name} how fee incd!!!!!')
        if state.pool > self.max_pool:
            raise Exception(f'{state} more mem_pool than allowed')
        if state.a - state.h > state.max_lead:
            raise Exception(f'{state} a-h is big!')

    def override(self, next_fork):
        assert self.a > self.h
        next_state = self.copy()
        next_state.a -= next_state.h + 1
        next_state.h = 0

        next_state.T_a = 0
        next_state.T_h = 0
        next_state.update_fork(next_fork)

        next_state.pool -= self.T_a + next_state.L.pop(0)
        self.check_state(next_state, 'override')
        return next_state

    def adopt(self, next_fork):
        next_state = self.copy()
        next_state.a = 0
        next_state.h = 0
        next_state.L = BinaryList([])
        next_state.T_a = 0
        next_state.T_h = 0
        next_state.pool = self.pool - self.T_h
        next_state.update_fork(next_fork)

        next_state.check_state(next_state, 'adopt')
        return next_state

    def match(self, next_fork):
        next_state = self.copy()
        next_state.update_fork(next_fork)
        return next_state

    def honest_find(self, next_fork=None):
        next_state = self.copy()
        next_state.h += 1

        honest_sum_transactions = self.T_h

        with_fee = int(honest_sum_transactions < self.pool)

        next_state.T_h += with_fee
        if self.a <= self.h:
            assert (next_state.L == [])
        else:  # before we had a>h
            next_state.T_a += next_state.L.pop(0)
        next_state.update_fork(next_fork)

        self.check_state(next_state, 'honest_find')
        return next_state

    def attacker_find(self, next_fork=None):
        next_state = self.copy()
        next_state.a += 1
        attacker_sum_transactions = self.T_a + sum(self.L)
        with_fee = int(attacker_sum_transactions < self.pool)
        if next_state.a <= next_state.h:
            next_state.T_a += with_fee
            assert (len(next_state.L) == 0)
        else:
            next_state.L.append(with_fee)

        next_state.update_fork(next_fork)
        self.check_state(next_state, 'attacker_find')
        return next_state

    def honest_support_find(self, next_fork=None):
        next_state = self.copy()
        next_state.a -= self.h

        next_state.pool -= next_state.T_a
        next_state.T_a = next_state.L.pop(0) if len(next_state.L) > 0 else 0

        next_state.h = 1

        with_fee = int(next_state.pool > 0)
        next_state.T_h = with_fee

        next_state.update_fork(next_fork)
        self.check_state(next_state, 'honest_support_find')
        return next_state

    def tryGetWithAndWithoutNewFee(self, prev_state: StateElement):
        # this might return the same next_state if the max fork size didn't change and no place for extra fee
        current_max_fork = max(self.h, self.a)
        prev_max_fork = max(prev_state.h, prev_state.a)
        new_with_fee = self.copy()
        new_without_fee = self.copy()

        if prev_max_fork < current_max_fork and self.pool < self.max_pool:
            new_with_fee.pool += 1
        return new_with_fee, new_without_fee

    def update_fork(self, fork):
        if fork is not None:
            self.fork = fork

    def copy(self) -> StateElement:
        return StateElement(self.a, self.h, BinaryList(self.L.copy()), self.T_a, self.T_h, self.fork, self.pool,
                            self.max_pool)


class BitcoinSimplifiedFeeModel(BlockchainModel):
    State = StateElement
    Policy = tuple

    def __init__(self, alpha: float, gamma: float, max_fork: int, max_pool: int, fee: float, transaction_chance: float,
                 max_lead=10, normalize_reward=1):
        self.max_lead = max_lead
        self.alpha = alpha
        self.gamma = gamma
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance
        self.Fork_list = ['Irrelevant', 'Relevant', 'Active']
        self.Fork = self.create_int_enum('Fork', self.Fork_list)
        self.Actions_list = ['Illegal', 'Adopt', 'Override', 'Match', 'Wait']
        self.Action = self.create_int_enum('Action', self.Actions_list)

        self.normalize_reward = normalize_reward

        self.max_pool = max_pool

        self.idx2state = None
        self.state2idx = None
        self.honest_policy = None
        super().__init__()

    def __str__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, max_fork={self.max_fork}, ' \
               f'max_pool={self.max_pool}, k={self.fee}, delta={self.transaction_chance}, max_lead={self.max_lead}, ' \
               f'normalize_reward={self.normalize_reward})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_fork}, ' \
               f'{self.max_pool}, {self.fee}, {self.transaction_chance}, {self.max_lead}, ' \
               f'{self.normalize_reward})'

    def __reduce__(self):
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.fee, self.transaction_chance)

    def get_state_space(self) -> Space:
        idx2state, state2idx = self.create_state_idx_dict()
        return DictSpace(idx2state=idx2state, state2idx=state2idx, default_value=self.get_final_state(), dimension=6)

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BitcoinSimplifiedFeeModel.State:
        return StateElement(a=0, h=0, L=BinaryList([]), T_a=0, T_h=0, fork=self.Fork.Irrelevant, pool=0,
                            max_pool=self.max_pool)

    def get_final_state(self) -> BitcoinSimplifiedFeeModel.State:
        return StateElement(a=-1, h=-1, L=BinaryList([]), T_a=-1, T_h=-1, fork=self.Fork.Irrelevant, pool=-1,
                            max_pool=self.max_pool)

    # noinspection DuplicatedCode
    def get_state_transitions(self, state: BitcoinSimplifiedFeeModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        # if f'now we have next_state={next_state}, action={action.name}' == \
        #         'now we have next_state=a=2, h=2, L=[], T_a=2, T_h=2, fork=Active pool=2, action=Wait':
        #     print('after this')

        transitions = StateTransitions()
        # print(f'now we have next_state={next_state}, action={action.name}')

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, L, T_a, T_h, fork, pool = state.a, state.h, state.L, state.T_a, state.T_h, state.fork, state.pool

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            return transitions

        if a - h == self.max_lead and action is not self.Action.Override:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 4)
            return transitions

        if action is self.Action.Adopt:
            if h > 0:
                next_state = state.adopt(self.Fork.Irrelevant)
                transitions.add(next_state.to_tuple(), probability=1,
                                difficulty_contribution=h)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Override:
            if a > h:
                next_state = state.override(next_fork=self.Fork.Irrelevant)
                num_of_fees = T_a + L[0]
                transitions.add(next_state.to_tuple(), probability=1,
                                reward=(h + 1 + self.fee * num_of_fees) / (
                                        1 + self.normalize_reward * self.fee * self.transaction_chance),

                                difficulty_contribution=h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty, allow_merging=True)

        if action is self.Action.Match:
            if 0 < h <= a < self.max_fork and fork is self.Fork.Relevant:
                next_state = state.match(self.Fork.Active)
                transitions.add(next_state.to_tuple(), probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block_with_extra_fee, attacker_block_without_extra_fee = \
                    state.attacker_find(self.Fork.Irrelevant).tryGetWithAndWithoutNewFee(state)

                transitions.add(attacker_block_with_extra_fee.to_tuple(),
                                probability=self.alpha * self.transaction_chance)
                transitions.add(attacker_block_without_extra_fee.to_tuple(),
                                probability=self.alpha * (1 - self.transaction_chance), allow_merging=True)

                honest_block_with_extra_fee, honest_block_without_extra_fee = \
                    state.honest_find(next_fork=self.Fork.Relevant).tryGetWithAndWithoutNewFee(state)
                transitions.add(honest_block_with_extra_fee.to_tuple(),
                                probability=(1 - self.alpha) * self.transaction_chance)
                transitions.add(honest_block_without_extra_fee.to_tuple(),
                                probability=(1 - self.alpha) * (1 - self.transaction_chance), allow_merging=True)

            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:

                attacker_block_with_extra_fee, attacker_block_without_extra_fee = \
                    state.attacker_find(next_fork=self.Fork.Active).tryGetWithAndWithoutNewFee(state)
                transitions.add(attacker_block_with_extra_fee.to_tuple(),
                                probability=self.alpha * self.transaction_chance)
                transitions.add(attacker_block_without_extra_fee.to_tuple(),
                                probability=self.alpha * (1 - self.transaction_chance), allow_merging=True)

                honest_block_with_extra_fee, honest_block_without_extra_fee = \
                    state.honest_find(self.Fork.Relevant).tryGetWithAndWithoutNewFee(state)
                transitions.add(honest_block_with_extra_fee.to_tuple(),
                                probability=(1 - self.gamma) * (1 - self.alpha) * self.transaction_chance)
                transitions.add(honest_block_without_extra_fee.to_tuple(),
                                probability=(1 - self.gamma) * (1 - self.alpha) * (1 - self.transaction_chance),
                                allow_merging=True)

                honest_support_block_with_extra_fee, honest_support_block_without_extra_fee = \
                    state.honest_support_find(self.Fork.Relevant).tryGetWithAndWithoutNewFee(state)
                transitions.add(honest_support_block_with_extra_fee.to_tuple(),
                                probability=self.gamma * (1 - self.alpha) * self.transaction_chance,
                                reward=(h + self.fee * T_a) / (
                                        1 + self.normalize_reward * self.fee * self.transaction_chance),
                                difficulty_contribution=h)

                transitions.add(honest_support_block_without_extra_fee.to_tuple(),
                                probability=self.gamma * (1 - self.alpha) * (1 - self.transaction_chance),
                                reward=(h + self.fee * T_a) / (
                                        1 + self.normalize_reward * self.fee * self.transaction_chance),
                                difficulty_contribution=h, allow_merging=True)

            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def get_honest_revenue(self) -> float:
        if self.normalize_reward == 0:
            return self.alpha * (1 + self.fee * self.transaction_chance)
        else:
            return self.alpha

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:

        honest_policy = self.build_honest_policy()
        reachable_in_honest_True_False = SparseBlockchainMDP(self).find_reachable_states(honest_policy)
        reachable_indices = np.where(reachable_in_honest_True_False == True)[0]
        l1 = np.array(list(honest_policy))
        l2 = np.array(list(policy))
        return all(np.array(l1[reachable_indices] == l2[reachable_indices]))

    def build_honest_policy(self) -> BlockchainModel.Policy:
        if self.honest_policy is not None:
            return self.honest_policy
        policy = np.zeros(self.state_space.size, dtype=int)
        policy_dict = {}

        for i in range(self.state_space.size):
            state = self.state_space.index_to_element(i)
            if state.h > state.a:
                action = self.Action.Adopt
            elif state.a > state.h:
                action = self.Action.Override
            else:
                action = self.Action.Wait
            policy[i] = action
            policy_dict[state] = action.name

        self.honest_policy = tuple(policy)
        return tuple(policy)

    def build_sm1_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            state = self.state_space.index_to_element(i)
            a, h, fork = state.a, state.h, state.fork

            if h > a:
                action = self.Action.Adopt
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Override
            elif (h == 1 and a == 1) and fork is self.Fork.Relevant:
                action = self.Action.Match
            else:
                action = self.Action.Wait

            policy[i] = action
        return tuple(policy)

    def create_state_idx_dict(self):
        if self.idx2state is not None:
            return self.idx2state, self.state2idx
        state2idx = {}
        idx2state = {}

        idx_counter = 0
        state2idx[self.get_final_state()] = idx_counter
        idx2state[idx_counter] = self.get_final_state()
        idx_counter += 1
        for a in range(self.max_fork + 1):
            # print(f'create_state_idx_dict:: a={a}')
            for h in range(self.max_fork + 1):

                a_h_diff = a - h

                if a_h_diff > self.max_lead:
                    print_dbg(f'big diff:: a={a}, h={h},ignoring')
                    continue

                L_a_len = max(a_h_diff, 0)
                L_a_options = 2 ** L_a_len

                T_a_options = min(a, h) + 1

                L_h_len = h
                # L_h_options = 2 ** L_h_len
                L_h_options = h + 1

                for L_a_num in range(L_a_options):
                    for T_a in range(T_a_options):
                        for L_h_num in range(L_h_options):
                            for fork in self.Fork:
                                for pool in range(self.max_pool + 1):
                                    L_a = create_binary_list(L_a_num, L_a_len)
                                    state = StateElement(a, h, L_a, T_a, L_h_num, fork, pool, self.max_pool)
                                    if state.is_legal() is False:
                                        continue
                                    state2idx[state] = idx_counter
                                    idx2state[idx_counter] = state
                                    idx_counter += 1
        self.idx2state = idx2state
        self.state2idx = state2idx
        return idx2state, state2idx

    def print_policy(self, policy: Policy, reachable_states: np.array, print_size: int = 8, x_axis: int = 1,
                     y_axis: int = 0, z_axis: int = 2, print_full_list=True, output=sys.stdout) -> None:
        global print_string
        print_string = print_full_list
        self.create_state_idx_dict()

        fork_mat = {}  # 3D matrix [fork][a][h] -> ((T_a,L,T_h,fee),fork)
        for fork in self.Fork:
            fork_mat[fork] = []
            for a in range(self.max_fork + 1):
                a_list = []
                for b in range(self.max_fork + 1):
                    a_list.append([])
                fork_mat[fork].append([a] + a_list)

        # adding the states->action to the data structure
        for idx in range(self.state_space.size):
            state = self.state_space.index_to_element(idx)
            if reachable_states[idx]:
                # print(f'{idx} {next_state} :: {self.Actions_list[policy[idx]]}')
                action = self.Actions_list[policy[idx]]
                # action2states[action].append(f'{next_state}')
                # state2action[next_state] = action
                fork_mat[state.fork][state.a][state.h + 1].append(
                    ((state.T_a, state.L, state.T_h, state.pool), action[0:1]))
        s = f'{self}\n'
        for fork in self.Fork:
            for a in range(self.max_fork + 1):
                for h in range(self.max_fork + 1):
                    state2action = fork_mat[fork][a][h + 1]
                    copress_fees = self.compress_state_action(state2action)
                    fork_mat[fork][a][h + 1] = copress_fees
            s += tabulate(fork_mat[fork], headers=[f'fork={fork.name}\na\\h'] + [*range(self.max_fork + 1)],
                          tablefmt='fancy_grid') + '\n'
        s += '~' * 150 + '\n'
        output.write(f'{s}')

    def extend_policy_to_transactions(self, policy_without: tuple):
        policy_with = np.zeros(self.state_space.size, dtype=int)
        mdp_without = BitcoinModel(self.alpha, self.gamma, self.max_fork)
        for i in range(mdp_without.state_space.size):
            a, h, fork = mdp_without.state_space.index_to_element(i)
            states_with_same_a_h_fork = self.state_space.get_all_indices_with(a, h, fork)
            policy_with[states_with_same_a_h_fork] = policy_without[i]
        return tuple(policy_with)

    def compress_state_action(self, states2action):
        dict2pool = {}
        for (T_a, L, T_h, pool), action in states2action:
            state = T_a, L, T_h, action
            if state not in dict2pool.keys():
                dict2pool[state] = mySet()
            (dict2pool[state]).add(pool)

        dict2Th = {}
        for (T_a, L, T_h, action), pool in dict2pool.items():
            state = T_a, L, action, pool
            if state not in dict2Th.keys():
                dict2Th[state] = mySet()
            (dict2Th[state]).add(T_h)

        dict2L = {}
        for (T_a, L, action, pool), T_h in dict2Th.items():
            state = T_a, action, T_h, pool
            if state not in dict2L.keys():
                dict2L[state] = mySet()
            (dict2L[state]).add(L)

        dict2Ta = {}
        for (T_a, action, T_h, pool), L in dict2L.items():
            state = action, L, T_h, pool
            if state not in dict2Ta.keys():
                dict2Ta[state] = mySet()
            (dict2Ta[state]).add(T_a)

        compressed = ''
        for (action, L, T_h, pool), T_a in dict2Ta.items():
            compressed += f'{T_a},{L}/{T_h}/{pool}->{action}\n'
        return compressed

    def print_states_part(self, reachable):
        idx_in_loop = 0
        for idx in range(len(reachable)):
            if reachable[idx]:
                print(f'idx={idx_in_loop}, state={self.idx2state[idx]}')
                idx_in_loop += 1


def create_binary_list(num, length):
    if length < 0 or num < 0:
        raise ValueError(f'{length=}, {num=}')
    if length == 0 and num != 0:
        raise ValueError
    if num > 2 ** length:
        raise ValueError

    l = BinaryList([0] * length)
    idx = num
    while idx > 0:
        place = math.floor(math.log2(idx))
        l[place] = 1
        idx -= 2 ** place
    return l


if __name__ == '__main__':
    print('bitcoin_mdp module test')

    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    alpha = 0.35
    gamma = 0.0
    max_fork = 6
    # horizon = int(1e5)
    epsilon = 1e-5
    max_iter = 100000

    mdp = BitcoinSimplifiedFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, max_pool=max_fork + 1, k=0,
                                    delta=0.5)  # print(mdp.state_space.size)
    # print(mdp.state_space.index_to_element(3))
    # mdp.print_states()

    p = mdp.build_honest_policy()
    my_policy = mdp.build_sm1_policy()
    # # mdp.print_states()
    #

    #
    solver = SparseBlockchainMDP(mdp)

    honest_reachable = solver.find_reachable_states(p)

    # my_reachable = solver.find_reachable_states(my_policy)
    # #
    # # mdp.print_policy(p, honest_reachable)
    # print()
    # mdp.print_policy(my_policy, my_reachable)
    # print(f'is my new policy honest? answer: {mdp.is_policy_honest(p)}')
    #
    # print(mdp.is_policy_honest(my_policy))
    # print(solver.calc_policy_revenue(my_policy))
    print(mdp.is_same_policy(p, p, honest_reachable, honest_reachable))
