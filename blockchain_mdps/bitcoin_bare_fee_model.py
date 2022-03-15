import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class BitcoinBareFeeModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int, fee: float, transaction_chance: float, max_pool: int):
        self.alpha = alpha
        self.gamma = gamma
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance
        self.max_pool = max(max_pool, max_fork)

        # self.block_reward = 1 / (1 + self.transaction_chance * self.fee)
        # No need for normalization
        self.block_reward = 1

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Reveal', 'Mine'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.fee, self.transaction_chance, self.max_pool)

    def get_state_space(self) -> Space:
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_fork))

    def get_initial_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (0,)

    def get_final_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,)

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int]:
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]
        fork = state[-2]
        pool = state[-1]

        return a, h, fork, pool

    def create_empty_chain(self) -> tuple:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:
        # Check chain length
        if len(chain) != self.max_fork * 2:
            return False

        # Check chain types
        valid_parts = sum(isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
                          for block, transaction in zip(chain[::2], chain[1::2]))
        if valid_parts < self.max_fork:
            return False

        # Check the chain starts with blocks
        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min([self.max_fork - 1]
                             + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        # Check no transactions in non-blocks
        invalid_transactions = sum(block is self.Block.NoBlock and transaction is self.Transaction.With
                                   for block, transaction in zip(chain[::2], chain[1::2]))
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        a, h, fork, pool = self.dissect_state(state)
        return self.is_chain_valid(a) and self.is_chain_valid(h) \
               and self.chain_transactions(a) <= pool \
               and self.chain_transactions(h) <= pool

    @staticmethod
    def truncate_chain(chain: tuple, truncate_to: int) -> tuple:
        return chain[:2 * truncate_to]

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def chain_length(self, chain: tuple) -> int:
        return len([block for block in chain[::2] if block is self.Block.Exists])

    def chain_transactions(self, chain: tuple) -> int:
        return len([block for block, transaction in zip(chain[::2], chain[1::2])
                    if block is self.Block.Exists and transaction is self.Transaction.With])

    def add_block(self, chain: tuple, add_transaction: bool) -> tuple:
        transaction = self.Transaction.With if add_transaction else self.Transaction.NoTransaction
        index = self.chain_length(chain)
        chain = list(chain)
        chain[2 * index] = self.Block.Exists
        chain[2 * index + 1] = transaction
        return tuple(chain)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, pool = self.dissect_state(state)
        action_type, action_param = action
        length_h = self.chain_length(h)
        length_a = self.chain_length(a)
        transactions_h = self.chain_transactions(h)
        transactions_a = self.chain_transactions(a)

        if action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action_type is self.Action.Adopt:
            if 0 < action_param <= length_h:
                accepted_transactions = self.chain_transactions(self.truncate_chain(h, action_param))
                next_state = self.create_empty_chain() + self.shift_back(h, action_param) \
                             + (self.Fork.Irrelevant, pool - accepted_transactions)
                transitions.add(next_state, probability=1, difficulty_contribution=action_param)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Reveal:
            if length_h < action_param <= length_a:
                accepted_transactions = self.chain_transactions(self.truncate_chain(a, action_param))
                next_state = self.shift_back(a, action_param) + self.create_empty_chain() \
                             + (self.Fork.Irrelevant, pool - accepted_transactions)
                reward = (action_param + accepted_transactions * self.fee) * self.block_reward
                transitions.add(next_state, probability=1, reward=reward, difficulty_contribution=action_param)

            elif 0 < length_h == action_param <= length_a < self.max_fork \
                    and fork is self.Fork.Relevant:
                next_state = a + h + (self.Fork.Active, pool)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Mine:
            if fork is not self.Fork.Active and length_a < self.max_fork \
                    and length_h < self.max_fork \
                    and action_param in [self.Transaction.With, self.Transaction.NoTransaction]:
                new_transaction_chance = self.transaction_chance if length_a >= length_h else 0

                add_transaction = action_param == self.Transaction.With and transactions_a < pool
                attacker_block_no_new_transaction = self.add_block(a, add_transaction) + h \
                                                    + (self.Fork.Irrelevant, pool)
                transitions.add(attacker_block_no_new_transaction,
                                probability=self.alpha * (1 - new_transaction_chance))

                attacker_block_new_transaction = self.add_block(a, add_transaction) + h \
                                                 + (self.Fork.Irrelevant, min(self.max_pool, pool + 1))
                transitions.add(attacker_block_new_transaction, probability=self.alpha * new_transaction_chance,
                                allow_merging=True)

                new_transaction_chance = self.transaction_chance if length_a <= length_h else 0

                add_transaction = transactions_h < pool
                honest_block_no_new_transaction = a + self.add_block(h, add_transaction) \
                                                  + (self.Fork.Relevant, pool)
                transitions.add(honest_block_no_new_transaction,
                                probability=(1 - self.alpha) * (1 - new_transaction_chance))

                honest_block_new_transaction = a + self.add_block(h, add_transaction) \
                                               + (self.Fork.Relevant, min(self.max_pool, pool + 1))
                transitions.add(honest_block_new_transaction, probability=(1 - self.alpha) * new_transaction_chance,
                                allow_merging=True)

            elif fork is self.Fork.Active and 0 < length_h <= length_a < self.max_fork \
                    and action_param in [self.Transaction.With, self.Transaction.NoTransaction]:
                add_transaction = action_param == self.Transaction.With and transactions_a < pool
                attacker_block_no_new_transaction = self.add_block(a, add_transaction) + h \
                                                    + (self.Fork.Active, pool)
                transitions.add(attacker_block_no_new_transaction,
                                probability=self.alpha * (1 - self.transaction_chance))

                attacker_block_new_transaction = self.add_block(a, add_transaction) + h \
                                                 + (self.Fork.Active, min(self.max_pool, pool + 1))
                transitions.add(attacker_block_new_transaction, probability=self.alpha * self.transaction_chance,
                                allow_merging=True)

                accepted_blocks = length_h
                accepted_transactions = self.chain_transactions(self.truncate_chain(a, accepted_blocks))
                reward = (accepted_blocks + accepted_transactions * self.fee) * self.block_reward

                new_transaction_chance = self.transaction_chance if length_a == length_h else 0
                add_transaction = accepted_transactions < pool
                honest_support_block_no_new_transaction = self.shift_back(a, accepted_blocks) \
                                                          + self.add_block(self.create_empty_chain(), add_transaction) \
                                                          + (self.Fork.Relevant, pool - accepted_transactions)
                transitions.add(honest_support_block_no_new_transaction,
                                probability=self.gamma * (1 - self.alpha) * (1 - new_transaction_chance),
                                reward=reward, difficulty_contribution=accepted_blocks)

                honest_support_block_new_transaction = self.shift_back(a, accepted_blocks) \
                                                       + self.add_block(self.create_empty_chain(), add_transaction) \
                                                       + (self.Fork.Relevant,
                                                          min(pool - accepted_transactions + 1, self.max_pool))
                transitions.add(honest_support_block_new_transaction,
                                probability=self.gamma * (1 - self.alpha) * new_transaction_chance,
                                reward=reward, difficulty_contribution=accepted_blocks, allow_merging=True)

                add_transaction = transactions_h < pool
                honest_adversary_block_no_new_transaction = a + self.add_block(h, add_transaction) \
                                                            + (self.Fork.Relevant, pool)
                transitions.add(honest_adversary_block_no_new_transaction,
                                probability=(1 - self.gamma) * (1 - self.alpha) * (1 - new_transaction_chance),
                                allow_merging=True)

                honest_adversary_block_new_transaction = a + self.add_block(h, add_transaction) \
                                                         + (self.Fork.Relevant, min(self.max_pool, pool + 1))
                transitions.add(honest_adversary_block_new_transaction,
                                probability=(1 - self.gamma) * (1 - self.alpha) * new_transaction_chance,
                                allow_merging=True)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha * self.block_reward * (1 + self.fee * self.transaction_chance)


if __name__ == '__main__':
    print('bitcoin_bare_fee_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = BitcoinBareFeeModel(0.35, 0.5, 2, fee=2, transaction_chance=0.1, max_pool=2)
    print(mdp.state_space.size)
