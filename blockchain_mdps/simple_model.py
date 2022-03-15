import sys
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class SimpleModel(BlockchainModel):
    def __init__(self, alpha: float, max_fork: int):
        self.alpha = alpha
        self.max_fork = max_fork

        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Override', 'Wait'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork)

    def get_state_space(self) -> Space:
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork))
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1

    # noinspection DuplicatedCode
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h = state

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action is self.Action.Adopt:
            if h > 0:
                next_state = 0, 0
                transitions.add(next_state, probability=1, difficulty_contribution=h)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Override:
            if a > h:
                next_state = a - h - 1, 0
                transitions.add(next_state, probability=1, reward=h + 1, difficulty_contribution=h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Wait:
            if a < self.max_fork and h < self.max_fork:
                attacker_block = a + 1, h
                transitions.add(attacker_block, probability=self.alpha)

                honest_block = a, h + 1
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        return policy[self.state_space.element_to_index((0, 0))] == self.Action.Wait \
               and policy[self.state_space.element_to_index((1, 0))] == self.Action.Override \
               and policy[self.state_space.element_to_index((0, 1))] == self.Action.Adopt

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h = self.state_space.index_to_element(i)

            if h > a:
                action = self.Action.Adopt
            elif a > h:
                action = self.Action.Override
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('bitcoin_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = SimpleModel(0.35, 100)
    print(mdp.state_space.size)
    p = mdp.build_honest_policy()
    print(mdp.is_policy_honest(p))
    print(p[:10])
