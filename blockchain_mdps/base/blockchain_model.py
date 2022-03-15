from abc import ABC, abstractmethod
from enum import Enum, EnumMeta, IntEnum
from typing import List

import numpy as np

from .base_space.space import Space
from .state_transitions import StateTransitions


class BlockchainModel(ABC):
    Policy = tuple
    State = tuple
    Action = Enum

    def __init__(self) -> None:
        self.state_space = self.get_state_space()
        self.action_space = self.get_action_space()

        self.initial_state = self.get_initial_state()
        self.final_state = self.get_final_state()

        self.error_penalty = int(-1e5)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def get_state_space(self) -> Space:
        pass

    @abstractmethod
    def get_action_space(self) -> Space:
        pass

    @abstractmethod
    def get_initial_state(self) -> State:
        pass

    @abstractmethod
    def get_final_state(self) -> State:
        pass

    @staticmethod
    def create_int_enum(enum_name: str, names: List[str]) -> EnumMeta:
        d = zip(names, range(len(names)))
        enum = IntEnum(enum_name, d)

        def print_start(value: IntEnum):
            return value.name[:3].lower()

        enum.__repr__ = print_start
        enum.__str__ = print_start
        return enum

    def print_states(self) -> None:
        for state in self.state_space.enumerate_elements():
            print(self.state_space.element_to_index(state), state)

    @abstractmethod
    def get_state_transitions(self, state: State, action: Action, check_valid: bool = True) -> StateTransitions:
        pass

    @abstractmethod
    def get_honest_revenue(self) -> float:
        pass

    def print_policy(self, policy: Policy, reachable_states: np.array, print_size: int = 8, x_axis: int = 1,
                     y_axis: int = 0, z_axis: int = 2) -> None:
        x_range = self.state_space.enumerate_dimension(x_axis)[:print_size]
        y_range = self.state_space.enumerate_dimension(y_axis)[:print_size]
        z_range = self.state_space.enumerate_dimension(z_axis)[:print_size]

        policy_table = np.zeros((len(y_range), len(x_range)), dtype=object)

        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                s = ""
                for z in z_range:
                    state = list(self.initial_state)
                    state[x_axis] = x
                    state[y_axis] = y
                    state[z_axis] = z
                    state = tuple(state)

                    state_index = self.state_space.element_to_index(state)
                    action = self.action_space.index_to_element(policy[state_index])
                    if not reachable_states[state_index]:
                        ch = '*'
                    elif action is self.Action.Illegal:
                        ch = '-'
                    else:
                        if isinstance(action, IntEnum):
                            ch = action.name[0].lower()
                        else:
                            ch = action[0].name[0].lower() + str(action[1])
                    s += ch

                policy_table[i, j] = s

        print(policy_table)
