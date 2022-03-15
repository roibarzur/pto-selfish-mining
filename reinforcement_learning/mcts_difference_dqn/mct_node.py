from collections import defaultdict
from typing import Dict, List

from blockchain_mdps import StateTransitions


class MCTNode:
    def __init__(self, state: tuple, legal_actions: List[int], action_state_transitions: Dict[int, StateTransitions],
                 approximated_value: float, prior_action_probabilities: Dict[int, float]):
        self.state = state

        self.expanded = False

        self.legal_actions = legal_actions
        self.action_state_transitions = action_state_transitions

        self.approximated_value = approximated_value
        self.prior_action_probabilities = prior_action_probabilities

        self.visit_count = 0
        self.action_counts: Dict[int, int] = defaultdict(int)

        self.mc_estimated_q_values: Dict[int, float] = {}

    def get_value(self, use_nn: bool = False) -> float:
        if self.visit_count == 0 or use_nn:
            return self.approximated_value
        else:
            return max(self.mc_estimated_q_values.values())

    def __repr__(self) -> str:
        return f'Node<{self.state}>'
