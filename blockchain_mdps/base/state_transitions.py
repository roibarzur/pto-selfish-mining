from typing import Dict


class StateTransitions:
    def __init__(self) -> None:
        self.probabilities: Dict[tuple, float] = {}
        self.rewards: Dict[tuple, float] = {}
        self.difficulty_contributions: Dict[tuple, float] = {}

    def add(self, next_state: tuple, probability: float = 0, reward: float = 0, difficulty_contribution: float = 0,
            allow_merging: bool = False) -> None:
        if next_state in self.probabilities:
            if not allow_merging:
                raise AssertionError('State already exists')

            # Merge with existing transitions
            self.probabilities[next_state] += probability
        else:
            # Add new transition
            self.probabilities[next_state] = probability
            self.rewards[next_state] = reward
            self.difficulty_contributions[next_state] = difficulty_contribution
