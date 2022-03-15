import random
from typing import Dict

from ....base.utility.parameter_schedule import ParameterSchedule
from .exploration_mechanism import ExplorationMechanism


class EpsilonGreedyExploration(ExplorationMechanism):
    def __init__(self, starting_epsilon: float, epsilon_step: float):
        self.epsilon_schedule = ParameterSchedule(starting_epsilon, epsilon_step)

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'epsilon_schedule': self.epsilon_schedule}
        return str(d)

    def explore(self, action_values: Dict[int, float], state_visit_count: int) -> int:
        if random.random() < self.epsilon_schedule.get_parameter():
            return random.choice(list(action_values.keys()))
        else:
            return self.choose_best_action(action_values)
