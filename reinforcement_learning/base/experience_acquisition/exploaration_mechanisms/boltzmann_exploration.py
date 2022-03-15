from typing import Dict

import numpy as np
import scipy.special

from .exploration_mechanism import ExplorationMechanism


class BoltzmannExploration(ExplorationMechanism):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'temperature': self.temperature}
        return str(d)

    def explore(self, action_values: Dict[int, float], state_visit_count: int) -> int:
        actions = np.array(list(action_values.keys()))
        values = np.array(list(action_values.values()))

        return np.random.choice(actions, p=scipy.special.softmax(values / self.temperature))
