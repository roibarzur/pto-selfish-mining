from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, Union

import torch


class ExplorationMechanism(ABC):
    def choose_action(self, action_values: Union[Dict[int, float], torch.tensor], explore: bool, state_visit_count: int
                      ) -> int:
        if torch.is_tensor(action_values):
            action_values_dict = {}
            for idx, value in enumerate(action_values):
                if value != float('-inf'):
                    action_values_dict[idx] = value
        else:
            action_values_dict = action_values

        if not explore:
            return self.choose_best_action(action_values_dict)
        else:
            return self.explore(action_values_dict, state_visit_count)

    @staticmethod
    def choose_best_action(action_values: Dict[int, float]) -> int:
        return max(action_values.items(), key=itemgetter(1))[0]

    @abstractmethod
    def explore(self, action_values: Dict[int, float], state_visit_count: int) -> int:
        pass
