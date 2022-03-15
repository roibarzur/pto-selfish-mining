from collections import deque
from typing import Optional

import numpy as np

from .synchronized_callback import SynchronizedCallback
from ...experience_acquisition.agents.bva_agent import BVAAgent
from ...experience_acquisition.experience import Experience
from ...utility.buffer_synchronizer import BufferSynchronizer
from ...utility.deque_buffer_wrapper import DequeBufferWrapper


class BVACallback(SynchronizedCallback):
    def __init__(self, smart_init: Optional[float] = None, num_of_episodes_for_average: int = 20,
                 stop_goal: Optional[float] = None, sort_episodes: bool = True):
        super().__init__()
        self.agent = None
        self.smart_init = smart_init
        self.num_of_episodes_for_average = num_of_episodes_for_average
        self.episode_values = deque(maxlen=num_of_episodes_for_average)
        self.stop_goal = stop_goal
        self.epoch_history = []
        self.sort_episodes = sort_episodes
        self.episode_values_synchronizer = None

    def before_running(self, agent: BVAAgent = None, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = agent
        self.episode_values_synchronizer = BufferSynchronizer(self.sync_manager,
                                                              DequeBufferWrapper(self.episode_values),
                                                              sort=self.sort_episodes)
        self.sync_dict['base_value_approximation'] = self.smart_init if self.smart_init is not None else 0

        if self.smart_init is not None:
            for _ in range(self.num_of_episodes_for_average):
                self.episode_values.append(self.smart_init)

    def after_training_update(self, **kwargs) -> None:
        self.episode_values_synchronizer.process(self.episode_values_synchronizer.max_size())

        self.update_base_value_approximation()

    def update_base_value_approximation(self) -> None:
        if len(self.episode_values) > 0:
            base_value_approximation = np.mean(np.array(self.episode_values, dtype=np.float64))
        elif self.smart_init is not None:
            base_value_approximation = self.smart_init
        else:
            base_value_approximation = 0
        self.sync_dict['base_value_approximation'] = base_value_approximation

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> bool:
        base_value_approximation = self.sync_dict['base_value_approximation']
        self.epoch_history.append(base_value_approximation)
        # Return True (to stop running) if the stop goal is reached once enough episodes are available
        return self.stop_goal is not None and base_value_approximation >= self.stop_goal \
               and len(self.episode_values) == self.num_of_episodes_for_average

    def after_agent_update(self, **kwargs) -> None:
        self.agent.update(base_value_approximation=self.sync_dict['base_value_approximation'])

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if evaluation:
            episode_value = exp.info['revenue']
            self.episode_values_synchronizer.append(episode_value)
