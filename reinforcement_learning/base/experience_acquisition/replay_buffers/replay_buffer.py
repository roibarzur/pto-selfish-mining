from abc import ABC, abstractmethod

from reinforcement_learning.base.experience_acquisition.experience_batch import ExperienceBatch
from reinforcement_learning.base.utility.buffer import Buffer


class ReplayBuffer(Buffer, ABC):
    @abstractmethod
    def sample(self) -> ExperienceBatch:
        pass

    @abstractmethod
    def get_all(self) -> ExperienceBatch:
        pass
