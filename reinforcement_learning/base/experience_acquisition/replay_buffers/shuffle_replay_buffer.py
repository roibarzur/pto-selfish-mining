from typing import Iterable

import numpy as np
import torch

from .replay_buffer import ReplayBuffer
from ..experience_batch import ExperienceBatch


class ShuffleReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        self.buffer = None
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'batch_size': self.batch_size, 'buffer_size': self.size}
        return str(d)

    def append(self, exp: ExperienceBatch) -> None:
        if self.buffer is None:
            self.buffer = exp
        else:
            self.buffer = self.buffer.combine(exp)

        self.index = 0
        self.size = len(self.buffer.rewards)

    def empty(self) -> None:
        self.buffer = None
        self.index = 0
        self.size = 0

    def sample(self) -> ExperienceBatch:
        if self.index == 0:
            self.shuffle()

        # Take samples sequentially starting from the last index not taken
        indices = range(self.index, min(self.size, self.index + self.batch_size))
        batch = self.select_from_buffer(indices)

        self.index = self.index + self.batch_size
        if self.index >= self.size:
            # Start over
            self.index = 0

        return batch

    def shuffle(self) -> None:
        indices = np.random.permutation(self.size)
        self.buffer = self.select_from_buffer(indices)

    def select_from_buffer(self, indices: Iterable[int]):
        indices = torch.tensor(indices, dtype=torch.long)
        # noinspection PyProtectedMember
        shuffled_tensors = {attr: tensor.index_select(0, indices) for attr, tensor in self.buffer._asdict().items()}
        selected = ExperienceBatch(**shuffled_tensors)
        return selected

    def max_size(self) -> int:
        return self.size

    def get_all(self) -> ExperienceBatch:
        return self.buffer
