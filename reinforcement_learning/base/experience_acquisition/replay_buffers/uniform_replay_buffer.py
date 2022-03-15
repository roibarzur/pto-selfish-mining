import random

import torch

from .replay_buffer import ReplayBuffer
from ..experience import Experience
from ..experience_batch import ExperienceBatch


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size: int, buffer_size: int, device: torch.device = torch.device('cpu')):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.buffer = [Experience.create_dummy()] * self.buffer_size
        self.index = int(0)
        self.size = 0

        self.device = device

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'batch_size': self.batch_size, 'buffer_size': self.buffer_size}
        return str(d)

    def append(self, exp: Experience) -> None:
        self.buffer[self.index] = exp
        self.size += 1

        # Cap at max size
        self.size = min(self.size, self.buffer_size)

        self.index += 1
        self.index %= self.buffer_size

    def empty(self) -> None:
        self.buffer = [None] * self.buffer_size
        self.index = 0
        self.size = 0

    def sample(self) -> ExperienceBatch:
        indices = random.choices(range(self.size), k=self.batch_size)
        samples = [self.buffer[idx] for idx in indices]
        return ExperienceBatch.from_experience_list(samples, self.device)

    def max_size(self) -> int:
        return self.buffer_size

    def get_all(self) -> ExperienceBatch:
        return ExperienceBatch.from_experience_list(self.buffer[:self.size], self.device)
