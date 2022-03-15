from collections import deque

import torch

from .replay_buffer import ReplayBuffer
from ..experience import Experience
from ..experience_batch import ExperienceBatch
from ...utility.deque_buffer_wrapper import DequeBufferWrapper


class SequentialReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size: int, buffer_size: int, device: torch.device = torch.device('cpu')):
        self.batch_size = batch_size

        self.buffer = DequeBufferWrapper(deque(maxlen=buffer_size))

        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'batch_size': self.batch_size, 'buffer_size': self.max_size()}
        return str(d)

    def append(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def empty(self) -> None:
        self.buffer.empty()

    def sample(self) -> ExperienceBatch:
        samples = [self.buffer.base_deque.popleft() for _ in range(self.batch_size)]
        return ExperienceBatch.from_experience_list(samples, self.device)

    def max_size(self) -> int:
        return self.buffer.max_size()

    def get_all(self) -> ExperienceBatch:
        return ExperienceBatch.from_experience_list(list(self.buffer.base_deque), self.device)
