from collections import deque
from typing import Any

from reinforcement_learning.base.utility.buffer import Buffer


class DequeBufferWrapper(Buffer):
    def __init__(self, base_deque: deque):
        self.base_deque = base_deque

    def __len__(self) -> int:
        return len(self.base_deque)

    def append(self, element: Any) -> None:
        self.base_deque.append(element)

    def empty(self) -> None:
        self.base_deque.clear()

    def max_size(self) -> int:
        return self.base_deque.maxlen
