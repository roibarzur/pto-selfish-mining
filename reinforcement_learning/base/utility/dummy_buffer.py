from typing import Any

from reinforcement_learning.base.utility.buffer import Buffer


class DummyBuffer(Buffer):
    def __init__(self, size: int = 500):
        self.size = size

    def __len__(self) -> int:
        return 0

    def append(self, element: Any) -> None:
        pass

    def empty(self) -> None:
        pass

    def max_size(self) -> int:
        return self.size
