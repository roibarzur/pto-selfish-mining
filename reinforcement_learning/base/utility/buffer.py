from abc import ABC, abstractmethod
from typing import Any


class Buffer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def append(self, element: Any) -> None:
        pass

    @abstractmethod
    def empty(self) -> None:
        pass

    @abstractmethod
    def max_size(self) -> int:
        pass
