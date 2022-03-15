from abc import ABC, abstractmethod
from random import randrange
from typing import Any, List, Iterable


class Space(ABC):
    def __init__(self) -> None:
        self.dimension = self._calc_dimension()
        self.size = self._calc_size()

    @abstractmethod
    def _calc_dimension(self) -> int:
        pass

    @abstractmethod
    def _calc_size(self) -> int:
        pass

    @abstractmethod
    def element_to_index(self, element: Any) -> int:
        pass

    @abstractmethod
    def index_to_element(self, index: int) -> Any:
        pass

    @abstractmethod
    def transform_element(self, element: Any) -> Any:
        pass

    def enumerate_elements(self) -> Iterable[Any]:
        for index in range(self.size):
            yield self.index_to_element(index)

    @abstractmethod
    def enumerate_dimension(self, coordinate: int) -> List[int]:
        pass

    def choose_random_element(self) -> Any:
        return self.index_to_element(randrange(self.size))
