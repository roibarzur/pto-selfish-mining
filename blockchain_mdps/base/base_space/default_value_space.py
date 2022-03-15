from typing import List, Any

from .space import Space


class DefaultValueSpace(Space):
    def __init__(self, underlying_space: Space, default_value: Any):
        self.underlying_space = underlying_space
        self.default_value = default_value
        super().__init__()

    def _calc_dimension(self) -> int:
        return self.underlying_space.dimension

    def _calc_size(self) -> int:
        return self.underlying_space.size + 1

    def element_to_index(self, element: Any) -> int:
        if element == self.default_value:
            return 0
        else:
            return self.underlying_space.element_to_index(element) + 1

    def index_to_element(self, index: int) -> Any:
        if index == 0:
            return self.default_value
        else:
            return self.underlying_space.index_to_element(index - 1)

    def transform_element(self, element: Any) -> Any:
        return self.underlying_space.transform_element(element)

    def enumerate_dimension(self, coordinate: int) -> List[int]:
        return self.underlying_space.enumerate_dimension(coordinate)

    def choose_random_element(self) -> Any:
        return self.underlying_space.choose_random_element()
