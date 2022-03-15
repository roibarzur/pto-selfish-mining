from enum import EnumMeta, IntEnum
from typing import Union, Tuple, List

import numpy as np

from .interval import Interval
from .space import Space


class DiscreteSpace(Space):
    def __init__(self, element: Union[int, np.array, Tuple[int, int], EnumMeta]):
        self.interval = Interval(element)
        super().__init__()

    def _calc_size(self) -> int:
        return self.interval.size

    def _calc_dimension(self) -> int:
        return 1

    def element_to_index(self, element: int) -> int:
        if not self.interval.is_element_inside(element):
            raise ValueError('Bad value in element', element)

        return int(element) - self.interval.boundaries[0]

    def index_to_element(self, index: int) -> int:
        if index < 0 or index > self.size:
            raise ValueError('Bad index')

        element = self.interval.boundaries[0] + index
        return self.transform_element(element)

    def transform_element(self, element: int) -> int:
        return self.interval.transform_element(element)

    def enumerate_dimension(self, coordinate: int = 0) -> List[int]:
        if coordinate != 0:
            raise ValueError('Bad coordinate')

        return self.interval.enumerate()


if __name__ == '__main__':
    class Fork(IntEnum):
        Relevant = 0
        Irrelevant = 1

    space = DiscreteSpace(Fork)

    print('Dimension:', space.dimension)
    print('Space size:', space.size)

    print('State 0:', space.index_to_element(0))

    s = Fork.Irrelevant
    i = space.element_to_index(s)
    print('Some state:', s)
    print('Its index:', i)
    print('Its reconstruction', space.index_to_element(i))
