from enum import EnumMeta, IntEnum
from functools import reduce
from operator import mul
from typing import Union, Tuple, List

import numpy as np

from .interval import Interval
from .space import Space


class MultiDimensionalDiscreteSpace(Space):
    def __init__(self, *elements: Union[int, np.array, Tuple[int, int], EnumMeta]):
        if len(elements) == 0:
            raise ValueError('No elements given')

        self.intervals = [Interval(element) for element in elements]

        super().__init__()

    def _calc_size(self) -> int:
        size = reduce(mul, [interval.size for interval in self.intervals], 1)
        return int(size)

    def _calc_dimension(self) -> int:
        return len(self.intervals)

    def element_to_index(self, element: tuple) -> int:
        if len(element) != self.dimension:
            raise ValueError('Bad length of element')

        index = 0
        for coordinate, interval in reversed(list(enumerate(self.intervals))):
            if coordinate < self.dimension - 1:
                index *= interval.size

            if not interval.is_element_inside(element[coordinate]):
                raise ValueError('Bad value', element)

            index += element[coordinate] - interval.boundaries[0]

        return index

    def index_to_element(self, index: int) -> tuple:
        if index < 0 or index >= self.size:
            raise ValueError('Bad index')

        element = []
        for interval in self.intervals:
            item = index % interval.size + interval.boundaries[0]
            element.append(item)
            index //= interval.size

        return self.transform_element(tuple(element))

    def transform_element(self, element: tuple) -> tuple:
        transformed_element = []
        for item, interval in zip(element, self.intervals):
            transformed_element.append(interval.transform_element(item))
        return tuple(transformed_element)

    def enumerate_dimension(self, coordinate: int) -> List[int]:
        if coordinate < 0 or coordinate >= self.dimension:
            raise ValueError('Bad coordinate')

        return self.intervals[coordinate].enumerate()


if __name__ == '__main__':
    class ForkTest(IntEnum):
        Relevant = 0
        Irrelevant = 1

    space = MultiDimensionalDiscreteSpace(ForkTest, (2, 3))

    print('Dimension:', space.dimension)
    print('Space size:', space.size)

    print('State 0:', space.index_to_element(0))
    print('State 1:', space.index_to_element(1))

    s = (ForkTest.Irrelevant, 3)
    i = space.element_to_index(s)
    print('Some state:', s)
    print('Its index:', i)
    print('Its reconstruction', space.index_to_element(i))
