from enum import EnumMeta
from typing import Union, Tuple, List

import numpy as np


class Interval:
    def __init__(self, element: Union[int, np.array, Tuple[int, int], EnumMeta]):
        if isinstance(element, int):
            boundaries = (0, element - 1)

        elif isinstance(element, np.ndarray):
            if len(element) != 2 or not isinstance(element[0], int) or \
                    not isinstance(element[1], int):
                raise ValueError('Bad array given')

            boundaries = (element[0], element[1])

        elif isinstance(element, EnumMeta):
            boundaries = (0, len(list(element)) - 1)

        else:
            boundaries = element

        if boundaries[0] > boundaries[1]:
            raise ValueError('Bad dimensions given')

        if isinstance(element, EnumMeta):
            self.enum = element
        else:
            self.enum = None

        self.boundaries = boundaries
        self.size = self.boundaries[1] - self.boundaries[0] + 1

    def transform_element(self, element: int) -> int:
        if self.enum is not None:
            element = self.enum(element)

        return element

    def enumerate(self) -> List[int]:
        return [self.transform_element(element) for element in range(self.boundaries[0], self.boundaries[1] + 1)]

    def is_element_inside(self, element: int) -> bool:
        return self.boundaries[0] <= element <= self.boundaries[1]
