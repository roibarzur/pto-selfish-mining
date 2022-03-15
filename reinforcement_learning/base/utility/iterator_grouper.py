from itertools import zip_longest
from typing import Any, Iterable


def padded_group(iterable: Iterable, size: int, fill_value: Any = None) -> Iterable:
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=fill_value)


def group(iterable: Iterable, max_size: int) -> Iterable:
    for batch in padded_group(iterable, max_size):
        yield tuple([item for item in batch if item is not None])
