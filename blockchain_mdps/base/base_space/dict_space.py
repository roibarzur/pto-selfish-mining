from typing import List, Any

from .space import Space

debug = False


def print_dbg(s):
    if debug:
        print(s)


class DictSpace(Space):
    def __init__(self, state2idx, idx2state, default_value, dimension):
        self.state2idx = state2idx
        self.idx2state = idx2state
        self.default_value = default_value
        self.dimension = dimension
        super().__init__()

    def _calc_dimension(self) -> int:
        return self.dimension

    def _calc_size(self) -> int:
        return len(self.state2idx)

    def element_to_index(self, element: Any) -> int:
        try:
            return self.state2idx[element]
        except:
            self.print_all_states()
            raise Exception(f'the state {element} does not exist')

    def index_to_element(self, index: int) -> Any:
        return self.idx2state[index]

    def transform_element(self, element: Any) -> Any:
        return element

    def enumerate_dimension(self, coordinate: int) -> List[int]:
        pass

    def get_all_indices_with(self, a, h, fork):
        indices = []
        for state in self.state2idx.keys():
            if state.a == a and state.h == h and state.fork == fork:
                indices.append(self.state2idx[state])
        return indices

    def print_all_states(self):
        for i in self.idx2state.keys():
            print(f'{i} : {self.idx2state[i]}')
