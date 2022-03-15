import inspect
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from operator import itemgetter
from pathlib import Path
from queue import Empty, Full
from typing import Any

from reinforcement_learning.base.utility.buffer import Buffer
from reinforcement_learning.base.utility.dummy_buffer import DummyBuffer
from reinforcement_learning.base.utility.multiprocessing_util import get_process_index


class BufferSynchronizer(Buffer):
    def __init__(self, sync_manager: SyncManager, target_buffer: Buffer = None, sort: bool = False,
                 size_multiplier: float = 1):
        self.target_buffer = target_buffer if target_buffer is not None else DummyBuffer()
        self.sort = sort
        self.synchronization_buffer = sync_manager.Queue(int(size_multiplier * self.target_buffer.max_size()))
        self.buffer_full_warn = False

    def __len__(self) -> int:
        return len(self.target_buffer)

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'target_buffer': self.target_buffer}
        return str(d)

    def append(self, element: Any) -> None:
        try:
            object_to_queue = element if not self.sort else (get_process_index(), element)
            self.synchronization_buffer.put_nowait(object_to_queue)

        except Full:
            if not self.buffer_full_warn:
                caller_info = inspect.stack()[1]
                mp.get_logger().warning(f'Synchronization Buffer Full: {self.target_buffer.__class__.__name__} -'
                                        f' {Path(caller_info.filename).name} {caller_info.lineno}')
                self.buffer_full_warn = True

    def empty(self) -> None:
        self.synchronization_buffer.empty()
        self.target_buffer.empty()
        self.buffer_full_warn = False

    def max_size(self) -> int:
        return self.target_buffer.max_size()

    def process(self, num_of_elements: int, wait: bool = False) -> None:
        intermediate_buffer = []
        try:
            for _ in range(num_of_elements):
                if wait:
                    element = self.synchronization_buffer.get()
                else:
                    element = self.synchronization_buffer.get_nowait()
                intermediate_buffer.append(element)
        except Empty:
            pass

        if self.sort:
            intermediate_buffer.sort(key=itemgetter(0))
            intermediate_buffer = [element[1] for element in intermediate_buffer]

        for element in intermediate_buffer:
            self.target_buffer.append(element)
