from collections import deque
from multiprocessing.managers import SyncManager
from typing import Any, Optional

from reinforcement_learning.base.training.callbacks.logging.loggers.training_logger import TrainingLogger
from reinforcement_learning.base.utility.buffer_synchronizer import BufferSynchronizer
from reinforcement_learning.base.utility.deque_buffer_wrapper import DequeBufferWrapper


class SynchronizedLogger(TrainingLogger):
    def __init__(self, base_logger: TrainingLogger, buffer_size: int = 5000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_logger = base_logger

        self.own_sync_manager = False
        self.sync_manager = None

        self.writing_buffer = deque(maxlen=buffer_size)
        self.buffer_synchronizer = None

    def __getstate__(self) -> dict:
        # To allow pickling
        state = self.__dict__.copy()
        state['sync_manager'] = None
        return state

    def initialize(self, sync_manager: Optional[SyncManager] = None, **kwargs) -> None:
        if sync_manager is None:
            self.own_sync_manager = True
            self.sync_manager = SyncManager()
            self.sync_manager.start()
        else:
            self.sync_manager = sync_manager

        self.buffer_synchronizer = BufferSynchronizer(self.sync_manager, DequeBufferWrapper(self.writing_buffer))

        self.base_logger.initialize(**kwargs)
        super().initialize(output_dir=self.base_logger.output_dir)

    def log(self, *info: Any) -> None:
        # Add log requests to the buffer
        self.buffer_synchronizer.append(info)

        if self.sync_manager is not None:
            # Running in main process
            # Log everything in the buffer
            self.buffer_synchronizer.process(self.buffer_synchronizer.max_size())
            for request in self.writing_buffer:
                self.base_logger.log(*request)

            self.writing_buffer.clear()

    def dispose(self) -> None:
        self.base_logger.dispose()

        if self.own_sync_manager:
            self.sync_manager.shutdown()
