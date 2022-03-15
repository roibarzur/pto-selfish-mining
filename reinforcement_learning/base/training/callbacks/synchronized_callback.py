from abc import ABC
from multiprocessing.managers import SyncManager
from typing import Optional

from .training_callback import TrainingCallback


class SynchronizedCallback(TrainingCallback, ABC):
    def __init__(self) -> None:
        self.own_sync_manager = False
        self.sync_manager = None
        self.sync_dict = None

    def __getstate__(self) -> dict:
        # To allow pickling
        state = self.__dict__.copy()
        state['sync_manager'] = None
        return state

    def before_running(self, sync_manager: Optional[SyncManager] = None, **kwargs) -> None:
        if sync_manager is None:
            self.own_sync_manager = True
            self.sync_manager = SyncManager()
            self.sync_manager.start()
        else:
            self.sync_manager = sync_manager

        self.sync_dict = self.sync_manager.dict()

    def after_running(self, **kwargs) -> None:
        if self.own_sync_manager:
            self.sync_manager.shutdown()
