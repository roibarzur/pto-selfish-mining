from abc import ABC
from typing import Dict

from .loggers.training_logger import TrainingLogger
from ..training_callback import TrainingCallback


class LoggingCallback(TrainingCallback, ABC):
    def __init__(self, logger_name: str):
        self.logger_name = logger_name
        self.logger = None

    def before_running(self, logger_dict: Dict[str, TrainingLogger], **kwargs) -> None:
        self.logger = logger_dict[self.logger_name]
