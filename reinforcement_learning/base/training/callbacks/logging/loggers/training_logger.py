from abc import ABC, abstractmethod
from typing import Any


class TrainingLogger(ABC):
    def __init__(self, **kwargs) -> None:
        self.output_dir = None

    def initialize(self, output_dir: str, **kwargs) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def log(self, *info: Any) -> None:
        pass

    def dispose(self) -> None:
        pass
