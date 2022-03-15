import logging
import multiprocessing as mp
from pathlib import Path

from .training_logger import TrainingLogger


class TextLogger(TrainingLogger):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.file_name = None
        self.logger = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['logger'] = None
        return state

    def initialize(self, output_dir: str, **kwargs) -> None:
        super().initialize(output_dir, **kwargs)

        self.file_name = str(Path(self.output_dir).joinpath(Path('log.txt')))
        self.logger = self.get_logger()

    def get_logger(self) -> logging.Logger:
        logger = mp.get_logger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.file_name)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        logging_format = '%(asctime)s - %(processName)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
        formatter = logging.Formatter(logging_format)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.handlers = []
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def log(self, info: str) -> None:
        if self.logger is None:
            self.logger = self.get_logger()

        self.logger.info(info)
