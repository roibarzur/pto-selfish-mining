import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Dict

import numpy as np
import torch

from blockchain_mdps import BlockchainModel
from ..callbacks.logging.loggers.training_logger import TrainingLogger
from ..callbacks.training_callback import TrainingCallback
from ..rl_algorithm import RLAlgorithm
from ...experience_acquisition.experience import Experience
from ...experience_acquisition.replay_buffers.replay_buffer import ReplayBuffer


class Orchestrator(ABC):
    Type = Literal['single_process', 'multi_process', 'synced_multi_process']

    def __init__(self, algorithm: RLAlgorithm, loggers: Dict[str, TrainingLogger], callback: TrainingCallback,
                 blockchain_model: BlockchainModel, build_info: Optional[str] = None,
                 output_root: Optional[str] = None, output_profile: bool = False, random_seed: Optional[int] = None,
                 expected_horizon: int = 10_000, replay_buffer_size: int = 5000, epoch_length: int = 10000,
                 evaluate_episode_length: int = 1000, num_of_epochs: int = 1000, batch_size: int = 10,
                 learning_rate: float = 2e-4, weight_decay: float = 0, episode_reset_rate: int = 10,
                 **creation_args) -> None:
        self.algorithm = algorithm
        self.loggers = loggers
        self.callback = callback

        self.experiment_name = None
        self.build_info = build_info

        self.output_root = output_root if output_root is not None else 'logs/'
        self.output_dir = None
        self.output_profile = output_profile

        self.random_seed = random_seed

        self.blockchain_model = blockchain_model
        self.expected_horizon = expected_horizon

        self.replay_buffer_size = replay_buffer_size
        self.epoch_length = epoch_length
        self.evaluate_episode_length = evaluate_episode_length
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.episode_reset_rate = episode_reset_rate

        self.creation_args = creation_args

        self.initialize_random_seed()

        self.replay_buffer = self.create_replay_buffer()

        self.algorithm.initialize()
        self.agent = self.algorithm.agent
        self.approximator = self.algorithm.approximator
        self.loss_fn = self.algorithm.loss_fn
        self.optimizer = self.algorithm.optimizer
        self.lr_scheduler = self.algorithm.lr_scheduler

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    @abstractmethod
    def create_replay_buffer(self) -> ReplayBuffer:
        pass

    def __enter__(self) -> None:
        self.before_running()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.after_running()

    def before_running(self) -> None:
        self.setup_experiment()

        for logger in self.loggers.values():
            logger.initialize(**self.create_logger_initialization_args())

        self.callback.before_running(**self.create_callback_initialization_args())

    def create_logger_initialization_args(self) -> dict:
        return {
            'output_dir': self.output_dir
        }

    def create_callback_initialization_args(self) -> dict:
        return {
            'logger_dict': self.loggers,
            'output_dir': self.output_dir,
            'orchestrator': self,
            'agent': self.agent,
            'blockchain_model': self.blockchain_model
        }

    def after_running(self) -> None:
        self.callback.after_running()

        for logger in self.loggers.values():
            logger.dispose()

    @staticmethod
    def get_current_time() -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    def setup_experiment(self, start_time: Optional[str] = None) -> None:
        start_time = start_time or self.get_current_time()
        build_info = f'_{self.build_info}' if self.build_info is not None else ''
        self.experiment_name = \
            f'{self.blockchain_model}{build_info}_{start_time}'

        dir_path = str(Path(self.output_root).joinpath(Path(self.experiment_name)))
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        self.output_dir = dir_path

        if self.output_profile:
            Path(self.output_dir).joinpath(Path('profiles')).mkdir(parents=True, exist_ok=True)

    def initialize_random_seed(self, nonce: int = 0):
        if self.random_seed is not None:
            seed = self.random_seed + nonce
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @abstractmethod
    def run(self) -> None:
        pass

    def train_epoch(self, epoch_idx: int) -> bool:
        self.callback.before_training_epoch(epoch_idx)

        keep_state = epoch_idx % self.episode_reset_rate != 0
        self.reset(keep_state)
        # self.update_agent()
        self.approximator.train()

        for batch_idx in range(self.epoch_length // self.batch_size):
            while True:
                is_done = self.gather_experience()

                if len(self.replay_buffer) >= self.batch_size:
                    break

            self.optimize()

            is_done = self.callback.after_training_batch(batch_idx) or is_done

            if is_done:
                break

        self.update()
        self.lr_scheduler.step()

        return self.callback.after_training_epoch(epoch_idx)

    @abstractmethod
    def gather_experience(self) -> bool:
        pass

    def optimize(self) -> None:
        # Forward pass
        loss = self.loss_fn(self.replay_buffer.sample())

        # Back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, episode_idx: int, episode_length: int, evaluation: bool) -> Experience:
        exp = None

        self.run_before_episode(episode_idx, evaluation)

        for step_idx in range(episode_length):
            exp = self.agent.step(explore=not evaluation)

            stop_episode = self.run_after_episode_step(step_idx, exp, evaluation)
            if stop_episode:
                break

        self.run_after_episode(episode_idx, exp, evaluation)

        return exp

    def run_before_episode(self, episode_idx: int, evaluation: bool) -> None:
        keep_state = episode_idx % self.episode_reset_rate != 0
        self.agent.reset(keep_state=keep_state)
        self.update_agent()
        self.callback.before_episode(episode_idx, evaluation)

    def run_after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool) -> bool:
        is_done = self.callback.after_episode_step(step_idx, exp, evaluation)
        return exp.is_done or is_done

    def run_after_episode(self, episode_idx: int, exp: Experience, evaluation: bool) -> None:
        self.callback.after_episode(episode_idx, exp, evaluation)

    def update(self) -> None:
        self.update_agent()

        # Update the target approximator in the loss function if exists
        self.loss_fn.update()

        self.callback.after_training_update()

    def update_agent(self) -> None:
        self.agent.update(self.approximator)
        self.callback.after_agent_update()

    def reset(self, keep_state: bool) -> None:
        self.agent.reset(keep_state=keep_state)
        self.replay_buffer.empty()
