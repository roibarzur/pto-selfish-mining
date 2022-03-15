import time
from typing import Dict, Optional

import torch

from blockchain_mdps import BlockchainModel
from .multi_process_orchestrator import MultiProcessOrchestrator
from ..callbacks.logging.loggers.training_logger import TrainingLogger
from ..callbacks.training_callback import TrainingCallback
from ..rl_algorithm import RLAlgorithm
from ...experience_acquisition.experience import Experience
from ...experience_acquisition.replay_buffers.sequential_replay_buffer import SequentialReplayBuffer
from ...utility.buffer_synchronizer import BufferSynchronizer


class SynchronizedMultiProcessOrchestrator(MultiProcessOrchestrator):
    def __init__(self, algorithm: RLAlgorithm, loggers: Dict[str, TrainingLogger], callback: TrainingCallback,
                 blockchain_model: BlockchainModel, epoch_shuffles: Optional[int] = 1, **kwargs):
        epoch_size = kwargs['number_of_training_agents'] * kwargs['train_episode_length']
        epoch_length = epoch_size * epoch_shuffles
        replay_buffer_size = 2 * epoch_size
        super().__init__(algorithm, loggers, callback, blockchain_model, epoch_length=epoch_length,
                         replay_buffer_size=replay_buffer_size, **kwargs)

        self.replay_buffer_synchronizer = None
        self.episode_synchronizer = None

    def initialize_synchronized_state(self) -> None:
        super().initialize_synchronized_state()
        self.sync_dict['training_epoch'] = None
        self.episode_synchronizer = BufferSynchronizer(self.sync_manager)

    def initialize_replay_buffer_synchronization(self) -> None:
        self.replay_buffer_agent_queue = SequentialReplayBuffer(batch_size=self.train_episode_length,
                                                                buffer_size=self.train_episode_length)
        self.replay_buffer_synchronizer = BufferSynchronizer(self.sync_manager, self.replay_buffer, sort=True)

    def train_epoch(self, epoch_idx: int) -> bool:
        # Update all agents to start the next episodes
        self.sync_dict['training_epoch'] = epoch_idx

        return super().train_epoch(epoch_idx)

    def gather_experience(self) -> bool:
        if len(self.replay_buffer) > 0:
            # Already training
            pass
        else:
            # Gather data from all agents
            self.replay_buffer_synchronizer.process(self.number_of_training_agents, wait=True)

        return False

    def before_running(self) -> None:
        torch.set_deterministic(True)
        torch.set_num_threads(1)
        super().before_running()

    def run_episodes(self, evaluation: bool) -> None:
        # Disable torch multithreading
        torch.set_deterministic(True)
        torch.set_num_threads(1)

        super().run_episodes(evaluation)

    def run_before_episode(self, episode_idx: int, evaluation: bool) -> None:
        super().run_before_episode(episode_idx, evaluation)

        if episode_idx > 0:
            # All episodes after the first one, need to notify they started before the main loop updates them
            self.episode_synchronizer.append(True)

    def run_after_episode(self, episode_idx: int, exp: Experience, evaluation: bool) -> None:
        while self.sync_dict['training_epoch'] is None or self.sync_dict['training_epoch'] < episode_idx:
            # Wait until the previous epoch finishes
            time.sleep(0.05)

            # Finish if training stopped
            if not self.sync_dict['training']:
                break

        super().run_after_episode(episode_idx, exp, evaluation)

        if not evaluation and self.sync_dict['training']:
            # Transfer all transitions from this episode
            self.replay_buffer_synchronizer.append(self.replay_buffer_agent_queue.sample())

    def update(self) -> None:
        # Make sure all agents started running already
        self.episode_synchronizer.process(self.number_of_training_agents + self.number_of_evaluation_agents, wait=True)

        super().update()

        # Delay the next epoch to make sure the most recent approximator is copied
        time.sleep(0.05)
