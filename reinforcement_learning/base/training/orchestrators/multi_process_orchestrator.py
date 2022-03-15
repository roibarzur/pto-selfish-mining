import cProfile
import multiprocessing as mp
import platform
import time
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from typing import List, Dict

import psutil

from blockchain_mdps import BlockchainModel
from .orchestrator import Orchestrator
from ..callbacks.logging.loggers.training_logger import TrainingLogger
from ..callbacks.training_callback import TrainingCallback
from ..rl_algorithm import RLAlgorithm
from ...experience_acquisition.experience import Experience
from ...experience_acquisition.replay_buffers.sequential_replay_buffer import SequentialReplayBuffer
from ...experience_acquisition.replay_buffers.shuffle_replay_buffer import ShuffleReplayBuffer
from ...function_approximation.approximator import Approximator
from ...utility.buffer_synchronizer import BufferSynchronizer
from ...utility.multiprocessing_util import get_process_name, get_process_index


class MultiProcessOrchestrator(Orchestrator):
    _start_method_set = False

    def __init__(self, algorithm: RLAlgorithm, loggers: Dict[str, TrainingLogger], callback: TrainingCallback,
                 blockchain_model: BlockchainModel, number_of_training_agents: int = 5,
                 number_of_evaluation_agents: int = 2, train_episode_length: int = 1000, epoch_size: int = 2000,
                 lower_priority: bool = True, bind_all: bool = False, **kwargs):
        self.lower_priority = lower_priority
        self.bind_all = bind_all
        self.number_of_training_agents = number_of_training_agents
        self.number_of_evaluation_agents = number_of_evaluation_agents

        self.train_episode_length = train_episode_length
        self.epoch_size = epoch_size

        self.sync_manager = None
        self.sync_dict = None
        self.replay_buffer_queue = None
        self.replay_buffer_agent_queue = None

        self.original_affinity = None
        self.processes: List[Process] = []

        super().__init__(algorithm, loggers, callback, blockchain_model, **kwargs)

    def create_replay_buffer(self) -> ShuffleReplayBuffer:
        return ShuffleReplayBuffer(batch_size=self.batch_size)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        del state['sync_manager']
        del state['processes']
        return state

    def before_running(self) -> None:
        if not MultiProcessOrchestrator._start_method_set:
            mp.set_start_method('spawn')
            MultiProcessOrchestrator._start_method_set = True

        self.sync_manager = SyncManager()
        self.sync_manager.start()
        self.initialize_synchronized_state()

        super().before_running()

    def create_callback_initialization_args(self) -> dict:
        return {
            **super().create_callback_initialization_args(),
            'sync_manager': self.sync_manager
        }

    def create_logger_initialization_args(self) -> dict:
        return {
            **super().create_logger_initialization_args(),
            'sync_manager': self.sync_manager
        }

    def initialize_synchronized_state(self) -> None:
        self.sync_dict = self.sync_manager.dict()
        self.sync_dict['training'] = True
        self.sync_dict['approximator'] = self.algorithm.create_approximator()
        self.sync_dict['approximator'].update(self.approximator)

        self.initialize_replay_buffer_synchronization()

        self.processes = []

    def initialize_replay_buffer_synchronization(self) -> None:
        self.replay_buffer_queue = SequentialReplayBuffer(batch_size=self.epoch_size,
                                                          buffer_size=self.replay_buffer_size)
        self.replay_buffer_agent_queue = BufferSynchronizer(self.sync_manager, self.replay_buffer_queue)

    def after_running(self) -> None:
        super().after_running()

        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()

        self.sync_manager.shutdown()

        self.reset_process_priorities()

    def run(self) -> None:
        # Start agents
        for i in range(self.number_of_evaluation_agents):
            name = f'Test Agent {i + 1}'
            p = Process(name=name, target=self.run_episodes_profiler_wrapper, args=(True,))
            p.start()
            self.processes.append(p)

        for i in range(self.number_of_training_agents):
            name = f'Train Agent {i + 1}'
            p = Process(name=name, target=self.run_episodes_profiler_wrapper, args=(False,))
            p.start()
            self.processes.append(p)

        if self.lower_priority:
            self.set_process_priorities()

        if self.output_profile:
            file_name = f'{self.output_dir}/profiles/main.prof'
            cProfile.runctx('self.run_training_epochs()', globals(), locals(), filename=file_name)
        else:
            self.run_training_epochs()

        # Stop agents
        self.sync_dict['training'] = False

        for p in self.processes:
            p.join()

    def run_training_epochs(self) -> None:
        # Train
        for epoch_idx in range(self.num_of_epochs):
            stop = self.train_epoch(epoch_idx)

            if stop:
                break

    def set_process_priorities(self, set_affinity: bool = True) -> None:
        system = platform.system()
        root_process = psutil.Process()
        cpus = list(root_process.cpu_affinity())

        if system == 'Windows':
            priority = psutil.BELOW_NORMAL_PRIORITY_CLASS
        elif system == 'Linux':
            priority = 10
        else:
            priority = None

        if priority is not None:
            for child_process in root_process.children():
                child_process.nice(priority)

        if set_affinity:
            self.original_affinity = root_process.cpu_affinity()
            root_process.cpu_affinity(cpus[-1:])
            for child_idx, child_process in enumerate(root_process.children()):
                child_process.cpu_affinity(cpus[:-1])

    def reset_process_priorities(self) -> None:
        root_process = psutil.Process()

        if self.original_affinity is not None:
            root_process.cpu_affinity(self.original_affinity)

    def run_episodes_profiler_wrapper(self, evaluation: bool) -> None:
        if self.output_profile:
            file_name = f'{self.output_dir}/profiles/{get_process_name()}.prof'
            cProfile.runctx('self.run_episodes(evaluation)', globals(), locals(), filename=file_name)
        else:
            self.run_episodes(evaluation)

    def run_episodes(self, evaluation: bool) -> None:
        if self.random_seed is not None:
            self.initialize_random_seed(self.random_seed + get_process_index()
                                        + self.number_of_training_agents * int(evaluation))

        self.detach_approximators()

        episode_length = self.evaluate_episode_length if evaluation else self.train_episode_length

        episode_idx = 0

        while self.sync_dict['training']:
            self.run_episode(episode_idx, episode_length, evaluation)
            episode_idx += 1

    def detach_approximators(self) -> None:
        self.approximator = self.create_detached_approximator()
        self.agent.approximator = self.create_detached_approximator()

        if self.loss_fn.target_approximator is not None:
            self.loss_fn.target_approximator = self.create_detached_approximator()

    def create_detached_approximator(self) -> Approximator:
        approximator = self.algorithm.create_approximator()
        approximator.update(self.sync_dict['approximator'])

        return approximator

    def run_after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool) -> bool:
        if not evaluation:
            # Save experience from training episodes
            self.replay_buffer_agent_queue.append(exp)

        is_done = super().run_after_episode_step(step_idx, exp, evaluation)
        return is_done or not self.sync_dict['training']

    def update(self) -> None:
        # Update agents
        self.sync_dict['approximator'].update(self.approximator)
        self.update_agent()

        # Update the target approximator in the loss function if exists
        self.loss_fn.update()

        self.callback.after_training_update()

    def update_agent(self) -> None:
        self.agent.update(self.sync_dict['approximator'])
        self.callback.after_agent_update()

    def gather_experience(self) -> bool:
        self.replay_buffer_agent_queue.process(self.batch_size)

        if len(self.replay_buffer) > 0:
            # Already training
            pass
        elif len(self.replay_buffer_queue) < self.epoch_size:
            # Wait for queue to fill up
            time.sleep(0.05)
        else:
            # Fill the replay buffer for training if it is empty
            self.replay_buffer.append(self.replay_buffer_queue.sample())

        return False
