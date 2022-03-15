from collections import deque
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy import stats

from .synchronized_callback import SynchronizedCallback
from ..orchestrators.synchronized_multi_process_orchestrator import SynchronizedMultiProcessOrchestrator
from ...experience_acquisition.experience import Experience
from ...utility.buffer_synchronizer import BufferSynchronizer
from ...utility.deque_buffer_wrapper import DequeBufferWrapper
from ...utility.multiprocessing_util import get_process_name


class PolicyRevenueCallback(SynchronizedCallback):
    def __init__(self, confidence: Optional[float] = None, long_simulation_rate: Optional[int] = None,
                 length_factor: Optional[float] = None, repeats: Optional[float] = 1,
                 dump_trajectories: Optional[bool] = False):
        super().__init__()
        self.agent = None
        self.orchestrator = None

        self.confidence = confidence if confidence is not None else 0.99
        self.long_simulation_rate = long_simulation_rate if long_simulation_rate is not None else 100
        self.length_factor = length_factor if length_factor is not None else 10
        self.repeats = repeats if repeats is not None else 1
        self.dump_trajectories = dump_trajectories if dump_trajectories is not None else False

        self.policy_revenue = 0
        self.policy_revenue_confidence_radius = 0
        self.num_of_agents = 0

        self.episode_values = None
        self.episode_values_synchronizer = None

        self.policy_test_revenue = 0
        self.policy_test_revenue_confidence_radius = 0
        self.num_of_evaluation_agents = 0

        self.test_episode_values = None
        self.test_episode_values_synchronizer = None

        self.dump_path = ''

    def before_running(self, orchestrator: SynchronizedMultiProcessOrchestrator = None, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = orchestrator.agent
        self.orchestrator = orchestrator
        self.num_of_agents = orchestrator.number_of_training_agents + orchestrator.number_of_evaluation_agents
        self.num_of_evaluation_agents = orchestrator.number_of_evaluation_agents

        self.episode_values = deque(maxlen=self.num_of_agents)
        self.episode_values_synchronizer = BufferSynchronizer(self.sync_manager,
                                                              DequeBufferWrapper(self.episode_values))

        self.test_episode_values = deque(maxlen=self.num_of_evaluation_agents)
        self.test_episode_values_synchronizer = BufferSynchronizer(self.sync_manager,
                                                                   DequeBufferWrapper(self.test_episode_values))

        if self.dump_trajectories:
            self.dump_path = f'{self.orchestrator.output_dir}/out/trajectories'
            Path(self.dump_path).mkdir(parents=True, exist_ok=True)

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> bool:
        if epoch_idx % self.long_simulation_rate == 0:
            self.episode_values_synchronizer.process(self.num_of_agents, wait=True)
            values = np.array(self.episode_values)
            self.policy_revenue, self.policy_revenue_confidence_radius = self.calculate_confidence_interval(values)

        self.test_episode_values_synchronizer.process(self.num_of_evaluation_agents, wait=True)
        values = np.array(self.test_episode_values)
        self.policy_test_revenue, self.policy_test_revenue_confidence_radius = self.calculate_confidence_interval(
            values)

        return False

    def calculate_confidence_interval(self, values: np.array) -> Tuple[float, float]:
        # noinspection PyTypeChecker
        mean: float = np.mean(values, dtype=np.float64)
        std_err = stats.sem(values)
        std_err * stats.t.ppf((1 + self.confidence) / 2, len(values) - 1)

        return mean, std_err

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if evaluation:
            episode_value = exp.info['revenue']
            self.test_episode_values_synchronizer.append(episode_value)

        if episode_idx % self.long_simulation_rate == 0:
            f = open(f'{self.dump_path}/{get_process_name()}_episode_{episode_idx}.txt',
                     'w') if self.dump_trajectories else None

            try:
                self.agent.reset()

                episode_length = int(self.orchestrator.evaluate_episode_length * self.length_factor)
                for step_idx in range(episode_length):
                    exp = self.agent.step(explore=False)
                    if self.dump_trajectories:
                        simulator = self.agent.simulator

                        prev_state = simulator.torch_to_tuple(exp.prev_state)
                        action = simulator.action_index_to_action(exp.action)
                        reward = exp.reward
                        diff = exp.difficulty_contribution
                        
                        f.write(f'{prev_state}\t{action}\t{reward:.2f}\t{diff:.2f}\n')

                episode_value = exp.info['revenue']
                self.episode_values_synchronizer.append(episode_value)

            finally:
                if f is not None:
                    f.close()
