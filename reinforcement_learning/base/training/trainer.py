from abc import ABC, abstractmethod
from typing import Dict, List, Type, Optional, Any

import torch

from blockchain_mdps import BlockchainModel
from .callbacks.bva_callback import BVACallback
from .callbacks.checkpoint_callback import CheckpointCallback
from .callbacks.composition_callback import CompositionCallback
from .callbacks.logging.bva_tensorboard_logging_callback import BVATensorboardLoggingCallback
from .callbacks.logging.bva_text_logging_callback import BVATextLoggingCallback
from .callbacks.logging.loggers.synchronized_logger import SynchronizedLogger
from .callbacks.logging.loggers.tensorboard_logger import TensorboardLogger
from .callbacks.logging.loggers.text_logger import TextLogger
from .callbacks.logging.loggers.training_logger import TrainingLogger
from .callbacks.logging.policy_revenue_tensorboard_logging_callback import PolicyRevenueTensorboardLoggingCallback
from .callbacks.logging.policy_revenue_text_logging_callback import PolicyRevenueTextLoggingCallback
from .callbacks.logging.tensorboard_logging_callback import TensorboardLoggingCallback
from .callbacks.logging.text_logging_callback import TextLoggingCallback
from .callbacks.memory_snapshot_callback import MemorySnapshotCallback
from .callbacks.policy_revenue_callback import PolicyRevenueCallback
from .callbacks.random_state_jump_callback import RandomStateJumpCallback
from .callbacks.training_callback import TrainingCallback
from .callbacks.value_heatmap_callback import ValueHeatmapCallback
from .orchestrators.multi_process_orchestrator import MultiProcessOrchestrator
from .orchestrators.orchestrator import Orchestrator
from .orchestrators.single_process_orchestrator import SingleProcessOrchestrator
from .orchestrators.synchronized_multi_process_orchestrator import SynchronizedMultiProcessOrchestrator
from .rl_algorithm import RLAlgorithm
from ..blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator


class Trainer(ABC):
    def __init__(self, blockchain_model: BlockchainModel, expected_horizon: int = 10 ** 4,
                 include_transition_info: bool = True, orchestrator_type: Orchestrator.Type = 'single_process',
                 check_valid_states: bool = False, device: torch.device = torch.device('cpu'),
                 output_memory_snapshots: bool = False, log_text: bool = True, log_tensorboard: bool = True,
                 use_bva: bool = False, output_value_heatmap: bool = False, plot_agent_values_heatmap: bool = False,
                 dump_trajectories: bool = False, random_state_jump_rate: int = 0, random_seed: Optional[int] = None,
                 save_rate: int = 2, load_experiment: Optional[str] = None, load_epoch: Optional[int] = None,
                 load_seed: bool = True, **kwargs):
        self.orchestrator_type = orchestrator_type

        self.blockchain_model = blockchain_model
        self.expected_horizon = expected_horizon

        self.check_valid_states = check_valid_states
        self.device = device
        self.include_transition_info = include_transition_info

        self.output_memory_snapshots = output_memory_snapshots

        self.log_text = log_text
        self.log_tensorboard = log_tensorboard
        self.use_bva = use_bva
        self.output_value_heatmap = output_value_heatmap
        self.plot_agent_values_heatmap = plot_agent_values_heatmap
        self.dump_trajectories = dump_trajectories
        self.random_state_jump_rate = random_state_jump_rate
        self.random_seed = random_seed
        self.save_rate = save_rate
        self.load_experiment = load_experiment
        self.load_epoch = load_epoch
        self.load_seed = load_seed

        self.simulator = self.create_simulator()

        self.creation_args: Dict[str, Any] = {
            'device': self.device,
            'simulator': self.simulator,
            **kwargs
        }

        self.loggers = self.create_loggers()
        self.algorithm = self.create_algorithm()
        self.callback = self.create_callback()
        self.orchestrator = self.create_orchestrator()

    def create_simulator(self) -> MDPBlockchainSimulator:
        return MDPBlockchainSimulator(self.blockchain_model, self.expected_horizon,
                                      check_valid_states=self.check_valid_states, device=self.device,
                                      include_transition_info=self.include_transition_info)

    def create_loggers(self) -> Dict[str, TrainingLogger]:
        loggers = {}
        if self.log_text:
            loggers['text'] = TextLogger()

        if self.log_tensorboard:
            if self.orchestrator_type == 'single_process':
                tensorboard_logger = TensorboardLogger()
            else:
                tensorboard_logger = SynchronizedLogger(TensorboardLogger())

            loggers['tensorboard'] = tensorboard_logger

        return loggers

    def log_info(self, info: str) -> None:
        self.loggers['text'].log(info)

    @abstractmethod
    def create_algorithm(self) -> RLAlgorithm:
        pass

    def create_callback(self) -> CompositionCallback:
        callbacks = []

        if self.log_text:
            callbacks.append(TextLoggingCallback())

        if self.log_tensorboard:
            callbacks.append(TensorboardLoggingCallback())

        if self.use_bva:
            bva_callback = BVACallback(
                num_of_episodes_for_average=self.creation_args.get('num_of_episodes_for_average'),
                smart_init=self.creation_args.get('bva_smart_init'),
                stop_goal=self.creation_args.get('stop_goal'))

            callbacks.append(bva_callback)

            if self.log_text:
                callbacks.append(BVATextLoggingCallback(bva_callback))

            if self.log_tensorboard:
                callbacks.append(BVATensorboardLoggingCallback(bva_callback))

        if self.output_value_heatmap:
            callbacks.append(ValueHeatmapCallback(plot_agent_values=self.plot_agent_values_heatmap,
                                                  plot_agent_policy=self.plot_agent_values_heatmap))

        if self.random_state_jump_rate > 0:
            callbacks.append(RandomStateJumpCallback(self.random_state_jump_rate))

        if self.save_rate > 0:
            # noinspection PyUnboundLocalVariable
            bva_callback = bva_callback if self.use_bva else None
            callbacks.append(CheckpointCallback(self.save_rate, load_experiment=self.load_experiment,
                                                load_epoch=self.load_epoch, load_seed=self.load_seed,
                                                bva_callback=bva_callback))

        if self.output_memory_snapshots:
            callbacks.append(MemorySnapshotCallback())

        if self.orchestrator_type == 'synced_multi_process':
            policy_revenue_callback = PolicyRevenueCallback(
                confidence=self.creation_args.get('confidence'),
                long_simulation_rate=self.creation_args.get('long_simulation_rate'),
                length_factor=self.creation_args.get('length_factor'),
                repeats=self.creation_args.get('repeats'),
                dump_trajectories=self.dump_trajectories)

            callbacks.append(policy_revenue_callback)

            if self.log_text:
                callbacks.append(PolicyRevenueTextLoggingCallback(policy_revenue_callback))

            if self.log_tensorboard:
                callbacks.append(PolicyRevenueTensorboardLoggingCallback(policy_revenue_callback))

        return CompositionCallback(*callbacks)

    def create_orchestrator(self) -> Orchestrator:
        type_name_to_class_dict = {
            'single_process': SingleProcessOrchestrator,
            'multi_process': MultiProcessOrchestrator,
            'synced_multi_process': SynchronizedMultiProcessOrchestrator
        }
        orchestrator_type = type_name_to_class_dict[self.orchestrator_type]

        return orchestrator_type(
            algorithm=self.algorithm,
            loggers=self.loggers,
            callback=self.callback,
            blockchain_model=self.blockchain_model,
            expected_horizon=self.expected_horizon,
            random_seed=self.random_seed,
            **self.creation_args
        )

    def get_callbacks_of_type(self, callback_type: Type[TrainingCallback]) -> List:
        return [callback for callback in self.callback.enumerate_callbacks() if isinstance(callback, callback_type)]

    def run(self) -> None:
        with self.orchestrator:
            self.orchestrator.run()
