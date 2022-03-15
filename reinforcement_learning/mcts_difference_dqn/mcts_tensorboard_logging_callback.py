from typing import Dict, List

import numpy as np

from .mcts_agent import MCTSAgent
from ..base.experience_acquisition.experience import Experience
from ..base.training.callbacks.logging.logging_callback import LoggingCallback
from ..base.training.orchestrators.orchestrator import Orchestrator
from ..base.utility.multiprocessing_util import get_process_name, get_process_index


class MCTSTensorboardLoggingCallback(LoggingCallback):
    def __init__(self, max_num_of_agents: int = 5) -> None:
        super().__init__('tensorboard')
        self.agent = None
        self.orchestrator = None
        self.max_num_of_agents = max_num_of_agents

    def before_running(self, agent: MCTSAgent = None, orchestrator: Orchestrator = None, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = orchestrator.agent
        self.agent = agent

        self.logger.log('register_layout', self.create_tensorboard_custom_layout())

    @staticmethod
    def create_tensorboard_custom_layout() -> Dict[str, Dict[str, List[str]]]:
        return {
            'Target Values': {
                'Mean Target P Value': ['Multiline', ['Replay Buffer/Mean Target P Value']]
            },
            'MC Graph': {
                'Train Graph Size': ['Multiline', ['Train Agent [0-9]+/MC Tree Size']],
                'Test Graph Size': ['Multiline', ['Test Agent [0-9]+/MC Tree Size']],
                'Train Unvisited Nodes': ['Multiline', ['Test Agent [0-9]+/Unvisited Nodes']],
                'Test Unvisited Nodes': ['Multiline', ['Test Agent [0-9]+/Unvisited Nodes']],
            },
            'MC Simulations': {
                'Train Mean Length': ['Multiline', ['Train Agent [0-9]+/Mean MC Simulation Length']],
                'Train Truncated %': ['Multiline', ['Train Agent [0-9]+/Truncated Simulation %']],
                'Test Mean Length': ['Multiline', ['Test Agent [0-9]+/Mean MC Simulation Length']],
                'Test Truncated %': ['Multiline', ['Test Agent [0-9]+/Truncated Simulation %']]
            }
        }

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        # Log replay buffer stats if available
        try:
            target_values = self.orchestrator.replay_buffer.get_all().target_values
            mean_target_values = target_values.mean(dim=0)
            p_target_values = {
                str(self.orchestrator.blockchain_model.action_space.index_to_element(i)):
                    mean_target_values[self.agent.simulator.num_of_actions + i].item()
                for i in range(self.orchestrator.blockchain_model.action_space.size)
            }
            self.logger.log('add_scalars', 'Replay Buffer/Mean Target P Value', p_target_values, epoch_idx)
        except AttributeError:
            # Replay buffer not available
            pass

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if get_process_index() > self.max_num_of_agents:
            # Do nothing
            return

        self.logger.log('add_scalar', f'{get_process_name()}/Mean MC Simulation Length',
                        np.mean(self.agent.mc_trajectory_lengths), episode_idx)

        self.logger.log('add_scalar', f'{get_process_name()}/Truncated Simulation %',
                        100 * np.mean(np.array(self.agent.mc_trajectory_lengths) < self.agent.depth), episode_idx)

        self.logger.log('add_scalar', f'{get_process_name()}/MC Tree Size', len(self.agent.monte_carlo_tree_nodes),
                        episode_idx)

        self.logger.log('add_scalar', f'{get_process_name()}/Unvisited Nodes',
                        sum([node.visit_count == 0 for node in self.agent.monte_carlo_tree_nodes.values()]),
                        episode_idx)
