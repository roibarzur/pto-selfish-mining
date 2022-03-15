from operator import itemgetter
from subprocess import Popen
from typing import Dict, List, Any, Tuple, Optional

from .logging_callback import LoggingCallback
from ....experience_acquisition.agents.agent import Agent
from ....experience_acquisition.experience import Experience
from ....utility.multiprocessing_util import get_process_name, get_process_index


class TensorboardLoggingCallback(LoggingCallback):
    def __init__(self, bind_all: bool = False, max_num_of_agents: int = 5, max_number_of_actions: int = 3) -> None:
        super().__init__('tensorboard')
        self.bind_all = bind_all
        self.num_of_q_values_in_approximator = 0
        self.max_num_of_agents = max_num_of_agents
        self.max_number_of_actions = max_number_of_actions

        self.orchestrator = None
        self.tensorboard_popen = None

    def __getstate__(self) -> dict:
        # To allow pickling
        state = self.__dict__.copy()
        del state['tensorboard_popen']
        return state

    def before_running(self, agent: Agent = None, num_of_q_values_in_approximator: int = None, orchestrator: Any = None,
                       **kwargs) -> None:
        super().before_running(**kwargs)
        self.orchestrator = orchestrator

        if num_of_q_values_in_approximator is None:
            num_of_q_values_in_approximator = agent.simulator.num_of_actions
        self.num_of_q_values_in_approximator = num_of_q_values_in_approximator

        # Must log something to create the event file
        self.logger.log('add_graph', agent.approximator, agent.simulator.tuple_to_torch(agent.simulator.initial_state))
        self.logger.log('register_layout', self.create_tensorboard_custom_layout())
        self.logger.log('register_hparams', *self.create_tensorboard_hparams_input())

        # Open tensorboard in the background
        tensorboard_popen_config = ['tensorboard', f'--logdir={self.logger.output_dir}', '--reload_multifile=True',
                                    '--reload_interval=15']
        if self.bind_all:
            tensorboard_popen_config.append('--bind_all')

        self.tensorboard_popen = Popen(tensorboard_popen_config)

    def create_tensorboard_custom_layout(self) -> Dict[str, Dict[str, List]]:
        try:
            number_of_evaluation_agents = min(self.orchestrator.number_of_evaluation_agents, self.max_num_of_agents)
            number_of_training_agents = min(self.orchestrator.number_of_training_agents, self.max_num_of_agents)
        except AttributeError:
            number_of_evaluation_agents = 0
            number_of_training_agents = 0

        return {
            'Revenue': {
                'Test Revenue': ['Multiline', ['Test Agent [0-9]+/Revenue', 'MainProcess/Revenue']],
                'Train Revenue': ['Multiline', ['Train Agent [0-9]+/Revenue']]
            },
            'Target Values': {
                'Mean Target Q Value': ['Multiline', ['Replay Buffer/Mean Target Q Value']]
            },
            'Test Action Distribution': {
                **{f'Agent {i + 1} ': ['Multiline', [f'Test Agent {i + 1}/Action Probability']]
                   for i in range(number_of_evaluation_agents)},
                **({'Agent': ['Multiline', ['MainProcess/Action Probability']]}
                   if number_of_evaluation_agents == 0 else {})
            },
            'Train Action Distribution': {
                f'Agent {i + 1} ': ['Multiline', [f'Train Agent {i + 1}/Action Probability']]
                for i in range(number_of_training_agents)
            },
            'Performance': {
                'Replay Buffer Size': ['Multiline', ['Replay Buffer/Size']],
                'Test Episode Length': ['Multiline', ['Test Agent [0-9]+/Episode Length',
                                                      'MainProcess/Episode Length']],
                'Train Episode Length': ['Multiline', ['Train Agent [0-9]+/Episode Length']]
            }
        }

    def create_tensorboard_hparams_input(self) -> Tuple[dict, dict, Optional[dict]]:
        hparam_dict = {
            **self.orchestrator.__getstate__(),
            **self.orchestrator.creation_args,
            **self.orchestrator.blockchain_model.__dict__
        }

        # Filter compatible hparam types
        hparam_dict = {name: value for name, value in hparam_dict.items() if type(value) in [bool, float, int]}

        number_of_evaluation_agents = min(self.orchestrator.number_of_evaluation_agents, self.max_num_of_agents)
        try:
            metric_dict = {f'Test Agent #{i}/Revenue': None
                           for i in range(1, 1 + number_of_evaluation_agents)}
        except AttributeError:
            metric_dict = {'MainProcess/Revenue': None}

        return hparam_dict, metric_dict, None

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        buffer_size = len(self.orchestrator.replay_buffer)
        self.logger.log('add_scalar', 'Replay Buffer/Size', buffer_size, epoch_idx)

        # Log replay buffer stats if available
        try:
            target_values = self.orchestrator.replay_buffer.get_all().target_values
            mean_target_values = target_values.mean(dim=0)
            mean_value = mean_target_values[:self.num_of_q_values_in_approximator].mean().item()
            self.logger.log('add_scalar', 'Replay Buffer/Mean Target Q Value', mean_value, epoch_idx)
        except AttributeError:
            # Replay buffer not available
            pass

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if get_process_index() > self.max_num_of_agents:
            # Do nothing
            return

        self.logger.log('add_scalar', f'{get_process_name()}/Revenue', exp.info['revenue'], episode_idx)
        self.logger.log('add_scalar', f'{get_process_name()}/Episode Length', exp.info['length'], episode_idx)

        action_probabilities = {str(action): action_prob for action, action_prob in exp.info['actions'].items()}
        action_probabilities = dict(sorted(action_probabilities.items(), key=itemgetter(1), reverse=True)[
                                    :self.max_number_of_actions])
        action_probabilities['other'] = 1 - sum(action_probabilities.values())

        self.logger.log('add_scalars', f'{get_process_name()}/Action Probability', action_probabilities,
                        episode_idx)

    def after_running(self, **kwargs) -> None:
        self.tensorboard_popen.terminate()
        self.tensorboard_popen.wait(timeout=5)
