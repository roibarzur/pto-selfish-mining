from multiprocessing.managers import SyncManager
from typing import Tuple

from reinforcement_learning.base.function_approximation.approximator import Approximator
from reinforcement_learning.base.function_approximation.synchronized.synchronized_approximator_server import \
    SynchronizedApproximatorServer
from reinforcement_learning.base.function_approximation.synchronized.synchronized_client_approximator import \
    SynchronizedClientApproximator


class SynchronizedApproximatorCreator:
    @staticmethod
    def create(base_approximator: Approximator, sync_manager: SyncManager, sync_dict: dict, creation_args: dict) -> \
            Tuple[SynchronizedApproximatorServer, SynchronizedClientApproximator]:
        agents_counter_lock = sync_manager.Lock()
        sync_dict['agents_running'] = 0

        approximator_request_queue = sync_manager.Queue(creation_args['replay_buffer_size'])

        agent_process_names = [f'Test Agent {i + 1}' for i in range(creation_args['number_of_evaluation_agents'])]
        agent_process_names += [f'Train Agent {i + 1}' for i in range(creation_args['number_of_training_agents'])]

        approximator_response_queue_dict = {}
        for name in agent_process_names:
            approximator_response_queue_dict[name] = sync_manager.Queue(creation_args['batch_size'])

        approximator_server_batch_lock = sync_manager.Lock()

        approximator_server = SynchronizedApproximatorServer(
            base_approximator=base_approximator,
            simulator=creation_args['simulator'],
            sync_dict=sync_dict,
            request_queue=approximator_request_queue,
            response_queue_dict=approximator_response_queue_dict,
            batch_synchronization_lock=approximator_server_batch_lock
        )

        client_approximator = SynchronizedClientApproximator(approximator_request_queue,
                                                             approximator_response_queue_dict,
                                                             agents_counter_lock, sync_dict=sync_dict,
                                                             device=creation_args['device'])

        return approximator_server, client_approximator
