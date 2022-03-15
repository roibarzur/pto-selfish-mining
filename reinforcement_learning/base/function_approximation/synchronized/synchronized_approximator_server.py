from multiprocessing import Queue
from multiprocessing.synchronize import Lock
from queue import Empty
from typing import Dict, List, Tuple, Optional

import torch

from blockchain_mdps import BlockchainModel
from ..approximator import Approximator
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator


class SynchronizedApproximatorServer:
    def __init__(self, base_approximator: Approximator, simulator: MDPBlockchainSimulator, sync_dict: dict,
                 request_queue: Queue, response_queue_dict: Dict[str, Queue], batch_synchronization_lock: Lock,
                 batch_size: int = 32, timeout: float = 0.05):
        self.base_approximator = base_approximator
        self.simulator = simulator

        self.batch_size = batch_size
        self.timeout = timeout

        self.training_epoch = 0
        self.sync_dict = sync_dict

        self.request_queue = request_queue
        self.response_queue_dict = response_queue_dict

        self.batch_synchronization_lock = batch_synchronization_lock

        self.cache: Dict[BlockchainModel.State, torch.Tensor] = {}

        self.update()

    def check_epoch(self) -> None:
        if self.sync_dict['training_epoch'] is not None and self.training_epoch < self.sync_dict['training_epoch']:
            self.training_epoch = self.sync_dict['training_epoch']
            self.update()

    def update(self) -> None:
        self.cache = {}
        self.base_approximator.update(self.sync_dict['approximator'])

    def serve(self) -> None:
        while self.sync_dict['training'] or self.sync_dict['agents_running'] > 0:
            with self.batch_synchronization_lock:
                response_addresses, batch = self.collect_batch()
            if response_addresses is not None and batch is not None:
                self.serve_batch(response_addresses, batch)

    def collect_batch(self) -> Tuple[Optional[List[str]], Optional[torch.Tensor]]:
        batch = []
        response_addresses = []
        while len(batch) < self.batch_size:
            try:
                response_address, state = self.request_queue.get(timeout=self.timeout)

                self.check_epoch()

                state_tuple = self.simulator.torch_to_tuple(state)
                if state_tuple in self.cache:
                    # Respond immediately and skip
                    self.respond(response_address, state, self.cache[state_tuple], save=False)
                    continue

                batch.append(state)
                response_addresses.append(response_address)
            except Empty:
                if len(batch) == 0:
                    return None, None
                else:
                    break

        batch = torch.stack(batch)
        return response_addresses, batch

    def serve_batch(self, response_addresses: List[str], batch: torch.Tensor) -> None:
        with torch.no_grad():
            values = self.base_approximator(batch)

        for request_id in range(len(response_addresses)):
            response_address = response_addresses[request_id]
            request = batch[request_id, :]
            response = values[request_id, :].flatten()
            self.respond(response_address, request, response, save=True)

    def respond(self, response_address: str, request: torch.Tensor, response: torch.Tensor, save: bool = True) -> None:
        if save:
            # Save to cache
            self.cache[self.simulator.torch_to_tuple(request)] = response

        # Send response to address
        self.response_queue_dict[response_address].put(response)
