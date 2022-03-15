from multiprocessing import Queue
from multiprocessing.synchronize import Lock
from typing import Dict

import torch

from ..approximator import Approximator
from ...utility.multiprocessing_util import get_process_name


class SynchronizedClientApproximator(Approximator):
    def __init__(self, request_queue: Queue, response_queue_dict: Dict[str, Queue], agents_counter_lock: Lock,
                 sync_dict: dict, device: torch.device):
        super().__init__(device)
        self.request_queue = request_queue
        self.response_queue_dict = response_queue_dict
        self.agents_counter_lock = agents_counter_lock
        self.sync_dict = sync_dict

        self.process_name = None
        self.response_queue = None

    def register(self) -> None:
        with self.agents_counter_lock:
            self.sync_dict['agents_running'] += 1

    def unregister(self) -> None:
        with self.agents_counter_lock:
            self.sync_dict['agents_running'] -= 1

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if self.process_name is None:
            self.process_name = get_process_name()
            self.response_queue = self.response_queue_dict[self.process_name]

        # Send state request
        request = self.process_name, state_tensor
        self.request_queue.put(request)

        # Get approximator response
        return self.response_queue.get()

    def update(self, approximator: Approximator):
        pass
