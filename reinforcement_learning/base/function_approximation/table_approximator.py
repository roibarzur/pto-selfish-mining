import torch
import torch.nn

from blockchain_mdps import BlockchainModel
from .approximator import Approximator
from ... import MDPBlockchainSimulator


# noinspection PyAbstractClass
class TableApproximator(Approximator):
    def __init__(self, device: torch.device, blockchain_model: BlockchainModel, simulator: MDPBlockchainSimulator,
                 weights: torch.Tensor = None):
        super().__init__(device)
        self.model = None
        self.blockchain_model = blockchain_model
        self.simulator = simulator

        if weights is None:
            weights = torch.zeros((simulator.num_of_states, simulator.num_of_actions), device=device)
        else:
            weights = weights.detach().clone().to(device)
        self.weights = torch.nn.Parameter(weights)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # Treat single state as a batch of 1
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.view(1, -1)

        state_indices = torch.zeros((state_tensor.shape[0],), device=self.device, dtype=torch.int64)

        for index_in_batch, state in enumerate(state_tensor):
            state_tuple = self.simulator.torch_to_tuple(state)
            state_index = self.blockchain_model.state_space.element_to_index(state_tuple)
            state_indices[index_in_batch] = state_index

        state_indices = state_indices.view(-1, 1).repeat(1, self.simulator.num_of_actions)

        return torch.gather(self.weights, dim=0, index=state_indices)
