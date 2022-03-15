from typing import List, Union, Optional

import torch

from ..base.function_approximation.mlp_approximator import MLPApproximator


class MCTSApproximator(MLPApproximator):
    def __init__(self, device: torch.device, dim_in: int, num_of_actions: Union[int, List],
                 hidden_layers_sizes: List[int], dropout: float = 0, bias_in_last_layer: bool = True,
                 use_normalization: Optional[List[bool]] = None, use_norm_bias: Optional[List[bool]] = None,
                 momentum: float = 0.99):
        self.num_of_actions = num_of_actions
        dim_out = 2 * [num_of_actions]
        use_normalization = [use_normalization, False]
        use_norm_bias = [use_norm_bias, False]

        super().__init__(device, dim_in, dim_out, hidden_layers_sizes, dropout, bias_in_last_layer, use_normalization,
                         use_norm_bias, momentum)
