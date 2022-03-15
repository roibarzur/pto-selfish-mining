from typing import List, Union, Optional

import torch
import torch.nn as nn

from .approximator import Approximator


# noinspection PyAbstractClass
class MLPApproximator(Approximator):
    def __init__(self, device: torch.device, dim_in: int, dim_out: Union[int, List], hidden_layers_sizes: List[int],
                 dropout: float = 0, bias_in_last_layer: bool = True, normalize_outputs: Optional[List[bool]] = None,
                 use_norm_bias: Optional[List[bool]] = None, momentum: float = 0.99):
        super().__init__(device)
        self.dim_out = [dim_out] if isinstance(dim_out, int) else dim_out
        self.bias_in_last_layer = bias_in_last_layer
        self.normalize_outputs = normalize_outputs if normalize_outputs is not None else [False] * len(self.dim_out)
        self.use_norm_bias = use_norm_bias if use_norm_bias is not None else [True] * len(self.dim_out)
        self.momentum = momentum

        if any(self.normalize_outputs):
            self.norm_layers = nn.ModuleList(nn.BatchNorm1d(1, momentum=1 - self.momentum, affine=use_bias)
                                             for use_bias in self.use_norm_bias)
        else:
            self.norm_layers = None

        self.model = self.build_mlp_model([dim_in] + list(hidden_layers_sizes) + [sum(self.dim_out)], dropout=dropout,
                                          bias_in_last_layer=bias_in_last_layer)

    def build_mlp_model(self, layer_features: List[int], dropout: float = 0, bias_in_last_layer: bool = True
                        ) -> nn.Module:
        layers = []
        for layer_index, layer_in_features, layer_out_features in zip(range(len(layer_features) - 1),
                                                                      layer_features[:-1], layer_features[1:]):
            bias = layer_index < len(layer_features) - 2 or bias_in_last_layer
            linear_layer = nn.Linear(in_features=layer_in_features, out_features=layer_out_features, bias=bias)
            layers.append(linear_layer)

            if layer_index < len(layer_features) - 2:
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(nn.ReLU())

        return nn.Sequential(*layers).to(self.device)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        result = super().forward(state_tensor)

        got_single_dim = result.dim() == 1
        if got_single_dim:
            result = result.unsqueeze(0)

        groups = []

        start_index = 0
        for section_index, section_size in enumerate(self.dim_out):
            # Cut the section out of the result
            end_index = start_index + section_size
            section_result = result[:, start_index:end_index]

            if not self.normalize_outputs[section_index]:
                groups.append(section_result)
            else:
                # Normalize using the matching norm layer
                groups.append(self.norm_layers[section_index](result[:, start_index:end_index].unsqueeze(1))
                              .squeeze(1))

            # Update for next iteration
            start_index += section_size

        # Concatenate the group to a single tensor
        result = torch.cat(groups, dim=1)

        if got_single_dim:
            result = result.squeeze(0)

        return result
