from typing import Any, Optional, Dict, List

import torch.utils.tensorboard
from torch.utils.tensorboard.summary import hparams

from reinforcement_learning.base.training.callbacks.logging.loggers.training_logger import TrainingLogger


class TensorboardLogger(TrainingLogger):
    def __init__(self, flush_secs: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)

        self.flush_secs = flush_secs
        self.tensorboard_writer = None

        self.layout = {}
        self.hparam_dict = {}
        self.metric_dict = {}
        self.hparam_domain_discrete = {}

        self.started_logging = False

    def __getstate__(self) -> dict:
        # To allow pickling
        state = self.__dict__.copy()
        state['tensorboard_writer'] = None
        return state

    def initialize(self, **kwargs) -> None:
        super().initialize(**kwargs)
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(self.output_dir, flush_secs=self.flush_secs)

    def register_layout(self, layout: Dict[str, Dict[str, List]], add_first: bool = False):
        for key in layout.keys():
            if key in self.layout:
                # Combine section
                if add_first:
                    combined = {**layout[key], **self.layout[key]}
                else:
                    combined = {**self.layout[key], **layout[key]}

                self.layout[key] = combined
            else:
                # Add section
                self.layout[key] = layout[key]

    def register_hparams(self, hparam_dict: Optional[dict] = None, metric_dict: Optional[dict] = None,
                         hparam_domain_discrete: Optional[dict] = None):
        if hparam_dict is not None:
            self.hparam_dict = {**self.hparam_dict, **hparam_dict}

        if metric_dict is not None:
            self.metric_dict = {**self.metric_dict, **metric_dict}

        if hparam_domain_discrete is not None:
            self.hparam_domain_discrete = {**self.hparam_domain_discrete, **hparam_domain_discrete}

    def log(self, *info: Any) -> None:
        # Extract request
        method = info[0]
        params = info[1:]

        # Run the given method
        if method == 'register_layout':
            self.register_layout(*params)
        elif method == 'register_hparams':
            self.register_hparams(*params)
        else:
            if not self.started_logging and method != 'add_graph':
                # Run next part on the first real log request only
                # Create custom layout
                self.tensorboard_writer.add_custom_scalars(self.layout)

                # Log hparams
                exp, ssi, sei = hparams(self.hparam_dict, self.metric_dict, self.hparam_domain_discrete)
                self.tensorboard_writer.file_writer.add_summary(exp)
                self.tensorboard_writer.file_writer.add_summary(ssi)
                self.tensorboard_writer.file_writer.add_summary(sei)

                self.started_logging = True

            method_func = getattr(self.tensorboard_writer, method)
            method_func(*params)

    def dispose(self) -> None:
        self.tensorboard_writer.close()
        super().dispose()
