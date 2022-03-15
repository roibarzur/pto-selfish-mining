from typing import Tuple, Optional, Dict, List

from .logging_callback import LoggingCallback
from ..bva_callback import BVACallback


class BVATensorboardLoggingCallback(LoggingCallback):
    def __init__(self, bva_callback: BVACallback) -> None:
        super().__init__('tensorboard')
        self.bva_callback = bva_callback

    def before_running(self, **kwargs) -> None:
        super().before_running(**kwargs)
        self.logger.log('register_layout', self.create_tensorboard_custom_layout(), True)
        self.logger.log('register_hparams', *self.create_tensorboard_hparams_input())

    @staticmethod
    def create_tensorboard_custom_layout() -> Dict[str, Dict[str, List[str]]]:
        return {'Revenue': {'Base Value Approximation': ['Multiline', ['BVA/Base Value Approximation']]}}

    @staticmethod
    def create_tensorboard_hparams_input() -> Tuple[Optional[dict], dict, Optional[dict]]:
        metric_dict = {'BVA/Base Value Approximation': None}
        return None, metric_dict, None

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        base_value_approximation = self.bva_callback.sync_dict['base_value_approximation']
        self.logger.log('add_scalar', 'BVA/Base Value Approximation', base_value_approximation, epoch_idx)
