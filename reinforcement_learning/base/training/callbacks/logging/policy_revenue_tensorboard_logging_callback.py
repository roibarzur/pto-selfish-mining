from typing import Tuple, Optional, Dict, List

from .logging_callback import LoggingCallback
from ..policy_revenue_callback import PolicyRevenueCallback


class PolicyRevenueTensorboardLoggingCallback(LoggingCallback):
    def __init__(self, policy_revenue_callback: PolicyRevenueCallback) -> None:
        super().__init__('tensorboard')
        self.policy_revenue_callback = policy_revenue_callback

    def before_running(self, **kwargs) -> None:
        super().before_running(**kwargs)
        self.logger.log('register_layout', self.create_tensorboard_custom_layout(), True)
        self.logger.log('register_hparams', *self.create_tensorboard_hparams_input())

    @staticmethod
    def create_tensorboard_custom_layout() -> Dict[str, Dict[str, List[str]]]:
        return {'Revenue': {'Simulated Policy Revenue': ['Multiline', ['Policy Revenue/Simulated Lower Bound',
                                                                       'Policy Revenue/Simulated Revenue',
                                                                       'Policy Revenue/Simulated Upper Bound']],
                            'Test Policy Revenue': ['Multiline', ['Policy Revenue/Test Lower Bound',
                                                                  'Policy Revenue/Test Revenue',
                                                                  'Policy Revenue/Test Upper Bound']]}}

    @staticmethod
    def create_tensorboard_hparams_input() -> Tuple[Optional[dict], dict, Optional[dict]]:
        metric_dict = {'Policy Revenue/Simulated Revenue': None, 'Policy Revenue/Test Revenue': None}
        return None, metric_dict, None

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        policy_revenue = self.policy_revenue_callback.policy_test_revenue
        radius = self.policy_revenue_callback.policy_test_revenue_confidence_radius
        self.logger.log('add_scalar', 'Policy Revenue/Test Revenue', policy_revenue, epoch_idx)
        self.logger.log('add_scalar', 'Policy Revenue/Test Upper Bound', policy_revenue + radius, epoch_idx)
        self.logger.log('add_scalar', 'Policy Revenue/Test Lower Bound', policy_revenue - radius, epoch_idx)

        if epoch_idx % self.policy_revenue_callback.long_simulation_rate == 0:
            policy_revenue = self.policy_revenue_callback.policy_revenue
            radius = self.policy_revenue_callback.policy_revenue_confidence_radius
            self.logger.log('add_scalar', 'Policy Revenue/Simulated Revenue', policy_revenue, epoch_idx)
            self.logger.log('add_scalar', 'Policy Revenue/Simulated Upper Bound', policy_revenue + radius, epoch_idx)
            self.logger.log('add_scalar', 'Policy Revenue/Simulated Lower Bound', policy_revenue - radius, epoch_idx)
