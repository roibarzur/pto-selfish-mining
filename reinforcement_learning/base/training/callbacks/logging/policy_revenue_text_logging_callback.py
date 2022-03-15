from .logging_callback import LoggingCallback
from ..policy_revenue_callback import PolicyRevenueCallback


class PolicyRevenueTextLoggingCallback(LoggingCallback):
    def __init__(self, policy_revenue_callback: PolicyRevenueCallback) -> None:
        super().__init__('text')
        self.policy_revenue_callback = policy_revenue_callback

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        policy_revenue = self.policy_revenue_callback.policy_test_revenue
        radius = self.policy_revenue_callback.policy_test_revenue_confidence_radius
        self.logger.log(f'Test Policy Revenue {policy_revenue:0.5f}\u00B1{radius:0.5f}')

        if epoch_idx % self.policy_revenue_callback.long_simulation_rate == 0:
            policy_revenue = self.policy_revenue_callback.policy_revenue
            radius = self.policy_revenue_callback.policy_revenue_confidence_radius
            self.logger.log(f'Simulated Policy Revenue {policy_revenue:0.5f}\u00B1{radius:0.5f}')
