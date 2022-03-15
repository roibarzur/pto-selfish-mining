from .logging_callback import LoggingCallback
from ..bva_callback import BVACallback
from ....experience_acquisition.agents.bva_agent import BVAAgent
from ....experience_acquisition.experience import Experience


class BVATextLoggingCallback(LoggingCallback):
    def __init__(self, bva_callback: BVACallback) -> None:
        super().__init__('text')
        self.bva_callback = bva_callback
        self.agent = None

    def before_running(self, agent: BVAAgent, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = agent

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        base_value_approximation = self.bva_callback.sync_dict['base_value_approximation']
        self.logger.log(f'Base Value Approximation {base_value_approximation}')

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        base_value_approximation = self.agent.base_value_approximation
        self.logger.log(f'Agent Base Value Approximation {base_value_approximation}')
