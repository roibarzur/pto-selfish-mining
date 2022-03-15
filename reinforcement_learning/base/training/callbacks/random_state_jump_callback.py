from typing import Optional

from .training_callback import TrainingCallback
from ...experience_acquisition.agents.agent import Agent
from ...experience_acquisition.experience import Experience


class RandomStateJumpCallback(TrainingCallback):
    def __init__(self, jump_rate: int = 100):
        self.agent: Optional[Agent] = None
        self.jump_rate = jump_rate

    def before_running(self, agent: Agent = None, **kwargs) -> None:
        super().before_running(**kwargs)
        self.agent = agent

    def after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool, **kwargs) -> bool:
        if not evaluation and step_idx % self.jump_rate == 0:
            state = self.agent.simulator.state_space.choose_random_element()
            self.agent.reset(state)

        return False
