from abc import ABC

from ...experience_acquisition.experience import Experience


class TrainingCallback(ABC):
    def before_running(self, **kwargs) -> None:
        pass

    def after_running(self, **kwargs) -> None:
        pass

    def before_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        pass

    def after_training_batch(self, batch_idx: int, **kwargs) -> bool:
        return False

    def after_training_update(self, **kwargs) -> None:
        pass

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> bool:
        return False

    def before_episode(self, episode_idx: int, evaluation: bool, **kwargs) -> None:
        pass

    def after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool, **kwargs) -> bool:
        return False

    def after_agent_update(self, **kwargs) -> None:
        pass

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        pass
