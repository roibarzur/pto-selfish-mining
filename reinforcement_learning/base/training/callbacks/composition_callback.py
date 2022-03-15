from typing import Iterable

from .training_callback import TrainingCallback
from ...experience_acquisition.experience import Experience


class CompositionCallback(TrainingCallback):
    def __init__(self, *callbacks: TrainingCallback):
        self.callbacks = callbacks

    def enumerate_callbacks(self) -> Iterable[TrainingCallback]:
        for callback in self.callbacks:
            if isinstance(callback, CompositionCallback):
                for sub_callback in callback.enumerate_callbacks():
                    yield sub_callback
            else:
                yield callback

    def before_running(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.before_running(**kwargs)

    def after_running(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_running(**kwargs)

    def before_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        for callback in self.callbacks:
            callback.before_training_epoch(epoch_idx, **kwargs)

    def after_training_batch(self, batch_idx: int, **kwargs) -> bool:
        results = []
        for callback in self.callbacks:
            result = callback.after_training_batch(batch_idx, **kwargs)
            results.append(result)

        return any(results)

    def after_training_update(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_training_update(**kwargs)

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> bool:
        results = []
        for callback in self.callbacks:
            result = callback.after_training_epoch(epoch_idx, **kwargs)
            results.append(result)

        return any(results)

    def before_episode(self, episode_idx: int, evaluation: bool, **kwargs) -> None:
        for callback in self.callbacks:
            callback.before_episode(episode_idx, evaluation, **kwargs)

    def after_episode_step(self, step_idx: int, exp: Experience, evaluation: bool, **kwargs) -> bool:
        results = []
        for callback in self.callbacks:
            result = callback.after_episode_step(step_idx, exp, evaluation, **kwargs)
            results.append(result)

        return any(results)

    def after_agent_update(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_agent_update(**kwargs)

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_episode(episode_idx, exp, evaluation, **kwargs)
