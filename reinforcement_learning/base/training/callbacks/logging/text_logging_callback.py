import io
from pathlib import Path

from beeprint import pp

from .logging_callback import LoggingCallback
from ....experience_acquisition.experience import Experience


class TextLoggingCallback(LoggingCallback):
    def __init__(self) -> None:
        super().__init__('text')
        self.orchestrator = None
        self.output_dir = None

    def before_running(self, orchestrator: object, output_dir: str = None, **kwargs) -> None:
        super().before_running(**kwargs)
        self.orchestrator = orchestrator
        self.output_dir = output_dir

        trainer_parameters = pp(orchestrator, output=False, hide_attr_by_prefixes=["_", "func_"], max_depth=10)

        self.logger.log(f'Starting to train trainer:\n{trainer_parameters}')

        config_path = str(Path(self.output_dir).joinpath(Path(f'config.txt')))
        with io.open(config_path, 'w') as f:
            f.write(trainer_parameters)

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        self.logger.log(f'Train epoch #{epoch_idx}')

        # Log replay buffer stats if available
        try:
            target_values = self.orchestrator.replay_buffer.get_all().target_values
            if len(target_values) > 0:
                self.logger.log(f'Target Values {target_values.mean(dim=0)} \u00b1 {target_values.std(dim=0)}'
                                f' ({len(target_values)})')
            else:
                self.logger.log('Replay buffer empty')
        except AttributeError:
            # Replay buffer not available
            pass

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        episode_type = 'Test' if evaluation else 'Train'
        self.logger.log(f'{episode_type} Episode #{episode_idx}: {exp.info}')

        try:
            state_value_cache = self.orchestrator.agent.state_value_cache
            values = state_value_cache.values()
            if len(values) > 0:
                self.logger.log(f'Q Values {sum(values) / len(values)} ({len(state_value_cache)})')
            else:
                self.logger.log('State value cache empty')
        except AttributeError:
            # Value cache not available
            pass
