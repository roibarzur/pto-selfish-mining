from pathlib import Path

from guppy import hpy

from blockchain_mdps import BlockchainModel
from .training_callback import TrainingCallback
from ...experience_acquisition.agents.bva_agent import BVAAgent
from ...experience_acquisition.experience import Experience
from ...utility.multiprocessing_util import get_process_name


class MemorySnapshotCallback(TrainingCallback):
    def __init__(self) -> None:
        self.output_dir = None

    def before_running(self, output_dir: str = None, agent: BVAAgent = None, blockchain_model: BlockchainModel = None,
                       **kwargs) -> None:
        super().before_running(**kwargs)

        self.output_dir = str(Path(output_dir).joinpath(Path('memory_snapshots')))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        self.take_snapshot(f'{get_process_name()}_episode_{episode_idx}_memory_dump')

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> bool:
        self.take_snapshot(f'epoch_{epoch_idx}_memory_dump')

        return False

    def take_snapshot(self, name: str) -> None:
        h = hpy()
        heap = h.heap()
        file_name = str(Path(self.output_dir).joinpath(Path(f'{name}.txt')))
        with open(file_name, 'w+') as f:
            heap.dump(f)
