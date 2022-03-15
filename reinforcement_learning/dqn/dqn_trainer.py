from blockchain_mdps import BlockchainModel
from .dqn_algorithm import DQNAlgorithm
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class DQNTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        super().__init__(blockchain_model, use_bva=True, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return DQNAlgorithm(**self.creation_args)
