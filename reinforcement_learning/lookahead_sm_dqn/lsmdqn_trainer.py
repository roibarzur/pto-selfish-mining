from blockchain_mdps import BlockchainModel
from .lsmdqn_algorithm import LSMDQNAlgorithm
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class LSMDQNTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        super().__init__(blockchain_model, use_bva=True, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return LSMDQNAlgorithm(**self.creation_args)
