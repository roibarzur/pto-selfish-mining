from blockchain_mdps import BlockchainModel
from .lddqn_algorithm import LDDQNAlgorithm
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class LDDQNTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        super().__init__(blockchain_model, use_bva=True, plot_agent_values_heatmap=True, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return LDDQNAlgorithm(**self.creation_args)
