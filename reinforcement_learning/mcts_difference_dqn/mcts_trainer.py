from blockchain_mdps import BlockchainModel
from .graph_drawing_callback import GraphDrawingCallback
from .mcts_algorithm import MCTSAlgorithm
from .mcts_tensorboard_logging_callback import MCTSTensorboardLoggingCallback
from ..base.training.callbacks.composition_callback import CompositionCallback
from ..base.training.callbacks.training_callback import TrainingCallback
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class MCTSTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, visualize_every_episode: bool = False,
                 visualize_every_step: bool = False, **kwargs) -> None:
        self.visualize_every_episode = visualize_every_episode
        self.visualize_every_step = visualize_every_step
        super().__init__(blockchain_model, use_bva=True, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return MCTSAlgorithm(blockchain_model=self.blockchain_model, **self.creation_args)

    def create_callback(self) -> TrainingCallback:
        callbacks = [super().create_callback(), MCTSTensorboardLoggingCallback()]

        if self.visualize_every_episode or self.visualize_every_step:
            callbacks.append(GraphDrawingCallback(visualize_every_episode=self.visualize_every_episode,
                                                  visualize_every_step=self.visualize_every_step))

        return CompositionCallback(*callbacks)
