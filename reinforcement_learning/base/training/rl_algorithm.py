from abc import ABC, abstractmethod

import torch

from ..blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ..experience_acquisition.agents.agent import Agent
from ..function_approximation.approximator import Approximator
from ..function_approximation.loss_function import LossFunction


class RLAlgorithm(ABC):
    def __init__(self, simulator: MDPBlockchainSimulator, device: torch.device, **creation_args) -> None:
        self.simulator = simulator
        self.device = device
        self.creation_args = creation_args

        self.agent = None
        self.approximator = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None

    def initialize(self) -> None:
        self.agent = self.create_agent()
        self.approximator = self.create_approximator()
        self.loss_fn = self.create_loss_fn()
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()

    @abstractmethod
    def create_approximator(self) -> Approximator:
        pass

    @abstractmethod
    def create_agent(self) -> Agent:
        pass

    @abstractmethod
    def create_loss_fn(self) -> LossFunction:
        pass

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.approximator.parameters(),
            lr=self.creation_args['learning_rate'],
            weight_decay=self.creation_args.get('weight_decay', 0)
        )

    def create_lr_scheduler(self) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.creation_args.get('lr_decay_factor', 0.1),
                                               step_size=self.creation_args.get('lr_decay_epoch', 1000))
