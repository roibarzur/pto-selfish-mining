import random
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict

import torch

from blockchain_mdps import BlockchainModel
from ..experience import Experience
from ..exploaration_mechanisms.exploration_mechanism import ExplorationMechanism
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator
from ....base.utility.iterator_grouper import group


class Agent(ABC):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator):
        self.approximator = approximator
        self.simulator = simulator

        self.current_state = torch.tensor([])
        self.legal_actions = torch.tensor([])

        self.last_episode_state_visits = {}
        self.state_visits = {}

        self.step_idx = 0

        self.reset()

    @abstractmethod
    def choose_action(self, explore: bool = True) -> int:
        pass

    def invoke_exploration_mechanism(self, exploration_mechanism: ExplorationMechanism,
                                     action_values: Union[Dict[int, float], torch.Tensor],
                                     explore: bool) -> int:
        current_state = self.simulator.torch_to_tuple(self.current_state)
        if current_state in self.state_visits:
            state_visits = self.state_visits[current_state]
        else:
            state_visits = 0
        return exploration_mechanism.choose_action(action_values, explore, state_visits)

    def random_action(self) -> int:
        legal_actions_indices = torch.nonzero(self.legal_actions, as_tuple=True)[0]
        return random.choice(legal_actions_indices.tolist())

    def step(self, explore: bool = True) -> Experience:
        action = self.choose_action(explore=explore)

        exp = self.simulator.step(action)

        self.current_state = exp.next_state
        self.legal_actions = exp.legal_actions

        self.step_idx += 1
        self.update_state_visit_count()

        return exp

    def update_state_visit_count(self) -> None:
        current_state = self.simulator.torch_to_tuple(self.current_state)
        if current_state not in self.state_visits:
            self.state_visits[current_state] = 0
        self.state_visits[current_state] += 1

    def reset(self, state: Optional[BlockchainModel.State] = None, keep_state: bool = False) -> None:
        if keep_state:
            assert state is None
            state = self.simulator.torch_to_tuple(self.current_state)

        exp = self.simulator.reset(state)
        self.current_state = exp.next_state
        self.legal_actions = exp.legal_actions

        self.step_idx = 0

        self.last_episode_state_visits = self.state_visits
        self.state_visits = {}
        # Make sure the count is empty so this object is picklable

    # noinspection PyUnusedLocal
    def update(self, approximator: Optional[Approximator] = None, **kwargs) -> None:
        # Copy given approximator to agent's approximator
        if approximator is not None:
            self.approximator.update(approximator)
            self.approximator.eval()

    def reduce_to_policy(self) -> BlockchainModel.Policy:
        self.approximator.eval()
        policy = []
        for state in self.simulator.enumerate_states():
            self.reset(state)

            try:
                exp = self.step(explore=False)
                action = exp.action
            except ValueError:
                action = 0

            policy.append(action)

        return tuple(policy)

    def get_raw_q_table(self) -> torch.Tensor:
        self.approximator.eval()
        q_table = torch.zeros((self.simulator.num_of_states, self.simulator.num_of_actions), dtype=torch.float)

        with torch.no_grad():
            for batch in group(enumerate(self.simulator.enumerate_state_tensors()), 100):
                batch_states = tuple(zip(*batch))[1]
                batch_tensor = torch.stack(batch_states, dim=0)
                batch_q_values = self.approximator(batch_tensor)

                for batch_index, (state_index, state) in enumerate(batch):
                    q_values = batch_q_values[batch_index, :self.simulator.num_of_actions]
                    legal_actions = self.simulator.get_state_legal_actions_tensor(state)
                    legal_q_values = q_values.masked_fill_(mask=~legal_actions, value=float('-inf'))
                    q_table[state_index, :] = legal_q_values

            return q_table

    def reduce_to_q_table(self) -> torch.Tensor:
        return self.get_raw_q_table()

    def reduce_to_policy_from_q_table(self, q_table: Optional[torch.Tensor] = None) -> BlockchainModel.Policy:
        if q_table is None:
            q_table = self.reduce_to_q_table()

        policy = torch.argmax(q_table, dim=1)
        return tuple(policy.tolist())
