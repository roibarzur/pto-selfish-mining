from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LogNorm

from blockchain_mdps import SimpleModel, PTOSolver, BlockchainModel
from .training_callback import TrainingCallback
from ...experience_acquisition.agents.bva_agent import BVAAgent
from ...experience_acquisition.experience import Experience
from ...utility.multiprocessing_util import get_process_name


class ValueHeatmapCallback(TrainingCallback):
    def __init__(self, plot_true_values: bool = True, plot_agent_policy: bool = True,
                 plot_state_visits: bool = True, plot_agent_values: bool = True) -> None:
        self.agent: Optional[BVAAgent] = None
        self.blockchain_model: Optional[BlockchainModel] = None
        self.output_dir = None

        self.plot_true_values = plot_true_values
        if self.plot_true_values:
            self.true_values = None

        self.plot_agent_policy = plot_agent_policy
        self.plot_state_visits = plot_state_visits
        self.plot_agent_values = plot_agent_values

    def before_running(self, output_dir: str = None, agent: BVAAgent = None, blockchain_model: BlockchainModel = None,
                       **kwargs) -> None:
        super().before_running(**kwargs)

        self.output_dir = str(Path(output_dir).joinpath(Path('q_values_heatmap')))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.agent = agent
        self.blockchain_model = blockchain_model

        if self.plot_true_values:
            expected_horizon = self.agent.simulator.expected_horizon
            solver = PTOSolver(blockchain_model, expected_horizon=expected_horizon)
            _, _, _, true_values = solver.calc_opt_policy()
            self.true_values = torch.tensor(true_values) / expected_horizon

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        heatmaps = []

        first_row = {}

        if self.plot_true_values:
            first_row['True'] = self.true_values, False, None

        state_visits = self.get_state_visits()

        if self.plot_state_visits:
            first_row['State Visits'] = state_visits, True, None

        if self.plot_agent_policy:
            first_row['Agent Policy'] = self.get_agent_policy(), False, list(self.blockchain_model.Action)

        if len(first_row) > 0:
            heatmaps.append(first_row)

        raw_eval = self.agent.get_raw_q_table().max(dim=1).values
        if hasattr(self.agent, 'nn_factor'):
            raw_eval *= self.agent.nn_factor

        eval_diff = raw_eval - self.true_values
        second_row = {
            'Raw NN Eval': (raw_eval, False, None),
            'Eval Diff': (eval_diff, False, None)
        }

        if self.plot_agent_policy:
            second_row['Agent Eval'] = self.agent.reduce_to_v_table(), False, None

        heatmaps.append(second_row)

        self.plot_heatmaps(
            f'{get_process_name()} - Heatmaps Episode {episode_idx} - {float(exp.info["revenue"]):.5}',
            heatmaps)

    def get_agent_policy(self) -> torch.Tensor:
        policy = self.agent.reduce_to_policy()
        return torch.tensor(policy)

    def get_state_visits(self) -> torch.Tensor:
        state_visits_dict = self.agent.state_visits
        state_visits = torch.zeros(self.blockchain_model.state_space.size)

        for state, count in state_visits_dict.items():
            state_index = self.blockchain_model.state_space.element_to_index(state)
            state_visits[state_index] = count

        return state_visits

    def plot_heatmaps(self, name: str,
                      heatmaps_dict_list: List[Dict[str, Tuple[torch.Tensor, bool, Optional[List]]]]) -> None:
        assert isinstance(self.blockchain_model, SimpleModel)

        num_of_rows = len(heatmaps_dict_list)
        num_of_cols = max(len(heatmaps_dict) for heatmaps_dict in heatmaps_dict_list)
        fig, ax = plt.subplots(ncols=num_of_cols, nrows=num_of_rows, squeeze=False,
                               figsize=(6 * num_of_cols, 6 * num_of_rows - 1))

        for row_index, heatmaps_dict in enumerate(heatmaps_dict_list):
            for col_index, (heatmap_name, (values, log_scale, y_labels)) in enumerate(heatmaps_dict.items()):
                value_table = values[1:].reshape(self.blockchain_model.max_fork + 1, -1)
                value_table = value_table.numpy()

                kw_args: Dict[str, Any] = {'norm': LogNorm()} if log_scale else {}

                if y_labels is None:
                    cmap = 'magma'
                else:
                    num_of_labels = len(y_labels)
                    cmap = sns.color_palette("hls", num_of_labels)
                    kw_args['vmin'] = 0
                    kw_args['vmax'] = num_of_labels - 1

                sns.heatmap(value_table, ax=ax[row_index, col_index], cmap=cmap, **kw_args)

                ax[row_index, col_index].set_ylim(0, value_table.shape[0])
                ax[row_index, col_index].set_xlabel('a')
                ax[row_index, col_index].set_ylabel('h')
                ax[row_index, col_index].set_title(heatmap_name)

                if y_labels is not None:
                    num_of_labels = len(y_labels)
                    color_bar = ax[row_index, col_index].collections[0].colorbar
                    color_bar_length = color_bar.vmax - color_bar.vmin
                    color_bar.set_ticks([color_bar.vmin + color_bar_length / num_of_labels * (0.5 + i)
                                         for i in range(num_of_labels)])
                    color_bar.set_ticklabels(y_labels)

        fig.suptitle(name)
        fig.savefig(f'{self.output_dir}/{name}.png')
        fig.show()
