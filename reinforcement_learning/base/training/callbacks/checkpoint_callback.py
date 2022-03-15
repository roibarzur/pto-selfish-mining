import copy
import io
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .bva_callback import BVACallback
from .synchronized_callback import SynchronizedCallback
from ..orchestrators.multi_process_orchestrator import MultiProcessOrchestrator
from ..orchestrators.orchestrator import Orchestrator
from ...experience_acquisition.experience import Experience
from ...utility.multiprocessing_util import get_process_name


class CheckpointCallback(SynchronizedCallback):
    def __init__(self, save_rate: Optional[int] = 2, load_experiment: Optional[str] = None,
                 load_epoch: Optional[int] = None, load_seed: bool = True, bva_callback: Optional[BVACallback] = None):
        super().__init__()
        self.orchestrator: Optional[Orchestrator] = None
        self.output_dir = None

        self.bva_callback = bva_callback
        self.random_seed_dict = None
        self.nn_state_before = None
        self.bva_before = 0
        self.latest_approximator = None

        self.save_rate = save_rate

        self.load_experiment = load_experiment
        self.load_epoch = load_epoch
        self.load_seed = load_seed

    def before_running(self, output_dir: str = None, orchestrator: Orchestrator = None,
                       **kwargs) -> None:
        super().before_running(**kwargs)
        self.orchestrator = orchestrator

        self.output_dir = str(Path(output_dir).joinpath(Path('checkpoints')))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.random_seed_dict = self.sync_manager.dict()

        if self.load_experiment is not None:
            experiment_dir = str(Path(output_dir).parent.joinpath(Path(self.load_experiment)))
            experiment_checkpoints_dir = str(Path(experiment_dir).joinpath(Path('checkpoints')))

            if not os.path.exists(experiment_dir):
                raise FileNotFoundError(f'Experiment not found: {experiment_dir}')

            previous_config_path = str(Path(experiment_dir).joinpath(Path('config.txt')))
            new_path = str(Path(self.output_dir).parent.joinpath(Path('previous_config.txt')))
            shutil.copyfile(previous_config_path, new_path)

            if not os.path.exists(experiment_checkpoints_dir):
                raise FileNotFoundError(f'Experiment checkpoints not found: {experiment_checkpoints_dir}')

            if self.load_epoch is None:
                sub_dirs = next(os.walk(experiment_checkpoints_dir))[1]
                checkpoint_epochs = [int(d) for d in sub_dirs]
                if len(checkpoint_epochs) == 0:
                    raise FileNotFoundError('No checkpoints exist')
                self.load_epoch = max(checkpoint_epochs)

            load_dir = Path(experiment_checkpoints_dir).joinpath(str(self.load_epoch))

            if not os.path.exists(load_dir):
                raise FileNotFoundError('Specified epoch does not exist')

            nn_before_path = str(Path(load_dir).joinpath(Path(f'nn_before.chkpt')))
            if isinstance(self.orchestrator, MultiProcessOrchestrator):
                self.orchestrator.sync_dict['approximator'].load_state_dict(torch.load(nn_before_path))

            nn_path = str(Path(load_dir).joinpath(Path(f'nn.chkpt')))
            self.orchestrator.approximator.load_state_dict(torch.load(nn_path))
            self.latest_approximator = self.orchestrator.algorithm.create_approximator()
            self.latest_approximator.update(self.orchestrator.approximator)

            if self.bva_callback is not None:
                bva_before_path = str(Path(load_dir).joinpath(Path(f'bva_before.chkpt')))
                with io.open(bva_before_path, 'rb') as f:
                    bva_before = pickle.load(f)
                self.bva_callback.sync_dict['base_value_approximation'] = bva_before

                bva_episode_values_path = str(Path(load_dir).joinpath(Path(f'bva_episode_values.chkpt')))
                with io.open(bva_episode_values_path, 'rb') as f:
                    episode_values = pickle.load(f)
                    for value in episode_values:
                        self.bva_callback.episode_values.append(value)

            if self.load_seed:
                random_seeds_path = str(Path(load_dir).joinpath(Path(f'random_seeds.chkpt')))
                with io.open(random_seeds_path, 'rb') as f:
                    random_seed_dict = pickle.load(f)
                    for process, seed in random_seed_dict.items():
                        self.random_seed_dict[process] = seed

            optim_path = str(Path(load_dir).joinpath(Path(f'optim.chkpt')))
            self.orchestrator.optimizer.load_state_dict(torch.load(optim_path))

            loss_fn_path = str(Path(load_dir).joinpath(Path(f'loss_fn.chkpt')))
            self.orchestrator.loss_fn.load_state_dict(torch.load(loss_fn_path))

    def before_episode(self, episode_idx: int, evaluation: bool, **kwargs) -> None:
        if episode_idx == 0 and self.load_experiment is not None:
            if self.load_seed:
                self.set_random_seed()
            if isinstance(self.orchestrator, MultiProcessOrchestrator):
                self.orchestrator.agent.update(self.orchestrator.sync_dict['approximator'])

        elif episode_idx == 1 and self.load_experiment is not None:
            # Update BVA and NN to more recent parameters
            self.orchestrator.agent.update(self.latest_approximator)
            if self.bva_callback is not None:
                self.bva_callback.update_base_value_approximation()
                self.bva_callback.after_agent_update()

    def save_random_seed(self) -> None:
        py_seed = random.getstate()
        np_seed = np.random.get_state()
        torch_seed = torch.get_rng_state()
        self.random_seed_dict[get_process_name()] = py_seed, np_seed, torch_seed

    def set_random_seed(self) -> None:
        py_seed, np_seed, torch_seed = self.random_seed_dict[get_process_name()]
        random.setstate(py_seed)
        np.random.set_state(np_seed)
        torch.set_rng_state(torch_seed)

    def after_episode(self, episode_idx: int, exp: Experience, evaluation: bool, **kwargs) -> None:
        if self.save_rate is not None and episode_idx % self.save_rate == 0:
            self.save_random_seed()

    def before_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        if self.load_experiment is not None and epoch_idx == 0:
            if self.load_seed:
                self.set_random_seed()

        if self.save_rate is None or epoch_idx % self.save_rate != 0:
            return

        self.nn_state_before = copy.deepcopy(self.orchestrator.approximator.state_dict())
        if self.bva_callback is not None:
            self.bva_before = self.bva_callback.sync_dict['base_value_approximation']

    def after_training_epoch(self, epoch_idx: int, **kwargs) -> None:
        if self.save_rate is None or epoch_idx % self.save_rate != 0:
            return

        epoch_dir = str(Path(self.output_dir).joinpath(Path(str(epoch_idx))))
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)

        optim_path = str(Path(epoch_dir).joinpath(Path(f'optim.chkpt')))
        torch.save(self.orchestrator.optimizer.state_dict(), optim_path)

        loss_fn_path = str(Path(epoch_dir).joinpath(Path(f'loss_fn.chkpt')))
        torch.save(self.orchestrator.loss_fn.state_dict(), loss_fn_path)

        nn_before_path = str(Path(epoch_dir).joinpath(Path(f'nn_before.chkpt')))
        torch.save(self.nn_state_before, nn_before_path)

        nn_path = str(Path(epoch_dir).joinpath(Path(f'nn.chkpt')))
        torch.save(self.orchestrator.approximator.state_dict(), nn_path)

        if self.bva_callback is not None:
            bva_before_path = str(Path(epoch_dir).joinpath(Path(f'bva_before.chkpt')))
            with io.open(bva_before_path, 'wb') as f:
                pickle.dump(self.bva_before, f)

            bva_episode_values_path = str(Path(epoch_dir).joinpath(Path(f'bva_episode_values.chkpt')))
            with io.open(bva_episode_values_path, 'wb') as f:
                pickle.dump(self.bva_callback.episode_values, f)

        self.save_random_seed()
        random_seeds_path = str(Path(epoch_dir).joinpath(Path(f'random_seeds.chkpt')))
        with io.open(random_seeds_path, 'wb') as f:
            pickle.dump(self.random_seed_dict.copy(), f)
