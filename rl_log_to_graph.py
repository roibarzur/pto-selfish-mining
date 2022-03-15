import os
import re
import sys
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np


def create_performance_figure(save_file_name: str, files: Dict[str, str], opt: Optional[float] = None) -> None:
    results = {}

    for name, file_name in files.items():
        with open(file_name) as file:
            log = file.read()

        values_by_epoch = np.array([float(value) for value in re.findall('base value approximation (.+)', log)])
        values_by_epoch /= 1e4

        values_by_test_episode = np.array([float(value) for value in re.findall(r'Test .*\'revenue\': (\S*),', log)])
        smoothed_values_by_test_episode = np.array([values_by_test_episode[max(n - 9, 0):n + 1].mean()
                                                    for n in range(len(values_by_test_episode))])

        results[name] = (values_by_epoch, smoothed_values_by_test_episode)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlabel('Training epoch')
    ax1.set_ylabel('Revenue (avg of last 100)')
    ax2.set_xlabel('Evaluation episode')
    ax2.set_ylabel('Revenue (avg of 10)')

    x_min_top = 1000
    x_max_top = 0
    x_min_bottom = 1000
    x_max_bottom = 0

    for name, (values_by_epoch, smoothed_values_by_test_episode) in results.items():
        num_of_warmup_epochs = int(np.sum(values_by_epoch == 0))
        x_min_top = min(x_min_top, num_of_warmup_epochs)
        x_max_top = max(x_max_top, len(values_by_epoch))
        ax1.plot(range(num_of_warmup_epochs, len(values_by_epoch)), values_by_epoch[num_of_warmup_epochs:], label=name)

        x_min_bottom = min(x_min_bottom, 0)
        x_max_bottom = max(x_max_bottom, len(smoothed_values_by_test_episode))
        ax2.plot(range(len(smoothed_values_by_test_episode)), smoothed_values_by_test_episode, label=name)

    if opt is not None:
        ax1.hlines(y=opt, xmin=x_min_top, xmax=x_max_top, label='Optimal', linestyles='dashed', colors='k')

        ax2.hlines(y=opt, xmin=x_min_bottom, xmax=x_max_bottom, label='Optimal', linestyles='dashed', colors='k')

    fig.align_ylabels([ax1, ax2])
    fig.subplots_adjust(bottom=0.2)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=len(results) + int(opt is not None))
    fig.show()
    fig.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')


def create_q_values_graph(save_file_name: str, files: Dict[str, str], opt: Optional[float] = None,
                          log_scale: bool = False) -> None:
    results = {}

    for name, file_name in files.items():
        with open(file_name) as file:
            log = file.read()

        q_values = np.array([float(value) for value in re.findall('q_values (.+) -', log)])
        q_values /= 1e4

        results[name] = q_values

    fig, ax = plt.subplots()
    ax.set_xlabel('Training episode')
    ax.set_ylabel('Mean Q value')

    x_min = 1000
    x_max = 0

    for name, q_values in results.items():
        x_min = min(x_min, 0)
        x_max = max(x_max, len(q_values))
        ax.plot(range(len(q_values)), q_values, label=name)

    if opt is not None:
        ax.hlines(y=opt, xmin=x_min, xmax=x_max, label='Optimal', linestyles='dashed', colors='k')

    fig.subplots_adjust(bottom=0.1)

    if log_scale:
        ax.set_yscale('log')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(results) + int(opt is not None))
    fig.show()
    fig.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')


def create_speed_graph(save_file_name: str, files: List[str]) -> None:
    space_state_sizes = []
    epochs_to_converge = []

    for file_name in files:
        with open(file_name) as file:
            log = file.read()

        values_by_epoch = np.array([float(value) for value in re.findall('base value approximation (.+)', log)])
        values_by_epoch /= 1e4

        number_of_states = float(re.findall('Number of states: (.+)', log)[0])

        opt = float(re.findall('Revenue in ARR-MDP: (.+)', log)[0])
        epoch_to_converge = np.amin(np.flatnonzero(values_by_epoch > 0.99 * opt))

        space_state_sizes.append(number_of_states)
        epochs_to_converge.append(epoch_to_converge)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of states')
    ax.set_ylabel('Number of epochs until convergence')

    ax.plot(space_state_sizes, epochs_to_converge)

    fig.show()
    fig.savefig(f'figures/{save_file_name}.png', bbox_inches='tight')


if __name__ == '__main__':
    name = 'scaling'
    file_name = rf'C:\Users\rbrz3\Desktop\ablations\{name}.txt'

    with open(file_name) as file:
        log = file.read()

    values_by_epoch = np.array([float(value) for value in re.findall('- Base Value Approximation (.+)', log)])
    np.savetxt(rf'C:\Users\rbrz3\Desktop\ablations\{name}.dat', values_by_epoch)

    exit()
    if len(sys.argv) < 2:
        print('Need to provide file')
        exit()

    fig_file_name = os.path.basename(sys.argv[1])
    create_performance_figure(fig_file_name, {'test': sys.argv[1]})
