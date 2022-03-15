import argparse
import io
import re
import signal
import sys
from pathlib import Path
from typing import Tuple, Any

import numpy as np

from blockchain_mdps import *
from reinforcement_learning import *


# noinspection PyUnusedLocal


# from rl_log_to_graph import create_performance_figure, create_q_values_graph, create_speed_graph


# noinspection PyUnusedLocal
def interrupt_handler(signum: int, frame: Any) -> None:
    # For exiting gracefully
    print(f'Process interrupted with signal {signum}')
    raise InterruptedError('Process interrupted')


def solve_mdp_exactly(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    expected_horizon = int(1e4)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    p, r, _, _ = solver.calc_opt_policy(epsilon=1e-7, max_iter=int(1e10))
    sys.stdout.flush()
    return np.float32(solver.mdp.calc_policy_revenue(p)), p


def run_mcts_fees(args: argparse.Namespace):
    experiment_load_log_path = str(
        Path(args.output_root).joinpath(Path(args.load_experiment)).joinpath(Path('log.txt')))

    with io.open(experiment_load_log_path, 'r') as file:
        log = file.read()

    m = re.search(r'blockchain_model: BitcoinFeeModel\((.*?)\),', log)
    alpha, gamma, max_fork, fee, transaction_chance, max_pool = tuple(
        [float(val.strip()) for val in m.group(1).split(',')])
    max_fork = int(max_fork)
    max_pool = int(max_pool)

    # m = re.search(r'blockchain_model: BitcoinModel\((.*?)\),', log)
    # alpha, gamma, max_fork = tuple([float(val.strip()) for val in m.group(1).split(',')])
    # max_fork = int(max_fork)

    if args.delta is not None:
        transaction_chance = args.delta

    mdp = BitcoinFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee, transaction_chance=transaction_chance,
                          max_pool=max_pool)
    # mdp = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)

    simulation_revenues = re.findall(r'Train epoch #(.*?)\n(?:.|\n){1,10000}?Simulated Policy Revenue (.*?)\u00B1', log)
    simulation_revenues = [(float(rev), int(epoch)) for (epoch, rev) in simulation_revenues]
    simulation_revenues = simulation_revenues[len(simulation_revenues) // 2:]

    load_epoch = max(simulation_revenues)[1]

    trainer = MCTSTrainer(mdp, orchestrator_type='synced_multi_process', build_info=args.build_info + '_simulation',
                          output_root=args.output_root, output_profile=False, output_memory_snapshots=False,
                          random_seed=args.seed, expected_horizon=10_000, depth=5, batch_size=1, dropout=0,
                          length_factor=100_000, starting_epsilon=0.05, epsilon_step=0, prune_tree_rate=250,
                          num_of_episodes_for_average=750, learning_rate=2e-5, nn_factor=0.001, mc_simulations=25,
                          num_of_epochs=1, epoch_shuffles=2, save_rate=20, use_base_approximation=True,
                          ground_initial_state=False, train_episode_length=1, evaluate_episode_length=1,
                          number_of_training_agents=args.train_agents, number_of_evaluation_agents=args.eval_agents,
                          lower_priority=args.no_bg, bind_all=args.bind_all, load_experiment=args.load_experiment,
                          load_epoch=load_epoch, load_seed=False, output_value_heatmap=False,
                          normalize_target_values=True, use_cached_values=False, dump_trajectories=True)

    trainer.run()

    '''
    sorted_epochs = sorted(simulation_revenues)
    load_epoch = sorted_epochs[len(sorted_epochs) // 5 * 4][1]
    
    trainer = MCTSTrainer(mdp, orchestrator_type='synced_multi_process',
                          build_info=args.build_info + '_simulation_80',
                          output_root=args.output_root, output_profile=False, output_memory_snapshots=False,
                          random_seed=args.seed, expected_horizon=10_000, depth=5, batch_size=1, dropout=0,
                          length_factor=100_000, starting_epsilon=0.05, epsilon_step=0, prune_tree_rate=250,
                          num_of_episodes_for_average=750, learning_rate=2e-5, nn_factor=0.001, mc_simulations=25,
                          num_of_epochs=1, epoch_shuffles=2, save_rate=20, use_base_approximation=True,
                          ground_initial_state=False, train_episode_length=1, evaluate_episode_length=1,
                          number_of_training_agents=args.train_agents, number_of_evaluation_agents=args.eval_agents,
                          lower_priority=args.no_bg, bind_all=args.bind_all, load_experiment=args.load_experiment,
                          load_epoch=load_epoch, load_seed=False, output_value_heatmap=False,
                          normalize_target_values=True, use_cached_values=False)
    
    trainer.run()
    '''


if __name__ == '__main__':
    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)

    parser = argparse.ArgumentParser(description='Run Deep RL for selfish mining in blockchain')
    parser.add_argument('--build_info', help='build identifier', default=None)
    parser.add_argument('--output_root', help='root destination for all output files', default=None)
    parser.add_argument('--train_agents', help='number of agents spawned to train', default=5, type=int)
    parser.add_argument('--eval_agents', help='number of agents spawned to evaluate', default=2, type=int)
    parser.add_argument('--no_bg', help='don\'t run the process in the background', action='store_false')
    parser.add_argument('--bind_all', help='pass bind_all to tensorboard', action='store_true')
    parser.add_argument('--load_experiment', help='name of experiment to continue', default=None)
    parser.add_argument('--alpha', help='miner size', default=0.35, type=float)
    parser.add_argument('--gamma', help='rushing factor', default=0.5, type=float)
    parser.add_argument('--max_fork', help='maximal fork size', default=10, type=int)
    parser.add_argument('--fee', help='transaction fee', default=10, type=float)
    parser.add_argument('--delta', help='chance for a transaction', type=float)
    parser.add_argument('--seed', help='random seed', default=0, type=int)

    # solve_pt(parser.parse_args())
    run_mcts_fees(parser.parse_args())
