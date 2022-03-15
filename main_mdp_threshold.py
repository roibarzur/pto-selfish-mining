import concurrent.futures as mp
import multiprocessing
from typing import Any

import psutil

import blockchain_mdps


class BinarySearch:
    def __init__(self, margin: float = 1e-3, alpha_epsilon: float = 1e-4, horizon: int = 10_000,
                 mdp_epsilon: float = 1e-5, max_iter: int = 100_000, **kwargs: Any):
        self.margin = margin
        self.alpha_epsilon = alpha_epsilon
        self.horizon = horizon
        self.mdp_epsilon = mdp_epsilon
        self.max_iter = max_iter

        # Fix margin
        self.margin = max(self.margin, self.mdp_epsilon)

        self.kwargs = kwargs

    def model_creator(self, alpha: float) -> blockchain_mdps.BlockchainModel:
        # return BitcoinSimplifiedFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee,
        #                                  transaction_chance=transaction_chance,
        #                                  max_pool=max_fork, max_lead=max_lead, normalize_reward=0)

        # return BitcoinBareFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee,
        #                             transaction_chance=transaction_chance, max_pool=max_fork)

        return BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)

    def run(self) -> float:
        alpha = 0
        low = 0
        high = 0.5

        while high - low > self.alpha_epsilon:
            alpha = (low + high) / 2
            # print(f'Alpha: {alpha}')
            model = self.model_creator(alpha)

            honest = model.get_honest_revenue()

            solver = blockchain_mdps.PTOSolver(model, expected_horizon=self.horizon)
            p, r, iterations, _ = solver.calc_opt_policy(epsilon=self.mdp_epsilon, max_iter=self.max_iter)
            rev = solver.mdp.calc_policy_revenue(p)
            # print(f'Honest: {honest}')
            # print(f'Optimal: {rev}')

            if rev > honest + self.margin:
                high = alpha
            else:
                low = alpha

        return alpha


def run_binary_search(param_name: str, param_value: Any, **kwargs: Any) -> float:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    b = BinarySearch(**{param_name: param_value}, **kwargs)
    print(f'Running {param_name}={param_value}, {kwargs}')
    return b.run()


def main() -> None:
    parameter_sets = {
        'acceptable_path_param': {
            'default_params': {},
            'options': list(range(5, 201, 5))  # [5, 10, 20, 30, 40, 50, 75, 100, 150]
        },
    }
    default_params = {'max_fork': 200, 'margin': 1e-6, 'mdp_epsilon': 1e-6}

    with mp.ProcessPoolExecutor(multiprocessing.cpu_count()) as pool:
        future_lists = {}
        for parameter, set_dict in parameter_sets.items():
            future_lists[parameter] = [pool.submit(run_binary_search, parameter, option,
                                                   **default_params, **set_dict['default_params'])
                                       for option in set_dict['options']]

        print('Submitted')

        for future_list in future_lists.values():
            mp.wait(future_list)

        for parameter, future_list in future_lists.items():
            print(parameter)
            print(parameter_sets[parameter]['options'])
            print([future.result() for future in future_list])


if __name__ == '__main__':
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
    main()
