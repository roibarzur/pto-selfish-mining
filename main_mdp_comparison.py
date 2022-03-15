import time

import matplotlib.pyplot as plt
import pandas as pd

from blockchain_mdps import *


def main() -> None:
    revs_df = None
    times_df = None
    iters_df = None

    for max_fork in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        alpha = 0.35
        gamma = 0.0
        epsilon = 1e-5
        max_iter = 100000

        model = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)
        print(model)
        print('Number of states:', model.state_space.size)

        solvers = {
            'PTO': PTOSolver(model, int(1e5)),
            'OSM': OSMSolver(model),
            'COSM': ContinuousOSMSolver(model),
            'SMDP': SMDPSolver(model),
        }

        revs = {'Size': model.state_space.size}
        times = {'Size': model.state_space.size}
        iters = {'Size': model.state_space.size}

        for solver_name, solver in solvers.items():
            t = time.time()
            p, _, iterations, _ = solver.calc_opt_policy(epsilon=epsilon, max_iter=max_iter)
            iters[solver_name] = iterations
            times[solver_name] = time.time() - t

            rev = solver.mdp.calc_policy_revenue(p)
            revs[solver_name] = rev

        # Normalize revenues
        max_rev = max(revs[solver_name] for solver_name in solvers.keys())
        for solver_name in solvers.keys():
            revs[solver_name] /= max_rev

        if revs_df is None:
            revs_df = pd.DataFrame(revs, index=range(1))
        else:
            revs_df = revs_df.append(revs, ignore_index=True)

        if iters_df is None:
            iters_df = pd.DataFrame(iters, index=range(1))
        else:
            iters_df = iters_df.append(iters, ignore_index=True)

        if times_df is None:
            times_df = pd.DataFrame(times, index=range(1))
        else:
            times_df = times_df.append(times, ignore_index=True)

    revs_df.to_csv('out/revs.csv')
    iters_df.to_csv('out/iters.csv')
    times_df.to_csv('out/times.csv')

    revs_df.plot(x='Size')
    plt.ylabel('Revenue')
    plt.savefig('out/revs.png')
    plt.show()

    iters_df.plot(x='Size')
    plt.ylabel('Iterations')
    plt.savefig('out/iters.png')
    plt.show()

    times_df.plot(x='Size')
    plt.ylabel('Time')
    plt.savefig('out/times.png')
    plt.show()


if __name__ == '__main__':
    main()
