import sys
import time

from blockchain_mdps.bitcoin_mdp import BitcoinMDP

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    t = time.time()

    alpha = 0.4
    gamma = 0.5
    max_fork = 50
    horizon = 10**4
    epsilon = 1e-5
    max_iter = 100000000

    print(alpha, gamma, max_fork, horizon, epsilon, max_iter)

    mdp = BitcoinMDP(alpha, gamma, max_fork)
    print(mdp.num_of_states)
    print(alpha)
    p, r, iterations = mdp.calc_opt_policy(expected_horizon=horizon, epsilon=epsilon, max_iter=max_iter)
    print(r)
    rev = mdp.calc_policy_revenue()
    print('Revenue:', rev)

    print('Iterations:', iterations)
    print('Time elapsed:', time.time() - t)


if __name__ == '__main__':
    main()
