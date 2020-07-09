import sys
import time

from blockchain_mdp import *
from ethereum.infinite_blockchain_mdp import *
from ethereum.section_counting_blockchain_mdp import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def main():
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    t = time.time()
    #V = [0.237, 0.263, 0.288, 0.313, 0.338, 0.364, 0.389, 0.414, 0.439, 0.465, 0.49]

    alpha = 0.4 #float(sys.argv[1])
    gamma = 0.5 #float(sys.argv[2])
    max_fork = 10
    horizon = 10**1 #float(sys.argv[3])
    epsilon = 1e-2 #float(sys.argv[4])
    max_iter = 100000000 #10000000000000 #int(sys.argv[5])

    print(alpha, gamma, max_fork, horizon, epsilon, max_iter)

    #expected_horizon = 1000
    #ran_mdp = RandomStopBlockchainMDP(alpha, gamma, max_fork, expected_horizon)
    #print(ran_mdp.num_of_states)
    #ran_mdp.calc_opt_policy()
    #ran_mdp.print_policy(print_size = 15)
    #ran_mdp.test_policy(verbose = True)

    #cnt_mdp = BlockCountingBlockchainMDP(alpha, gamma, max_fork, 25)
    mdp = SectionCountingBlockchainMDP(alpha, gamma, max_fork, 1, horizon)
    print(mdp.num_of_states)
    print(alpha)
    #p, r = mdp.calc_opt_policy(epsilon = 1e-5, max_iter = 10000)
    print(str(sys.argv))
    #mdp.build_MDP()
    #print(mdp.P.M[1].todense())
    p, r, iter = mdp.calc_opt_policy(epsilon=epsilon, max_iter=max_iter)
    print(r)
    policy = mdp.translate_to_infinite_policy()
    print('Test policy')

    inf_mdp = InfiniteBlockchainMDP(alpha, gamma, max_fork)
    inf_mdp.print_policy(policy)
    rev = inf_mdp.calc_policy_revenue(policy)
    print('Revenue:', rev)
    #inf_mdp.test_policy(policy, T = 10000, times = 1000, verbose = True)

    print('Iterations:', iter)
    print('Time elapsed:', time.time() - t)

main()
