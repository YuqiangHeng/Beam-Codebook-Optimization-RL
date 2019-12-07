import sys
import numpy as np
import gym
from matplotlib import pyplot as plt

from reinforce_rnn import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN, RNNModel
from InitialAccessEnv import InitialAccessEnv

import ipdb 
def test_reinforce(with_baseline):
    env = env = InitialAccessEnv(bandit = False)
    gamma = 0.97
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

    pi = PiApproximationWithNN(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,1000,pi,B)

if __name__ == "__main__":
    num_iter = 5

    # Test REINFORCE without baseline
    without_baseline = []
    for it in range(num_iter):
        print(it)
        training_progress = test_reinforce(with_baseline=False)
        without_baseline.append(training_progress)
    without_baseline = np.mean(without_baseline,axis=0)

    # Test REINFORCE with baseline
    # with_baseline = []
    # for _ in range(num_iter):
    #     training_progress = test_reinforce(with_baseline=True)
    #     with_baseline.append(training_progress)
    # with_baseline = np.mean(with_baseline,axis=0)

    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(without_baseline)),without_baseline, label='without baseline')
    # ax.plot(np.arange(len(with_baseline)),with_baseline, label='with baseline')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()
    ipdb.set_trace()
