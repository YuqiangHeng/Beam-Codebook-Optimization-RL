# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:55:17 2019

@author: (Ethan) Yuqiang Heng
"""
import numpy as np
from typing import Tuple
from rl.policy import Policy


default_means = np.array([[640,470],[600,460],[680,460],[640,400]])
bs_loc = [641,435]
default_covs = np.array([[[5,0],[0,5]] for i in default_means])
default_arr_rates = np.array([5 for i in default_means])

class GaussianCenters():
    def __init__(self, 
               means:np.array=default_means, #2-d array w/. shape lx2, l centers
               covs:np.array=default_covs, #3-d array w/. shape lx2x2, covariance of each center
               arrival_rates:np.array=default_arr_rates # 1-d lx1 array, arrival rates of UEs at each center
               ):
        assert means.shape[1] == covs.shape[1] == covs.shape[2] == 2
        assert means.shape[0] == covs.shape[0] == arrival_rates.shape[0]
        self.means = means
        self.covs = covs
        self.arrival_rates = arrival_rates
        
    def sample(self) -> Tuple[int, np.array]:
        """
        output:
            n x 2 array, coordinates of n UEs generated according to arrival rates and centers
            assuming poisson arrival at each center
        """
        num_UEs = np.random.poisson(lam = self.arrival_rates).astype(int)
        total_num_UEs = sum(num_UEs)
        all_samples = np.zeros((total_num_UEs,2))
        for i in range(self.arrival_rates.shape[0]):
            samples = np.random.multivariate_normal(self.means[i,:], self.covs[i,:,:], num_UEs[i])
            all_samples[sum(num_UEs[0:i]):sum(num_UEs[0:i+1]),:] = samples
        return total_num_UEs, all_samples
    
if __name__ == "__main__":
    gc = GaussianCenters()
    for i in range(5):
        print(gc.sample()[1].shape)
        

class MaxBoltzmannQMultiBinaryPolicy(Policy):
    """
    A combination of the eps-greedy and Boltzman q-policy.
    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)
    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    Adapted to multibinary action space
    """
    def __init__(self, num_selected = 32, eps=.1, tau=1., clip=(-500., 500.)):
        super(MaxBoltzmannQMultiBinaryPolicy, self).__init__()
        self.eps = eps
        self.tau = tau
        self.clip = clip
        self.num_selected = num_selected

    def select_action(self, q_values):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        action = np.zeros((nb_actions)).astype(int)
        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            active_idx = np.random.choice(a= nb_actions, size = self.num_selected, p=probs)
            action[active_idx] = 1
        else:
            active_idx = q_values.argsort()[-self.num_selected:][::-1]
            action[active_idx] = 1
#            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of MaxBoltzmannQPolicy
        # Returns
            Dict of config
        """
        config = super(MaxBoltzmannQMultiBinaryPolicy, self).get_config()
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        config['num_selected'] = self.num_selected
        return config 