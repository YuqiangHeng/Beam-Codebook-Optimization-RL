# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:59:47 2019

@author: (Ethan) Yuqiang Heng
"""
from rl.policy import Policy

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