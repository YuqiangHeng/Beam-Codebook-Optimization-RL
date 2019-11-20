# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:55:17 2019

@author: (Ethan) Yuqiang Heng
"""
import numpy as np
from typing import Tuple

class GaussianCenters():
    def _init_(self, 
               means:np.array, #2-d array w/. shape lx2, l centers
               covs:np.array, #3-d array w/. shape lx2x2, covariance of each center
               arrival_rates:np.array # 1-d lx1 array, arrival rates of UEs at each center
               ):
        assert means.shape[1] == covs.shape[1] == covs.shape[2] == 2
        assert means.shape[0] == covs.shapep[0] == arrival_rates.shape[0]
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