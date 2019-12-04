# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:55:17 2019

@author: (Ethan) Yuqiang Heng
"""
import numpy as np
from typing import Tuple


#time_factor = 20
#cov_factor = 5
#default_means = np.array([[640,470],[600,460],[680,460],[640,400]])
#bs_loc = [641,435]
#default_covs = np.array([[[cov_factor,0],[0,cov_factor]] for i in default_means])
#default_arr_rates = np.array([time_factor for i in default_means])

class GaussianCenters():
    def __init__(self, 
#               means:np.array=default_means, #2-d array w/. shape lx2, l centers
#               covs:np.array=default_covs, #3-d array w/. shape lx2x2, covariance of each center
#               arrival_rates:np.array=default_arr_rates # 1-d lx1 array, arrival rates of UEs at each center
               arrival_rate = 5, cluster_variance = 5
               ):
        self.arrival_rate = arrival_rate
        self.cluster_variance = cluster_variance
#        assert means.shape[1] == covs.shape[1] == covs.shape[2] == 2
#        assert means.shape[0] == covs.shape[0] == arrival_rates.shape[0]
        default_means = np.array([[640,470],[600,460],[680,460],[640,400]])
        bs_loc = [641,435]
        default_covs = np.array([[[self.cluster_variance,0],[0,self.cluster_variance]] for i in default_means])
        default_arr_rates = np.array([arrival_rate for i in default_means])
        self.means = default_means
        self.covs = default_covs
        self.arrival_rates = default_arr_rates
        
    def sample(self) -> Tuple[int, np.array]:
        """
        output:
            n x 2 array, coordinates of n UEs generated according to arrival rates and centers
            assuming poisson arrival at each center
        """
#        num_UEs = np.random.poisson(lam = self.arrival_rates).astype(int)
        num_UEs = np.random.randint(0,self.arrival_rate*2,len(self.arrival_rates)) #uniform arrival rate so that its bounded
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
        

