# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:02:52 2019
Modified Version of LSBGreedy
https://papers.nips.cc/paper/4445-linear-submodular-bandits-and-their-application-to-diversified-retrieval.pdf
@author: (Ethan) Yuqiang Heng
"""
import gym
import numpy as np
#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#from InitialAccessEnv import InitialAccessEnv
#import matplotlib.pyplot as plt
import time


default_nssb = 32
n_antenna = 64
oversample_factor = 1

nseg = int(n_antenna*oversample_factor)
#generate array response vectors
bins = np.linspace(-np.pi/2,np.pi/2,nseg+1)
#bins = [(i-nseg/2)*2*np.pi/nseg for i in range(nseg+1)]
#bins = [i*2*np.pi/nseg for i in range(nseg+1)]
#bfdirections = [(bins[i]+bins[i+1])/2 for i in range(nseg)]
bfdirections = np.arccos(np.linspace(np.cos(-np.pi/2),np.cos(np.pi/2),nseg))
#bfdirections = np.linspace(-np.pi/2,np.pi/2,nseg)
codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)

for i in range(nseg):
    phi = bfdirections[i]
    #array response vector original
#    arr_response_vec = [1j*np.pi*k*np.sin(phi) for k in range(n_antenna)]
    arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
    #array response vector for rotated ULA
    #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
    codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
num_beams_sel = default_nssb
codebook_size = nseg    

codebook_xcorrelation = abs(np.matmul(codebook_all, np.transpose(np.conj(codebook_all))))**2

#class LSTM(nn.Module):
#
#    def __init__(self, input_size = codebook_size, output_size = codebook_size, hidden_size = 100, num_layers=2):
#        super(LSTM, self).__init__()
#        
#        self.output_size = output_size
#        self.num_layers = num_layers
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        
#        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                            num_layers=num_layers, batch_first=True)
#        
#        self.fc = nn.Linear(hidden_size, output_size)
#
#    def forward(self, x):
#        h_0 = Variable(torch.zeros(
#            self.num_layers, x.size(0), self.hidden_size))
#        
#        c_0 = Variable(torch.zeros(
#            self.num_layers, x.size(0), self.hidden_size))
#        
#        # Propagate input through LSTM
#        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
#        
#        h_out = h_out.view(-1, self.hidden_size)
#        
#        out = self.fc(h_out)
#        
#        return out
    
lambda_t = 1
alpha_t = 1

def calc_pairwise_coorrelation(a, At):
#    pairwise_correlation = [abs(sum(np.multiply(codebook_all[a,:],np.conj(codebook_all[i,:]))))**2 for i in At]
    pairwise_correlation = [codebook_xcorrelation[a,i] for i in At]
    return pairwise_correlation

def incremental_util_vec(a, At, s):
    if len(At) == 0:
        raw_utility_vec = np.zeros(codebook_size)
        pairwise_correlation_all = calc_pairwise_coorrelation(a, np.arange(codebook_size))
        for i in np.arange(codebook_size):
            raw_utility_vec[i] = pairwise_correlation_all[i]*s[i]
        return raw_utility_vec
    else:
        unselected = np.setdiff1d(np.arange(codebook_size),At)
        pairwise_correlation_all = calc_pairwise_coorrelation(a, np.arange(codebook_size))
        raw_utility_vec = np.zeros(codebook_size)
        for i in unselected:
            correlation_i = sum(calc_pairwise_coorrelation(i,At))/len(At)
            raw_utility_vec[i] = pairwise_correlation_all[i]*s[i]*(1-correlation_i)
        return raw_utility_vec

def LSBGreedy(env:gym.Env, num_episodes, num_steps):
    for epi_idx in range(num_episodes):
        env.reset()
#        beam_schedule_time_log = {i:0 for i in range(codebook_size)} 
#        beam_schedule_results_log = {i:0 for i in range(codebook_size)}
        beam_schedule_time_log = np.zeros(codebook_size)
        beam_schedule_results_log = np.zeros(codebook_size)
        mean_arrival_rate_per_beam = 10*np.ones(codebook_size)#optimistic initialization
        num_ue_per_beam = mean_arrival_rate_per_beam
        num_ue_per_beam = env.get_info()["nue_per_beam"]
        Mt = lambda_t*np.eye(codebook_size)
        bt = np.zeros(codebook_size)
        wt = np.matmul(np.linalg.inv(Mt),bt) #may use pseudoinverse for stability
        At = []
        for t in range(num_steps):
            cov_matrix = np.zeros((codebook_size,codebook_size))
            delta_vec = []
            t1 = time.time()
            for l in range(num_beams_sel):
                if l == 0:
                    sel = np.argmax(num_ue_per_beam)
                    delta_t_l = incremental_util_vec(sel, [], num_ue_per_beam)
                else:
                    unselected = np.setdiff1d(np.arange(codebook_size),At)
                    u = np.zeros(len(unselected))
                    c = np.zeros(len(unselected))
                    for unsel_idx in range(len(unselected)):
                        a = unselected[unsel_idx]
                        delta_a_At = incremental_util_vec(a, At, num_ue_per_beam)
                        u[unsel_idx] = np.matmul(wt,delta_a_At)
                        c[unsel_idx] = alpha_t*np.sqrt(np.matmul(np.transpose(delta_a_At),np.matmul(np.linalg.inv(Mt),delta_a_At)))
                    sel_idx = np.argmax(u+c) #select a with highest confidence bound
                    sel = unselected[sel_idx] 
                    delta_t_l = incremental_util_vec(sel,At,num_ue_per_beam)
                delta_vec.append(delta_t_l)
                cov_matrix += np.outer(delta_t_l,delta_t_l)
                At.append(sel)
            t2 = time.time()
#            print('Time per step is %2.2f seconds' %(t2-t1))
            Mt += cov_matrix
            sorted_At = np.sort(At)
            multi_binary_action = np.zeros(codebook_size).astype(int)
            multi_binary_action[At] = 1
            s_t_1, r_t, done, info = env.step(multi_binary_action)
            print('reward is %2.2f' %(r_t))
            for i in range(num_beams_sel):
                bt += s_t_1[At[i]]*delta_vec[i]
            #update estimate of num_ue_per_beam
            for i in At:
                beam_schedule_time_log[i] = t+1
                beam_schedule_results_log[i] += s_t_1[i]
                mean_arrival_rate_per_beam[i] = beam_schedule_results_log[i]/beam_schedule_time_log[i]

#            for i in range(codebook_size):
#                if i in At:
#                    num_ue_per_beam[i] = 0
#                else:
#                    num_ue_per_beam[i] += mean_arrival_rate_per_beam[i]
                    
            num_ue_per_beam = env.get_info()["nue_per_beam"]
            wt = np.matmul(np.linalg.inv(Mt),bt)
            At = []
        
    #select actions 
    
#def LSBGreedy(env: gym.Env):
#    s0 = env.reset() 
#    Mt = np.eye(32)
#    while True:
        
if __name__ == "__main__":
    env = InitialAccessEnv()
    np.random.seed(123)
    env.seed(123)
    LSBGreedy(env,100,300)
        