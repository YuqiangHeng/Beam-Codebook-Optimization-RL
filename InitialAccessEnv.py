# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:39:29 2019

@author: (Ethan) Yuqiang Heng
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from BeamRLUtils import GaussianCenters

h_imag_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy"
h_real_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy"
ue_loc_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy"
default_nssb = 32
IA_SNR_threshold = 8 #min snr for BPSK 80MHZ BW is 8dB, for BPSK 160MHz BW is 11 dB
n_antenna = 64
oversample_factor = 4

nseg = int(n_antenna*oversample_factor)
#generate array response vectors
bins = np.linspace(-np.pi/2,np.pi/2,nseg+1)
#bins = [(i-nseg/2)*2*np.pi/nseg for i in range(nseg+1)]
#bins = [i*2*np.pi/nseg for i in range(nseg+1)]
bfdirections = [(bins[i]+bins[i+1])/2 for i in range(nseg)]
codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)

for i in range(nseg):
    phi = bfdirections[i]
    #array response vector original
    arr_response_vec = [1j*np.pi*k*np.sin(phi) for k in range(n_antenna)]
    #array response vector for rotated ULA
    #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
    codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)

all_h = np.load(h_imag_fname)*1j+np.load(h_real_fname)

bf_gains = np.absolute(np.matmul(all_h, np.transpose(codebook_all)))**2 #shape n_ue x codebook_size
all_snr = 30+10*np.log10(bf_gains)-(-94)
IA_thold_pencentile = 95
thold_per_ue = np.percentile(all_snr,IA_thold_pencentile,axis=1)

class InitialAccessEnv(gym.Env):
    
    def __init__(self,
               num_beams_possible: int = default_nssb,
               codebook_size: int = nseg,
               reward_type = "sum_delay",
               snr_threshold = thold_per_ue):
        self.reward_type = reward_type
        self.num_beams_possible = num_beams_possible
        self.codebook_size = codebook_size
        self.action_space = spaces.MultiBinary(codebook_size)
        self.observation_space = spaces.MultiDiscrete(np.inf*np.ones(codebook_size))
        self.n_ue_per_beam = np.zeros((codebook_size))
        self.true_state = np.zeros((codebook_size))
        self.gaussian_center = GaussianCenters()
        self.h = np.load(h_real_fname) + 1j*np.load(h_imag_fname)
        self.ue_loc = np.load(ue_loc_fname)
#        self.unique_x = np.unique(self.ue_loc[:,0])
#        self.unique_y = np.unique(self.ue_loc[:,1])
        self.codebook_all = codebook_all
        self.IA_thold = snr_threshold 
        self.new_UE_idx = 0
        self.existing_UEs = {}
        self.reachable_UEs_per_beam = {i:[] for i in range(self.codebook_size)}
        self.t = 0
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
    
    def step(self, action):
        ue_idc = self.gen_arriving_ue()
        nvalid_ue_per_beam = self.beam_association(ue_idc)
        observation = self.schedule_beam(action)
        self.previous_num_ue_served = sum(observation)
        reward = self.get_reward()
        self.previous_reward = reward
        self.t += 1
        return observation, reward, False, {}
    
    def reset(self):
        self.new_UE_idx = 0
        self.existing_UEs = {}
        self.reachable_UEs_per_beam = {i:[] for i in range(self.codebook_size)}
        self.t = 0    
        ue_idc = self.gen_arriving_ue()
        nvalid_ue_per_beam = self.beam_association(ue_idc)
        return nvalid_ue_per_beam
    
    def render(self, mode='human', close=False):
        ptfmt = "t = %f, previous num UEs served = %f, previous reward = %f"%(self.t, self.previous_num_ue_served, self.previous_reward)
        print(ptfmt)
    
    def beam_association(self, ue_idc:np.array):
        """
        input: 
            ue_h: n_ue x n_antenna array of channel matrices of arriving UEs
        procedure:
            1. find the number of UEs that can achieve IA using each beam in codebook_all -> nvalid_ue_per_beam
            2. store arriving UEs into existing_UEs, dict: ue_idx -> (enter_time, [viable beam indices])
            3. store arriving UEs into reachable_UEs_per_beam, dict: beam_idx -> [IDs of UEs that can achieve IA w/. this beam]
        """
        ue_h = self.h[ue_idc,:] #n_ue x n_antenna channel matrices
        bf_gains = np.absolute(np.matmul(ue_h, np.transpose(self.codebook_all)))**2 #shape n_ue x codebook_size
        all_snr = 30+10*np.log10(bf_gains)-(-94)
#        viable_bf = all_snr >= self.IA_thold
        viable_bf = np.transpose(np.transpose(all_snr) >= self.IA_thold[ue_idc])
        nvalid_ue_per_beam = np.sum(viable_bf, axis=0)
        assert len(nvalid_ue_per_beam) == self.codebook_size
        
        ue_store_idc = np.array([self.new_UE_idx+i for i in range(len(ue_idc))])
        self.new_UE_idx += len(ue_idc)
        for i in range(len(ue_idc)):
            self.existing_UEs[ue_store_idc[i]] = (self.t, np.where(viable_bf[i,:])[0])
        
        for i in range(self.codebook_size):
            self.reachable_UEs_per_beam[i].extend(ue_store_idc[np.where(viable_bf[:,i])])
        return nvalid_ue_per_beam
    
    def schedule_beam(self, action:np.array):
        """
        input:
            action: multi-binary codebook_size x 1 array of indices of selected beams
        procedure:
            remove UEs from self.existing_UEs which can achieve IA w/. any beam in action
        output:
            observation:codebook_size x 1 array of number of UEs served with each selected beam
        """
        observation = np.zeros((self.codebook_size))
        selected_beams = np.where(action)[0]
        for beam in selected_beams:
            scheduled_ues = self.reachable_UEs_per_beam[beam]
            observation[beam] = len(scheduled_ues)
            #debug this: clearing one beam not enough
            self.reachable_UEs_per_beam[beam] = []
            for ue in scheduled_ues:
                enter_time, viable_beams = self.existing_UEs.pop(ue)
                for vbeam in viable_beams:
                    if not vbeam == beam:
                        self.reachable_UEs_per_beam[vbeam].remove(ue)
        return observation
    
    def closest_ue(self, ue_pos:np.array):
        """
        input: 
            ue_loc: lx2 array of x,y coordinates of ues generated from gaussian center
        output:
            lx1 vector of index of ues with ray-traced channels that are closest to the target ues
        """
        #currently calc. l2 distance of all ue data points, can be more efficient
        closest_idx = [np.argmin((self.ue_loc[:,0]-ue_pos[i,0])**2 + (self.ue_loc[:,1]-ue_pos[i,1])**2) for i in range(ue_pos.shape[0])]
        return closest_idx
            
    def gen_arriving_ue(self):
        """
        generate n_ue (from Poisson dist.) new arriving UEs w/. locations (from Gaussian dist.)
        find the closest UE position that has ray-traced channel matrix
        return a n_ue x 1 array of the UE indices
        """
        n_ue, ue_raw_pos = self.gaussian_center.sample() #ue_raw_pos has shape n_ue x 2
        ue_idc = self.closest_ue(ue_raw_pos) #idx of closest ues  
        return ue_idc
    
    
    def get_reward(self):
        """
        return current reward:
            1. sum delay
            2. max delay
        """
        delays = []
        for ue in self.existing_UEs:
            time_enter = self.existing_UEs[ue][0]
            delays.append(self.t - time_enter)
        if self.reward_type == "sum_delay":
            reward = sum(delays)
        elif self.reward_type == "max_delay":
            reward = max(delays)
        return -reward
        

import matplotlib.pyplot as plt
       
if __name__ == "__main__":
    env = InitialAccessEnv()
    s = env.reset()
    for i in range(10):
        a_t = np.zeros((nseg)).astype(int)
        s_t, r_t, done, info = env.step(a_t)
        print("step")
        print([(i, env.existing_UEs[i][0]) for i in env.existing_UEs], r_t)
