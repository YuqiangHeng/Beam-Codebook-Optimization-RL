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
#default_nssb = 32
##IA_SNR_threshold = 8 #min snr for BPSK 80MHZ BW is 8dB, for BPSK 160MHz BW is 11 dB
##n_antenna = 64
##oversample_factor = 2
##
##nseg = int(n_antenna*oversample_factor)
###generate array response vectors
##bins = np.linspace(-np.pi/2,np.pi/2,nseg+1)
###bins = [(i-nseg/2)*2*np.pi/nseg for i in range(nseg+1)]
###bins = [i*2*np.pi/nseg for i in range(nseg+1)]
##bfdirections = [(bins[i]+bins[i+1])/2 for i in range(nseg)]
##
##bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
##codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
##
##for i in range(nseg):
##    phi = bfdirections[i]
##    #array response vector original
##    arr_response_vec = [1j*np.pi*k*np.sin(phi) for k in range(n_antenna)]
##    arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
##    #array response vector for rotated ULA
##    #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
##    codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
#
#all_h = np.load(h_imag_fname)*1j+np.load(h_real_fname)
#
#bf_gains = np.absolute(np.matmul(all_h, np.transpose(np.conj(codebook_all))))**2 #shape n_ue x codebook_size
#all_snr = 30+10*np.log10(bf_gains)-(-94)
#IA_thold_pencentile = 95
#thold_per_ue = np.percentile(all_snr,IA_thold_pencentile,axis=1)

def DFT_codebook(oversampling_factor):
    n_antenna = 64
    nseg = int(n_antenna*oversampling_factor)
    bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    
    for i in range(nseg):
        phi = bfdirections[i]
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all, bfdirections


class InitialAccessEnv(gym.Env):
    
    def __init__(self, oversampling_factor = 1,
               num_beams_possible: int = 32,
               snr_thold_percentil = 100,
               mean_arr_rate = 10,
               cluster_var = 5,
               bandit = False):
        self.refresh = bandit
        self.n_antenna = 64
        self.mean_arr_rate = mean_arr_rate
        self.cluster_var = cluster_var
        self.oversampling_factor = oversampling_factor
        self.codebook_size = int(self.n_antenna*self.oversampling_factor)
        self.snr_thold_percentil = snr_thold_percentil
        self.num_beams_possible = num_beams_possible
        self.action_space = spaces.MultiBinary(self.codebook_size)
        self.observation_space = spaces.MultiDiscrete(np.inf*np.ones(self.codebook_size))
        self.n_ue_per_beam = np.zeros((self.codebook_size))
        self.true_state = np.zeros((self.codebook_size))
        self.gaussian_center = GaussianCenters(n_clusters = 4, arrival_rate = self.mean_arr_rate, cluster_variance = self.cluster_var)
        self.h = np.load(h_real_fname) + 1j*np.load(h_imag_fname)
        self.ue_loc = np.load(ue_loc_fname)
#        self.unique_x = np.unique(self.ue_loc[:,0])
#        self.unique_y = np.unique(self.ue_loc[:,1])
        self.codebook_all, self.bf_directions = DFT_codebook(self.oversampling_factor)
        self.IA_thold = self.calc_IA_thold() 
        self.new_UE_idx = 0
        self.existing_UEs = {}
        self.reachable_UEs_per_beam = {i:[] for i in range(self.codebook_size)}
        self.t = 0
        self.step_reward_log = []
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
    
    def calc_IA_thold(self):
        bf_gains = np.absolute(np.matmul(self.h, np.transpose(np.conj(self.codebook_all))))**2 #shape n_ue x codebook_size
        all_snr = 30+10*np.log10(bf_gains)-(-94)
        thold_per_ue = np.percentile(all_snr,self.snr_thold_percentil,axis=1)
        return thold_per_ue
    
    def step(self, action):
        if self.refresh:
            self.existing_UEs = {}
            self.reachable_UEs_per_beam = {i:[] for i in range(self.codebook_size)}
#        print(self.get_info()["nue_per_beam"])
        ue_idc = self.gen_arriving_ue()
        self.nvalid_ue_per_beam = self.beam_association(ue_idc)
        observation = self.schedule_beam(action)
        self.previous_num_ue_served = sum(observation)
#        delay = self.get_delay() #current delay metric in the system
        reward = sum(observation)
        self.previous_reward = reward
        self.t += 1
        #reward is total number of UEs scheduled
        self.step_reward_log.append(sum(observation))
        return observation, reward, False, {"new_arrival":self.nvalid_ue_per_beam, "nue_per_beam":[len(self.reachable_UEs_per_beam[i]) for i in range(self.codebook_size)]}
    
    def reset(self):
        self.new_UE_idx = 0
        self.existing_UEs = {}
        self.reachable_UEs_per_beam = {i:[] for i in range(self.codebook_size)}
        self.t = 0    
        ue_idc = self.gen_arriving_ue()
        nvalid_ue_per_beam = self.beam_association(ue_idc)
        self.step_reward_log = []
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
        todo:
            schedule UE with the beam with highest RSRP in selected beams
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
    
    
    def get_delay(self):
        """
        return current reward:
            1. sum delay
            2. max delay
        """
        delays = []
        for ue in self.existing_UEs:
            time_enter = self.existing_UEs[ue][0]
            delays.append(self.t - time_enter)
        sum_delay = sum(delays)
        max_delay = max(delays)
        return sum_delay, max_delay
    
    def get_info(self):
#        return {"new_arrival":self.nvalid_ue_per_beam, "nue_per_beam":[len(self.reachable_UEs_per_beam[i]) for i in range(self.codebook_size)]}
        return {"nue_per_beam":[len(self.reachable_UEs_per_beam[i]) for i in range(self.codebook_size)]}
        

import matplotlib.pyplot as plt
from tqdm import tqdm     
if __name__ == "__main__":
    env = InitialAccessEnv(bandit = True)
    nseg = env.codebook_size
    s = env.reset()
    all_ues = np.zeros((nseg))
    for i in tqdm(range(100)):
        all_ues += env.get_info()["nue_per_beam"]
        a_t = np.zeros((nseg)).astype(int)
        s_t, r_t, done, info = env.step(a_t)
        
#    plt.bar(np.arange(nseg), env.get_info()["nue_per_beam"])
#    print(np.count_nonzero(env.get_info()["nue_per_beam"]))
    plt.bar(np.arange(nseg), all_ues)
    print(np.count_nonzero(all_ues))
    
