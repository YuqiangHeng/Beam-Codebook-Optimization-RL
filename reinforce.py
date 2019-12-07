from typing import Iterable
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import ipdb
torch.manual_seed(10)
class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        
        self.NumNodes = 128
        self.m = 16
        self.num_actions = num_actions
        self.model = nn.Sequential(
                nn.Linear(state_dims, self.NumNodes),
                nn.ReLU(),
                nn.Linear(self.NumNodes,self.NumNodes),
                nn.ReLU(),
                nn.Linear(self.NumNodes, num_actions),
                nn.Softmax())
        self.model.float()
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = alpha, betas=(0.9,0.999))

    def __call__(self,s) -> int:
        # TODO: implement this method
        action_probs = self.model.forward(torch.tensor(s).float())
        act_idx = np.random.choice(np.arange(self.num_actions),size=self.m,replace=False,p=action_probs.detach().numpy())
        action = np.zeros(self.num_actions)
        action[act_idx] = 1
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        #action_vec_one_hot = np.zeros(self.num_actions)
        #action_vec_one_hot[a] = 1 # * gamma_t*delta
        predict = self.model.forward(torch.tensor(s).float())
        #output_val = torch.tensor([action_vec_one_hot]).float()
        # ipdb.set_trace()
        loss = -torch.sum((torch.log(predict).double())*(torch.tensor(a).double()))
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.model.parameters():
            p.grad *= gamma_t*delta
        self.optimizer.step()
        #self.model.fit(s[np.newaxis,:], action_vec_one_hot[np.newaxis,:], batch_size=1, epochs=1, verbose = 0)

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.NumNodes = 128
        self.model = nn.Sequential(
                nn.Linear(state_dims, self.NumNodes),
                nn.ReLU(),
                nn.Linear(self.NumNodes, self.NumNodes),
                nn.ReLU(),
                nn.Linear(self.NumNodes,1))
        self.model = self.model.float()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr = 0.001, betas=(0.9,0.999))
        self.mse = nn.MSELoss()

    def __call__(self,s):
        # TODO: implement this method
        #return self.model.predict(s[np.newaxis,:])[0][0]
        return self.model.forward(torch.tensor(s).float()).tolist()[0]

    def update(self,s,G):
        # TODO: implement this method
        input_val = s
        predict = self.model.forward(torch.tensor(input_val).float())
        output_val = G
        loss = self.mse(predict, torch.tensor([output_val]).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.model.fit(s_tau[np.newaxis,:], np.array([G])[np.newaxis,:], batch_size=1, epochs=1, verbose =0 )


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    G0 = []
    for i in range(num_episodes):
        S = [env.reset()]
        A = []
        R = [0]
        while True:
            A.append(pi(S[-1]))
            ns, re, done, _ = env.step(A[-1])
            S.append(ns)
            R.append(re)
            if done:
                break
        T = len(A)
        for t in range(T):
            G = sum([gamma**(k - t - 1) *R[k] for k in range(t+1, T+1)])
            delta = G-V(S[t])
            V.update(S[t], G)
            pi.update(S[t],A[t], gamma**t, delta)
            if t ==0:
                G0.append(G)

    return np.array(G0)
