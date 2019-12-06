from typing import Iterable
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import ipdb
torch.manual_seed(10)

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        # batch_first: input of the for (batch_size,seq_length,feature_size)
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.softmax(self.fc(out))
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        Model the policy using an RNN with the sequence o fall past states as the current input
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        
        self.NumNodes = 128
        self.m = 16
        self.num_actions = num_actions
        self.model = RNNModel(input_size=state_dims, output_size=num_actions,
                                 hidden_dim=self.NumNodes, n_layers=2)
        # self.model.float()
        self.state_dims = state_dims
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = alpha, betas=(0.9,0.999))
        self.hidden_state = None

    def __call__(self,s) -> int:
        # TODO: implement this method
        # s is a sequence of past and current states
        seq_len = len(s)
        s = torch.tensor(s).float()
        action_probs,_ = self.model.forward(s.view(-1,seq_len,self.state_dims))
        act_idx = np.random.choice(np.arange(self.num_actions),size=self.m,replace=False,p=action_probs.detach().numpy()[0])
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

        seq_len = len(s)
        s = torch.tensor(s).float()
        predict,self.hidden_state = self.model.forward(s.view(-1,seq_len,self.state_dims))
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
        # s in a sequence (list) of past and current states
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
        pi.hidden_state = None
        while True:
            A.append(pi([S[-1]]))
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
            if t==0:
                pi.update([S[:t]],A[t], gamma**t, delta)
                G0.append(G)
            else:
                pi.update(S[:t],A[t], gamma**t, delta)
            # if t ==0:
            #     G0.append(G)

    return np.array(G0)
