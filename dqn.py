# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:39:28 2019

@author: (Ethan) Yuqiang Heng
"""

import numpy as np
import gym
import matplotlib.pyplot as plt

from InitialAccessEnv import InitialAccessEnv
from Agent_Utils import MaxBoltzmannQMultiBinaryPolicy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
#from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Get the environment and extract the number of actions.
env = InitialAccessEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
window_length = 5
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = MaxBoltzmannQMultiBinaryPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-3, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
train_history = dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
train_rewards = env.step_reward_log
plt.plot(train_rewards)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=False)