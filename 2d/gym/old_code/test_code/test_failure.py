from rope_gym import RopeEnv
import gym 
import numpy as np 

env = RopeEnv(True)

env.reset(seed = 133)
action = np.load("fail_action.npy")

env.action = action
env.step(0)

env.close()