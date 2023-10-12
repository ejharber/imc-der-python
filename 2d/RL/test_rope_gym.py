from rope_gym_RL import RopeEnvRL
import gym 
import numpy as np 

env = RopeEnvRL(True, "Both")

for i in range(10):
    # for i in range(1, 1000):
    env.reset(seed = i)

    for rollout in range(5):
        action = env.action_space.sample()
        env.step(action)
   
env.close()