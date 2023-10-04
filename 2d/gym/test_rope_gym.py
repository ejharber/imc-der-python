from rope_gym import RopeEnv
import gym 
import numpy as np 

env = RopeEnv(True, "Both")


for i in range(10):
    # for i in range(1, 1000):
    env.reset(seed = 0)
        # env.action_space.seed = 1
    # action = env.action_space.sample()
    out = env.step(np.array([[0, 0, np.pi]]).T)
    # print(action)
        # print('step')

    env.reset(seed = 0)
        # env.action_space.seed = 1
    # action = env.action_space.sample()
    out = env.step(np.array([[0, 0, np.pi]]).T)

env.close()