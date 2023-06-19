from rope_gym import RopeEnv
import gym 
import numpy as np 

env = RopeEnv(False, "Human")
state_0 = env.rope.getState().x

for _ in range(500):
    env.reset()
    # state = env.rope.getState().x
    # print(np.linalg.norm(state - state_0) / np.linalg.norm(state_0))
    # print(state)
    for _ in range(10):
        action = env.action_space.sample()
    #     print(action)
        state, reward, terminate, _ = env.step(action)
    #     print(reward)
        print(state)

    #     if terminate:
    #         break

env.close()