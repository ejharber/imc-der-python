import sys
sys.path.append("../../gym/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import matplotlib.pyplot as plt

env = RopeEnv(True)
MSE = []
for act in range(1000):

    # get sample action 

    goals = []
    action = env.action_space.sample()

    expected_MSE = []

    while len(goals) < 200:

        print(act, len(goals))
        env_ = RopeEnv(True)
        obs, _, _, _ = env_.step(action)

        if np.all(obs["pos_traj"] == 0): 
            print("failed")
            continue 

        goal = obs["pos_traj"][-2:,-1]
        goals.append(goal)

        goals_ = np.array(goals)
        # print(np.std(goals_, axis=0))
        # goals_ -= np.mean(goals_, axis = 0) 
        expected_MSE.append(np.mean(np.std(goals_, axis=0)))

    plt.plot(expected_MSE)

    goals = np.array(goals)
    goals -= np.mean(goals, axis = 0)

    dist = np.linalg.norm(goals, axis = 1)
    MSE.append(dist)

np.save("analysis/groundtruth", MSE)
plt.show()    

