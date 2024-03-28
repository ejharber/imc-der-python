import sys
sys.path.append("../../gym/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import matplotlib.pyplot as plt

env = RopeEnv(True)

actions = []
poses = []

for act in range(200):

    # get sample action 

    goals = []
    action = env.action_space.sample()  

    failure_count = 0

    while len(poses) < 100:

        print(act, len(goals), action)

        env_ = RopeEnv(True)
        env_.reset()
        obs, _, _, _ = env_.step(action)

        if np.all(obs["pos_traj"] == 0): 
            print("failed")
            failure_count += 1
            if failure_count > 10:
                break
            continue 

        goal = obs["pos_traj"][-2:,-1]
        # goals.append(goal)

        # goals_ = np.array(goals)
        # print(np.std(goals_, axis=0))
        # goals_ -= np.mean(goals_, axis = 0) 
        actions.append(action)
        poses.append(goal)
        # expected_MSE.append(np.mean(np.std(goals_, axis=0)))

        print(len(poses))

    # if failure_count

    # plt.plot(expected_MSE)

    # goals = np.array(goals)
    # goals -= np.mean(goals, axis = 0)

    # dist = np.linalg.norm(goals, axis = 1)
    # MSE.append(dist)

np.savez("analysis/groundtruth", actions=actions, poses=poses)
plt.show()    

