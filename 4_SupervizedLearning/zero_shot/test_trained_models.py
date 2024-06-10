import sys

import gym 
import numpy as np 
import matplotlib.pyplot as plt

import os 


for file in os.listdir("models"):
    if not file[-8:] == "1000.npz": continue 
    data = np.load("models/" + file)
    loss_values_test = data["loss_values_test"]
    loss_values_train = data["loss_values_train"]

    plt.figure(1)
    plt.plot((loss_values_test), label=file[:-4])
    plt.legend()

    plt.figure(2)
    plt.plot((loss_values_train), label=file[:-4])
    plt.legend()

plt.show()

# for act in range(1000):

#     # get sample action 

#     goals = []
#     action = env.action_space.sample()

#     expected_MSE = []

#     while len(goals) < 200:

#         print(act, len(goals), action)
#         env_ = RopeEnv(True)
#         env_.reset()
#         obs, _, _, _ = env_.step(action)

#         if np.all(obs["pos_traj"] == 0): 
#             print("failed")
#             continue 

#         goal = obs["pos_traj"][-2:,-1]
#         goals.append(goal)

#         goals_ = np.array(goals)
#         # print(np.std(goals_, axis=0))
#         # goals_ -= np.mean(goals_, axis = 0) 
#         expected_MSE.append(np.mean(np.std(goals_, axis=0)))

#     plt.plot(expected_MSE)

#     goals = np.array(goals)
#     goals -= np.mean(goals, axis = 0)

#     dist = np.linalg.norm(goals, axis = 1)
#     MSE.append(dist)

# np.save("analysis/groundtruth", MSE)
# plt.show()    

