import sys
sys.path.append("../../gym/")
sys.path.append("../zero_shot/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import *

model = MLP_zeroshot(mlp_num_layers=3, mlp_hidden_size=500, train=False)
model.load_state_dict(torch.load("models/MLPZS_3_500.pkg", map_location=torch.device('cpu')))

env = RopeEnv(True, "Human")
obs = env.reset()
env.step(np.array([0,0,0]))

X = np.random.rand(3, 100_000)
X[:2, :] = X[:2, :] * (env.high - env.low) - env.high
X[2, :] = X[2, :] * 2 * np.pi - np.pi
X = X.T

X = torch.from_numpy(X.astype(np.float32))
y = model(X).detach().numpy()

MSE = []

for trials in range(10):

    print('new trial')

    mse = []

    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y, axis = 1))
    action = X[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(5):

        print(mse[-1])
        X_iter = (np.random.rand(10_000, 3) * 2 - 1) * (5.0 * mse[-1])
        X_iter_ = torch.from_numpy(X_iter.astype(np.float32))
        action_ = torch.from_numpy(action.astype(np.float32))

        pred_offset = model.forward(action_ + X_iter_).detach().numpy() -  model.forward(action_).detach().numpy()
        obs_pos_tip = obs["pos_traj"][-2:, -1]
        offset = (goal - obs_pos_tip)

        i = np.argmin(np.linalg.norm(offset - pred_offset, axis = 1))

        action += X_iter[i, :]

        obs, rewards, done, info = env.step(action)

        mse.append(env.costFun())

    MSE.append(mse)

MSE = np.array(MSE)
np.save('mse', MSE)

print(MSE.T)
plt.figure()
plt.plot(MSE.T)
plt.show()

np.save('mse', MSE)
env.close()