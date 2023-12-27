import sys
sys.path.append("../../gym/")
sys.path.append("../zero_shot/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import NeuralNetwork
from models import *

env = RopeEnv(True, "Human")
env.reset()
# I should replace this with a yaml file
input_dim = 3
hidden_dim = 256
output_dim = 2

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("../zero_shot/model_params/256_256", map_location=torch.device('cpu')))

# load zero shot info 
X_zero_shot = np.random.rand(3, 10_000)
X_zero_shot[:2, :] = X_zero_shot[:2, :] * (env.high - env.low) - env.high
X_zero_shot[2, :] = X_zero_shot[2, :] * 2 * np.pi - np.pi
X_zero_shot = X_zero_shot.T

X_zero_shot = torch.from_numpy(X_zero_shot.astype(np.float32))
y_zero_shot = model(X_zero_shot).detach().numpy()

model_iter = LSTM_iter(True, True, 1, 50, 3, 500, True)
model_iter.load_state_dict(torch.load("models/LSTM_alldata_1_50_3_500.pkg", map_location=torch.device('cpu')))

MSE = []

for trials in range(10):

    print('new trial')

    mse = []

    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y_zero_shot, axis = 1))
    action = X_zero_shot[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(5):

        print(mse[-1])
        X_iter = np.random.rand(10_000, 3)
        X_iter[:, :2] = (X_iter[:, :2] * (env.high - env.low) + env.low) * (5.0 * mse[-1])
        X_iter[:, 2] = (X_iter[:, 2] * 2 * np.pi - np.pi) * (5.0 * mse[-1])

        X_iter_ = np.copy(X_iter) # need to save the unaltered array for later 
        X_iter_ = torch.from_numpy(X_iter_.astype(np.float32))
        X_iter_ = X_iter_ - -3.1409014385288523
        X_iter_ = X_iter_ / (6.282368568831922)

        obs_pos = obs["pos_traj"]
        obs_pos = obs_pos - -0.4776209237113345
        obs_pos = obs_pos / (0.9556569193725786)

        obs_force = obs["force_traj"]
        obs_force = obs_force - -246.35813243543595
        obs_force = obs_force / (941.6271082062078)

        # obs_pos = np.random.normal(obs_pos, 0.003, obs_pos.shape) # artificial mocap noise added
        # obs_force = np.random.normal(obs_force, 0.15, obs_force.shape) # artificial force sensor noise added

        obs_iter = np.append(obs_pos, obs_force, axis = 0)
        obs_iter = np.expand_dims(obs_iter, axis = 0)
        obs_iter = np.repeat(obs_iter, repeats=10_000, axis=0)

        obs_iter = torch.from_numpy(obs_iter.astype(np.float32))

        pred_offset = model_iter.forward(obs_iter, X_iter_).detach().numpy()

        pred_offset = pred_offset * 0.7415825353522707
        pred_offset = pred_offset + -0.3727949567744294

        offset = -(goal - obs["pos_traj"][-2:, -1])
        offset = np.expand_dims(offset, axis = 0)
        offset = np.repeat(offset, repeats=10_000, axis=0)

        # print(offset.shape, pred_offset.shape, (offset - pred_offset).shape, np.linalg.norm(offset - pred_offset, axis = 1).shape)

        i = np.argmin(np.linalg.norm(offset - pred_offset, axis = 1))

        # print(X_iter[i, :], action)

        action -= X_iter[i, :]
        # print("UPDATE", X_iter[i, :])

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