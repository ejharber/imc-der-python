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
from models import *

model_name = "MLPZS_5_1000"
model_zeroshot = MLP_zeroshot(mlp_num_layers=5, mlp_hidden_size=1000, train=False)
model_zeroshot.load_state_dict(torch.load("../zero_shot/models/MLPZS_5_1000.pkg", map_location=torch.device('cpu')))

model_name = "LSTM_alldata_2_50_2_1000_iter400"
model_iter = LSTM_iter(include_force=True, include_pos=True, lstm_num_layers=2, lstm_hidden_size=50, mlp_num_layers=3, mlp_hidden_size=500)
model_iter.load_state_dict(torch.load("models_daction_dgoal/LSTM_alldata_2_50_3_500_iter1600.pkg", map_location=torch.device('cpu')))

env = RopeEnv(True, "Human")
# env = RopeEnv(True)
obs = env.reset()
env.step(np.array([0,0,0]))

X_zeroshot = np.random.rand(10_000, 3)
X_zeroshot[:, :2] = X_zeroshot[:, :2] * (env.high - env.low) - env.high
X_zeroshot[:, 2] = X_zeroshot[:, 2] * 2 * np.pi - np.pi

X_zeroshot = torch.from_numpy(X_zeroshot.astype(np.float32))
y_zeroshot = model_zeroshot(X_zeroshot).detach().numpy()

MSE = []

for trial in range(1000):

    print('new trial')

    mse = []

    obs = env.reset(seed = trial)

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y_zeroshot, axis = 1))
    action = X_zeroshot[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(5):

        print("measured", mse[-1])
        X_iter = (np.random.rand(10_000, 3) * 2 - 1) * (mse[-1])
        X_iter_ = torch.from_numpy(X_iter.astype(np.float32))

        obs_pos = obs["pos_traj"]
        obs_force = obs["force_traj"]

        # obs_pos = np.random.normal(obs_pos, 0.003, obs_pos.shape) # artificial mocap noise added
        # obs_force = np.random.normal(obs_force, 0.15, obs_force.shape) # artificial force sensor noise added

        obs_iter = np.append(obs_pos, obs_force, axis = 0)
        obs_iter = np.expand_dims(obs_iter, axis = 0)
        obs_iter = np.repeat(obs_iter, repeats=10_000, axis=0)
        obs_iter = torch.from_numpy(obs_iter.astype(np.float32))

        # print(obs_iter.shape)

        # pred_goal = model_iter.forward(obs_iter, X_iter_, train=False).detach().numpy()
        # i = np.argmin(np.linalg.norm(goal - pred_goal, axis = 1))

        pred_delta_goal = model_iter.forward(obs_iter, X_iter_, train=False).detach().numpy()
        desired_delta_goal = goal - obs["pos_traj"][-2:, -1]
        print(desired_delta_goal)
        print(pred_delta_goal)
        # plt.plot(pred_delta_goal[:, 0], pred_delta_goal[:, 1], 'b.')
        # plt.show()
        i = np.argmin(np.linalg.norm(desired_delta_goal - pred_delta_goal, axis = 1))

        # print("expected", np.linalg.norm(goal - pred_goal[i, :]))

        action += X_iter[i, :]

        obs, rewards, done, info = env.step(action)

        mse.append(env.costFun())

    MSE.append(mse)

MSE = np.array(MSE)
np.save("analysis/" + model_name + "_iterresults", MSE)

print(MSE.T)
plt.figure()
plt.plot(MSE.T)
plt.show()

env.close()