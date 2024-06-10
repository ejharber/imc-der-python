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
from model_iter import *

env = RopeEnv(True, "Human")
env.reset()
env.step(np.array([0, 0, 0]))
# I should replace this with a yaml file
input_dim = 3
hidden_dim = 256
output_dim = 2

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("../zero_shot/model_params/256_256", map_location=torch.device('cpu')))

# load zero shot info 
X_zero_shot = np.random.rand(3, 100000)
X_zero_shot[:2, :] = X_zero_shot[:2, :] * (env.high - env.low) - env.high
X_zero_shot[2, :] = X_zero_shot[2, :] * 2 * np.pi - np.pi
X_zero_shot = X_zero_shot.T

X_zero_shot = torch.from_numpy(X_zero_shot.astype(np.float32))
y_zero_shot = model(X_zero_shot).detach().numpy()

model_iter = IterativeNeuralNetwork()
model_iter.load_state_dict(torch.load("iterative_delta_force", map_location=torch.device('cpu')))

# load iterative info 

# y_zero_shot = model(X_zero_shot).detach().numpy() # need to wait for feedback 

MSE = []

for trials in range(10):

    mse = []

    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y_zero_shot, axis = 1))
    action = X_zero_shot[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(5):

        img_pos, img_force = data2imgs_test(obs["pos_traj"], obs["force_traj"])  
        
        # load to gpu
        # X_ = X
        X_iter = np.random.rand(3, 100_000)
        X_iter[:2, :] = (X_iter[:2, :] * (env.high - env.low) + env.low) / 100.0
        X_iter[2, :] = (X_iter[2, :] * 2 * np.pi - np.pi) / 100.0
        X_iter = X_iter.T

        X_iter_ = np.copy(X_iter)
        X_iter_ = torch.from_numpy(X_iter_.astype(np.float32))

        X_iter_ = X_iter_ - -0.03141361697681777
        X_iter_ = X_iter_ / (0.03141440957175545 - -0.03141361697681777)
        X_iter_ = 2 * (X_iter_ - 0.5)
        # X = torch.from_numpy(X.astype(np.float32)).to(device)

        img_pos = img_pos - -0.2995163480218865
        img_pos = img_pos / (0.3014384602381632 - -0.2995163480218865)
        img_pos = 2 * (img_pos - 0.5)
        # img_pos = torch.from_numpy(img_pos.astype(np.float32)).to(device)

        img_force = img_force - -43.33904941588156
        img_force = img_force / (90.96007853103129 - -43.33904941588156)
        img_force = 2 * (img_force - 0.5)
        # img_force = torch.from_numpy(img_force.astype(np.float32)).to(device)

        pred_offset = model_iter.forward(img_pos, img_force, X_iter_).detach().numpy()

        offset =  (goal - obs["pos_traj"][-2:, -1])

        print(offset.shape, pred_offset.shape, (offset - pred_offset).shape, np.linalg.norm(offset - pred_offset, axis = 1).shape)

        i = np.argmin(np.linalg.norm(offset - pred_offset, axis = 1))

        print(X_iter[i, :], action)

        action += 3 * X_iter[i, :]
        print("UPDATE", X_iter[i, :])

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