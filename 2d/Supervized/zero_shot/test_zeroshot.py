import sys
sys.path.append("../../gym/")

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
    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y, axis = 1))
    action = X[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    MSE.append(env.costFun())

    print(y.shape)

    print(trials, MSE[-1], MSE[-1]**2, np.linalg.norm(goal - y[i, :]))

print(np.mean(np.array(MSE)))

# np.save("analysis/64_64", MSE)

env.close()