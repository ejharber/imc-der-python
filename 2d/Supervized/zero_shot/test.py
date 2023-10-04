import sys
sys.path.append("../../gym/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import NeuralNetwork

# I should replace this with a yaml file
# I should replace this with a yaml file
input_dim = 3
hidden_dim = 128
output_dim = 2

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model_params/128_128", map_location=torch.device('cpu')))

env = RopeEnv(True, "Both")

X = np.random.rand(3, 10000000)
X[:2, :] = X[:2, :] * (env.high - env.low) - env.high
X[2, :] = X[2, :] * 2 * np.pi - np.pi
X = X.T

X = torch.from_numpy(X.astype(np.float32))
y = model(X).detach().numpy()

MSE = []
for trials in range(20):
    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y, axis = 1))
    action = X[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    if rewards == -1000: 
        print('yes failed')
        continue 

    MSE.append(env.costFun())

    print(y.shape)

    print(trials, MSE[-1], MSE[-1]**2, np.linalg.norm(goal - y[i, :]))

print(np.mean(np.array(MSE)))

# np.save("analysis/64_64", MSE)

env.close()