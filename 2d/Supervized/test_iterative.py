import sys
sys.path.append("../gym/")

from rope_gym2 import RopeEnv
import gym 
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt

input_dim = 26
hidden_dim = 64
output_dim = 2

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim) 
        self.layer_3 = nn.Linear(hidden_dim, output_dim)         
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("iterative_noise_64"))

env = RopeEnv()

MSE = []
for experiment in range(6):

    print(experiment)
    env.reset()
    mse = []

    for step in range(5):

        X = np.random.rand(3, 1000000)
        X[:2, :] = X[:2, :] * (env.high - env.low) - env.high
        X[2, :] = X[2, :] * 2*np.pi - np.pi
        X = X.T

        state = np.concatenate((env.action, env.rope.getState().x))
        state = np.repeat([state], 1000000, axis=0)

        X = np.concatenate((state, X), axis=1)

        X = torch.from_numpy(X.astype(np.float32))
        y = model(X).detach().numpy()

        goal = env.goal
        # print(np.min(np.linalg.norm(goal - y, axis = 1)))
        # print(np.argmin(np.linalg.norm(goal - y, axis = 1)))
        action = X[np.argmin(np.linalg.norm(goal - y, axis = 1)), 23:].detach().numpy()
        # print(action)
        obs, rewards, done, info = env.step(action)

        mse.append(env.costFun())

    MSE.append(mse)

MSE =  np.array(MSE)

plt.plot(MSE)
plt.show()
print(np.mean(MSE))

env.close()