import sys
sys.path.append("../gym/")

from rope_gym import RopeEnv
import gym 
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt

input_dim = 3
hidden_dim = 32
output_dim = 2

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim) 
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim, output_dim) 
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("oneshot", map_location=torch.device('cpu')))

env = RopeEnv()

X = np.random.rand(3, 1000000)
X[:2, :] = X[:2, :] * (env.high - env.low) - env.high
# plt.plot(X[0, :1000], X[1, :1000])
# plt.show()

X[2, :] = X[2, :] * 2*np.pi - np.pi
X = X.T
X = torch.from_numpy(X.astype(np.float32))

y = model(X).detach().numpy()

MSE = []
for _ in range(10000):
    obs = env.reset()

    goal = env.goal

    # print(np.min(np.linalg.norm(goal - y, axis = 1)))
    # print(np.argmin(np.linalg.norm(goal - y, axis = 1)))
    action = X[np.argmin(np.linalg.norm(goal - y, axis = 1)), :].detach().numpy()
    # print(action)
    obs, rewards, done, info = env.step(action)

    if rewards == -1000: 
        print('yes failed')
        continue 

    MSE.append(env.costFun())

print(np.mean(np.array(MSE)))



env.close()