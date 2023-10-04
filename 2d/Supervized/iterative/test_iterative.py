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

class IterativeNeuralNetwork(nn.Module):
    def __init__(self): 

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        super(IterativeNeuralNetwork, self).__init__()

        self.cnn_pos = nn.Sequential(
            layer_init(nn.Conv2d(2, 8, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 16, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            layer_init(nn.Linear(1408, 256)),
            nn.ReLU(),
        )

        self.cnn_force = nn.Sequential(
            layer_init(nn.Conv2d(2, 8, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 16, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            layer_init(nn.Linear(1408, 256)),
            nn.ReLU(),
        )

        self.network = nn.Sequential(
            layer_init(nn.Linear(6 + 253, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 2)),
            )

        
    def forward(self, img, next_action):
        x = self.cnn_pos(img)
        x = torch.cat((x, next_action), dim=1)
        x = self.network(x)
        return x

env = RopeEnv(True)

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
model_iter.load_state_dict(torch.load("model_params/iterative_delta", map_location=torch.device('cpu')))

# load iterative info 
X_iter = np.random.rand(3, 100_000)
X_iter[:2, :] = (X_iter[:2, :] * (env.high - env.low) - env.high) / 100.0
X_iter[2, :] = (X_iter[2, :] * 2 * np.pi - np.pi) / 100.0
X_iter = X_iter.T

X_iter = torch.from_numpy(X_iter.astype(np.float32))
# y_zero_shot = model(X_zero_shot).detach().numpy() # need to wait for feedback 

MSE = []

for trials in range(2):

    mse = []

    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y_zero_shot, axis = 1))
    action = X_zero_shot[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(5):

        img_pos = obs["pos_traj"]
        print(img_pos.shape)
        img_pos_x = np.expand_dims(img_pos[::2, :], 0)
        img_pos_y = np.expand_dims(img_pos[1::2, :], 0)

        img = np.concatenate((img_pos_x, img_pos_y), axis = 0)
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 100_000, 0)
        img = torch.from_numpy(img.astype(np.float32))

        pred_offset = model_iter.forward(img, X_iter).detach().numpy()

        offset =  (goal - obs["pos_traj"][-2:, -1])

        print(offset.shape, pred_offset.shape, (offset - pred_offset).shape, np.linalg.norm(offset - pred_offset, axis = 1).shape)

        i = np.argmin(np.linalg.norm(offset - pred_offset, axis = 1))

        print(X_iter[i, :], action)
        action += X_iter[i, :].detach().numpy()

        obs, rewards, done, info = env.step(action)

        mse.append(env.costFun())

    MSE.append(mse)

MSE = np.array(MSE)
print(MSE.T)
plt.figure()
plt.plot(MSE.T)
plt.show()

        # exit()

    # print(y.shape)

    # print(trials, MSE[-1], MSE[-1]**2, np.linalg.norm(goal - y[i, :]))

# print(np.mean(np.array(MSE)))

# np.save("analysis/64_64", MSE)

env.close()