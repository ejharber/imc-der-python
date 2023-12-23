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

env = RopeEnv(True, 'Both')

def data2imgs_test_id(data_traj, data_force):
    img_pos = data_traj
    img_pos_x = np.expand_dims(img_pos[::2, :], 0)
    img_pos_y = np.expand_dims(img_pos[1::2, :], 0)
    img_pos = np.concatenate((img_pos_x, img_pos_y), axis=0)
    img_pos = np.expand_dims(img_pos, 0)
    img_pos = torch.from_numpy(img_pos.astype(np.float32))

    img_force = data_force
    # img_force_x = np.expand_dims(img_force[::2, :], 0)
    # img_force_y = np.expand_dims(img_force[1::2, :], 0)
    # img_force = np.concatenate((img_force_x, img_force_y), axis=0)
    img_force = np.expand_dims(img_force, 0)
    img_force = torch.from_numpy(img_force.astype(np.float32))

    return img_pos, img_force


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

model_iter = IterativeNeuralNetwork_2(True)
model_iter.load_state_dict(torch.load("iterative_delta_force_id", map_location=torch.device('cpu')))

# load iterative info 

# y_zero_shot = model(X_zero_shot).detach().numpy() # need to wait for feedback 

MSE = []

for trials in range(100):

    mse = []

    obs = env.reset()

    goal = env.goal
    i = np.argmin(np.linalg.norm(goal - y_zero_shot, axis = 1))
    action = X_zero_shot[i, :].detach().numpy()
    obs, rewards, done, info = env.step(action)

    mse.append(env.costFun())
   
    for iterative in range(10):
    
        offset = (goal - obs["pos_traj"][-2:, -1])
        offset = np.array([offset])
        offset = torch.from_numpy(offset.astype(np.float32))
        img_pos, img_force = data2imgs_test_id(obs["pos_traj"], obs["force_traj"])  

        # print(offest)
        offset = offset - -0.004928639420564651
        offset = offset / (0.004513980317678079 - -0.004928639420564651)
        offset = 2 * (offset - 0.5)
        # X = torch.from_numpy(X.astype(np.float32)).to(device)

        img_pos = img_pos - -0.2995163480218865
        img_pos = img_pos / (0.3014384602381632 - -0.2995163480218865)
        img_pos = 2 * (img_pos - 0.5)
        # img_pos = torch.from_numpy(img_pos.astype(np.float32)).to(device)

        img_force = img_force - -43.33904941588156
        img_force = img_force / (90.96007853103129 - -43.33904941588156)
        img_force = 2 * (img_force - 0.5)
        # img_force = torch.from_numpy(img_force.astype(np.float32)).to(device)

        action_offset = model_iter.forward(img_pos, img_force, offset).detach().numpy()

        print(offset, "UPDATE", action_offset)

        action += action_offset[0,:]

        print(action)

        obs, rewards, done, info = env.step(action)

        mse.append(env.costFun())

    exit()

    MSE.append(mse)

MSE = np.array(MSE)
print(MSE.T)
plt.figure()
plt.plot(MSE.T)
plt.show()

np.save('mse_id', MSE)
env.close()