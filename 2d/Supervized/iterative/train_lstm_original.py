import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader 
from torch import nn
from torch import optim
from models import *

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# set hyperparameters for training
include_force = True
include_pos = True
input_dim = 42
# note we assume at least one is true 

num_lstm_layers, lstm_hidden_size, mlp_num_layers, hidden_dim, train = 2, 100, 3, 512, True

file = "rope_motion_m=0_2.npz"
data = np.load('data/' + file)

# collect data
X_0 = data["actions_1"]
y_0 = np.array(data["traj_pos_1"])[:,-2:,-1]

X = np.empty((0, 3))
traj_data = np.empty((0, 42, 101))
y = np.empty((0, 2))

for i in range(2, 10):
    X_ = data["actions_" + str(i)] - X_0 # delta action 
    traj_pos_, traj_force_ = data["traj_pos_" + str(i-1)], data["traj_force_" + str(i-1)]
    y_ = data["traj_pos_" + str(i)][:,-2:,-1] - y_0 # delta goal 

    traj_data_ = np.append(traj_pos_, traj_force_, axis = 1)

    X = np.append(X, X_, axis = 0)
    traj_data = np.append(traj_data, traj_data_, axis = 0)
    y = np.append(y, y_, axis = 0)

# add artifical noise (similar to that as we would expect from the sensors)
traj_data[:, :40, :] = np.random.normal(traj_data[:, :40, :], 0.003, traj_data[:, :40, :].shape) # artificial mocap noise added
traj_data[:, 40:, :] = np.random.normal(traj_data[:, 40:, :], 0.15, traj_data[:, 40:, :].shape) # artificial force sensor noise added

# normalize data
traj_data[:, :40, :] = traj_data[:, :40, :] - np.mean(traj_data[:, :40, :])
traj_data[:, :40, :] = traj_data[:, :40, :] / np.std(traj_data[:, :40, :])

traj_data[:, 40:, :] = traj_data[:, 40:, :] - np.mean(traj_data[:, 40:, :])
traj_data[:, 40:, :] = traj_data[:, 40:, :] / np.std(traj_data[:, 40:, :])

X = X - np.mean(X)
X = X / np.std(X)

y = y - np.mean(y)
y = y / np.std(y)

if not include_force:
    traj_data = traj_data[:, :40, :]
    input_dim = 40

if not include_pos:
    traj_data = traj_data[:, 40:, :]
    input_dim = 2

# begin training
print(X.shape, traj_data.shape, y.shape)

X = torch.from_numpy(X.astype(np.float32)).to(device)
traj_data = torch.from_numpy(traj_data.astype(np.float32)).to(device)
y = torch.from_numpy(y.astype(np.float32)).to(device)

print(X.shape, traj_data.shape, y.shape)

rand_i = np.linspace(0, X.shape[0] - 1, X.shape[0], dtype = int)
np.random.shuffle(rand_i)
split = X.shape[0] * 2 // 3

X_train, X_test = X[rand_i[:split], :], X[rand_i[split:], :]
traj_data_train, traj_data_test = traj_data[rand_i[:split], :, :], traj_data[rand_i[split:], :, :]
y_train, y_test = y[rand_i[:split], :], y[rand_i[split:], :]

model = LSTM(input_dim, num_lstm_layers, hidden_dim, train)
model.to(device)

learning_rate = .2
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
loss_values_train = []
loss_values_test = []
for epoch in range(num_epochs):

    last = 0
    for batch in np.linspace(0, X_train.shape[0], 10, endpoint=False, dtype=np.int32):
        if batch == 0: continue 

        # print(batch)

        # print("training", X_train.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        pred = model(traj_data_train[last:batch, :, :], X_train[last:batch, :])
        loss = loss_fn(pred, y_train[last:batch, :])
        loss_values_train.append(loss.item())
        loss.backward()
        optimizer.step()

        last = batch

    pred = model(traj_data_test, X_test)
    loss = loss_fn(pred, y_test)
    loss_values_test.append(loss.item())
    print(epoch, loss_values_test[-1], loss_values_train[-1])

torch.save(model.state_dict(), "iterative_delta_force_foward_dynamics")

print("Training Complete")

step = np.linspace(0, num_epochs, len(np.array(loss_values_train)))

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values_train))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.figure()
plt.plot(loss_values_test)
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()
