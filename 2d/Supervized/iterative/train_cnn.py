import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader 
from torch import nn
from torch import optim

class NeuralNetwork(nn.Module):
    def __init__(self): 

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        super(NeuralNetwork, self).__init__()

        self.cnn_pos = nn.Sequential(
            layer_init(nn.Conv2d(2, 4, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 1, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        self.cnn_force = nn.Sequential(
            layer_init(nn.Conv1d(2, 4, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 4, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv1d(4, 1, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # layer_init(nn.Linear(11, 256)),
            # nn.ReLU(),
        )

        self.network = nn.Sequential(
            layer_init(nn.Linear(11 + 3, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 2)),
            )

        
    def forward(self, img_pos, img_force, next_action):
        img_force = img_force[:,:,0,:]
        x_pos = self.cnn_pos(img_pos)
        # x_force = self.cnn_force(img_force)
        x = torch.cat((x_pos, next_action), dim=1)
        x = self.network(x)
        return x
      
def data2imgs(data_traj, data_force):
    img_pos = data_traj
    img_pos_x = np.expand_dims(img_pos[:, ::2, :], 1)
    img_pos_y = np.expand_dims(img_pos[:, 1::2, :], 1)
    img_pos = np.concatenate((img_pos_x, img_pos_y), axis=1)

    img_force = data_force
    img_force_x = np.expand_dims(img_force[:, ::2, :], 1)
    img_force_y = np.expand_dims(img_force[:, 1::2, :], 1)
    img_force = np.concatenate((img_force_x, img_force_y), axis=1)

    return img_pos, img_force

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"
# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

file_names = ["rope_motion_noise_0.npz", "rope_motion_noise_10000.npz"]
# file_names = ["rope_motion_noise_350000.npz", "rope_motion_noise_400000.npz", "rope_motion_noise_0.npz", "rope_motion_noise_50000.npz"]

X = np.array([])
img_pos = np.array([])
img_force = np.array([])
y = np.array([])

for file in file_names:
    print(file)

    try:
        data = np.load('data/' + file)
    except:
        continue 

    X_0 = data["actions_1"]
    y_0 = np.array(data["traj_pos_1"])[:,-2:,-1]

    X_1 = data["actions_2"] - X_0
    img_pos_1, img_force_1 = data2imgs(data["traj_pos_1"], data["traj_force_1"])
    y_1 = np.array(data["traj_pos_2"])[:,-2:,-1] - y_0

    X_2 = data["actions_3"] - X_0
    img_pos_2, img_force_2 = data2imgs(data["traj_pos_2"], data["traj_force_2"])  
    y_2 = np.array(data["traj_pos_3"])[:,-2:,-1] - y_0

    X_3 = data["actions_4"] - X_0
    img_pos_3, img_force_3 = data2imgs(data["traj_pos_3"], data["traj_force_3"])  
    y_3 = np.array(data["traj_pos_4"])[:,-2:,-1] - y_0 

    X_4 = data["actions_5"] - X_0
    img_pos_4, img_force_4 = data2imgs(data["traj_pos_4"], data["traj_force_4"])  
    y_4 = np.array(data["traj_pos_5"])[:,-2:,-1] - y_0 

    if X.shape[0] == 0:
        X = X_1
        img_pos = img_pos_1
        img_force = img_force_1
        y = y_1
    else:
        X = np.append(X, X_1, axis = 0)
        img_pos = np.append(img_pos, img_pos_1, axis = 0)
        img_force = np.append(img_force, img_force_1, axis = 0)
        y = np.append(y, y_1, axis = 0)

    X = np.append(X, X_2, axis = 0)
    img_pos = np.append(img_pos, img_pos_2, axis = 0)
    img_force = np.append(img_force, img_force_2, axis = 0)
    y = np.append(y, y_2, axis = 0)

    X = np.append(X, X_3, axis = 0)
    img_pos = np.append(img_pos, img_pos_3, axis = 0)
    img_force = np.append(img_force, img_force_3, axis = 0)
    y = np.append(y, y_3, axis = 0)

    X = np.append(X, X_4, axis = 0)
    img_pos = np.append(img_pos, img_pos_4, axis = 0)
    img_force = np.append(img_force, img_force_4, axis = 0)
    y = np.append(y, y_4, axis = 0)

# load to gpu
# X_ = X
X = X - np.min(X)
X = X / np.max(X)
X = 2 * (X - 0.5)
X = torch.from_numpy(X.astype(np.float32)).to(device)

img_pos = img_pos - np.min(img_pos)
img_pos = img_pos / np.max(img_pos)
img_pos = 2 * (img_pos - 0.5)
img_pos = torch.from_numpy(img_pos.astype(np.float32)).to(device)

img_force = img_force - np.min(img_force)
img_force = img_force / np.max(img_force)
img_force = 2 * (img_force - 0.5)
img_force = torch.from_numpy(img_force.astype(np.float32)).to(device)

y = y - np.min(y)
y = y / np.max(y)
y = 2 * (y - 0.5)
y = torch.from_numpy(y.astype(np.float32)).to(device)

# split training and testing 

rand_i = np.linspace(0, X.shape[0] - 1, X.shape[0], dtype = int)
np.random.shuffle(rand_i)
split = X.shape[0] * 2 // 3

# print(X.shape, split, rand_i.shape)

X_train, X_test = X[rand_i[:split], :], X[rand_i[split:], :]
img_pos_train, img_pos_test = img_pos[rand_i[:split], :, :, :], img_pos[rand_i[split:], :, :, :]
img_force_train, img_force_test = img_force[rand_i[:split], :, :, :], img_force[rand_i[split:], :, :, :]
y_train, y_test = y[rand_i[:split], :], y[rand_i[split:], :]



# print(X_train.shape, X_test.shape, img_train.shape, img_test.shape)
# exit()
# create model
model = NeuralNetwork()
model.to(device)

print(model)

# model.forward(img_train, X_train)

learning_rate = 0.04
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 10000
loss_values_train = []
loss_values_test = []
for epoch in range(num_epochs):

    print("training", X_train.shape)
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    pred = model(img_pos_train, img_force_train, X_train)
    loss = loss_fn(pred, y_train)
    loss_values_train.append(loss.item())
    loss.backward()
    optimizer.step()

    pred = model(img_pos_test, img_force_test, X_test)
    loss = loss_fn(pred, y_test)
    loss_values_test.append(loss.item())
    print("no force", epoch, loss_values_test[-1], loss_values_train[-1])

torch.save(model.state_dict(), "iterative_delta_no_force")

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

