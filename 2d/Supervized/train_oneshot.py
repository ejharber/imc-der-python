import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader 
from torch import nn
from torch import optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

data = np.load("rope_motion_noise_0.npz")
X = data["actions_1"]
y = data["x_ee_1"]

data = np.load("rope_motion_noise_350000.npz")
X = np.append(X, data["actions_1"], axis = 0)
y = np.append(y, data["x_ee_1"], axis = 0)

data = np.load("rope_motion_noise_400000.npz")
X = np.append(X, data["actions_1"], axis = 0)
y = np.append(y, data["x_ee_1"], axis = 0)

data = np.load("rope_motion_noise_50000.npz")
X = np.append(X, data["actions_1"], axis = 0)
y = np.append(y, data["x_ee_1"], axis = 0)

# data = np.load("rope_motion_noise_250000.npz")
# X = np.append(X, data["actions_1"], axis = 0)
# y = np.append(y, data["x_ee_1"], axis = 0)

# data = np.load("rope_motion_noise_300000.npz")
# X = np.append(X, data["actions_1"], axis = 0)
# y = np.append(y, data["x_ee_1"], axis = 0)

# data = np.load("rope_motion_noise_350000.npz")
# X = np.append(X, data["actions_1"], axis = 0)
# y = np.append(y, data["x_ee_1"], axis = 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

# plt.plot(X_train[:1000,0], X_train[:1000,1])
# plt.show()
# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.X = self.X.to(device)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.y = self.y.to(device)
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self): 
        return self.len
    
batch_size = 64*64*64*32

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    # break

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
model.to(device)

print(model)

learning_rate = 0.2
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 5000
loss_values_train = []
loss_values_test = []
for epoch in range(num_epochs):
    for X, y in train_dataloader: 
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_values_train.append(loss.item())
        loss.backward()
        optimizer.step()

    for X, y in test_dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_values_test.append(loss.item())
        print(epoch, loss)

torch.save(model.state_dict(), "oneshot")

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

