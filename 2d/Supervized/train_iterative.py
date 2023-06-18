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

file_names = ["rope_motion_noise_0.npz", "rope_motion_noise_50000.npz", "rope_motion_noise_100000.npz", "rope_motion_noise_200000.npz", 
             "rope_motion_noise_300000.npz", "rope_motion_noise_350000.npz", "rope_motion_noise_500000.npz"]
# file_names = ["rope_motion_noise_350000.npz", "rope_motion_noise_400000.npz", "rope_motion_noise_0.npz", "rope_motion_noise_50000.npz"]

X = np.array([])
y = np.array([])

for file in file_names:
    print(file)
    data = np.load(file)

    # if file == "rope_motion_noise_50000.npz":
    X_1 = data["actions_0"]
    X_1 = np.append(X_1, data["x_0"], axis=1)
    X_1 = np.append(X_1, data["actions_1"], axis=1)
    y_1 = data["x_ee_1"]

    # else:
        # X_1 = np.array([])
        # y_1 = np.array([])


    X_2 = data["actions_1"]
    X_2 = np.append(X_2, data["x_1"], axis=1)
    X_2 = np.append(X_2, data["actions_2"], axis=1)

    y_2 = data["x_ee_2"]

    X_3 = data["actions_2"]
    X_3 = np.append(X_3, data["x_2"], axis=1)
    X_3 = np.append(X_3, data["actions_3"], axis=1)

    y_3 = data["x_ee_3"]

    if X.shape[0] == 0:
        X = X_1
    X = np.append(X, X_2, axis=0)
    X = np.append(X, X_3, axis=0)

    if y.shape[0] == 0:
        y = y_1
    y = np.append(y, y_2, axis=0)
    y = np.append(y, y_3, axis=0)

print(X.shape, X_1.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

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
    
batch_size = 64*64*32*4

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
    break

input_dim = 26
hidden_dim = 64
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

torch.save(model.state_dict(), "iterative_noise_64_downsample_first_motion")

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

