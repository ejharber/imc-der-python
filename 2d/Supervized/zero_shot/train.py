import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader 
from torch import nn
from torch import optim
from model import *

def load_data():
    file = "rope_motion_m=0_2.npz"
    data = np.load('../data/' + file)

    # collect data
    X = np.empty((0, 3))
    y = np.empty((0, 2))

    for i in range(1, 10):
        X_ = data["actions_" + str(i)] # action 
        y_ = data["traj_pos_" + str(i)][:,-2:,-1] # tip pose 

        X = np.append(X, X_, axis = 0)
        y = np.append(y, y_, axis = 0)

    return X, y

def train_zeroshot(mlp_num_layers, mlp_hidden_size):

    save_file_name = "MLPZS_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
 
    model = MLP_zeroshot(mlp_num_layers, mlp_hidden_size)

    train(model, save_file_name)

def train(model, save_file_name):

    print("start training")

    X, y = load_data()

    # normalize data
    X_mean = np.mean(X, axis = 0)
    X = X - X_mean
    X_std = np.std(X, axis = 0)
    X = X / X_std

    y_mean = np.mean(y, axis = 0)
    y = y - y_mean
    y_std = np.std(y, axis = 0)
    y = y / y_std

    model.setNorms(X_mean, X_std, y_mean, y_std) # save these for evaluation 

    rand_i = np.linspace(0, X.shape[0] - 1, X.shape[0], dtype = int)
    np.random.shuffle(rand_i)
    split = X.shape[0] * 5 // 6

    X_train, X_test = X[rand_i[:split], :], X[rand_i[split:], :]
    y_train, y_test = y[rand_i[:split], :], y[rand_i[split:], :]

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    learning_rate = 0.05
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 10000
    loss_values_train = []
    loss_values_test = []
    for epoch in range(num_epochs):

        last = 0
        for batch in np.linspace(0, X_train.shape[0], 10, endpoint=False, dtype=np.int32):
            if batch == 0: continue 

            X_train_batched = X_train[last:batch, :]
            y_train_batched = y_train[last:batch, :]

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            pred = model(X_train_batched)
            loss = loss_fn(pred, y_train_batched)
            loss_values_train.append(loss.item())
            loss.backward()
            optimizer.step()

            last = batch

        pred = model(X_test)
        loss = loss_fn(pred, y_test)
        loss_values_test.append(loss.item())

        print(epoch, loss_values_test[-1], loss_values_train[-1])

    np.savez("models/" + save_file_name, loss_values_test = loss_values_test, loss_values_train = loss_values_train)
    torch.save(model.state_dict(), "models/" + save_file_name + '.pkg')

for mlp_num_layers in [1, 2, 3, 4, 5]:
    for mlp_hidden_size in [10, 50, 100, 500, 1000]:
        train_zeroshot(mlp_num_layers = 5, mlp_hidden_size=1000)
        exit()

        if mlp_num_layers == 1:
            break 
