import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader 
from torch import nn
from torch import optim
from models import *

import sys
sys.path.append("../zero_shot/")
from model import MLP_zeroshot

def load_data():
    file = "rope_motion_m=0_2.npz"
    data = np.load('../data/' + file)

    # collect data
    # X_0 = data["actions_1"]
    # y_0 = np.array(data["traj_pos_1"])[:,-2:,-1]

    X_train = np.empty((0, 3))
    obs_train = np.empty((0, 42, 101))
    y_train = np.empty((0, 2))

    X_test = np.empty((0, 3))
    obs_test = np.empty((0, 42, 101))
    y_test = np.empty((0, 2))

    # model = MLP_zeroshot(mlp_num_layers=3, mlp_hidden_size=500, train=False)
    # model.load_state_dict(torch.load("../zero_shot/models/MLPZS_3_500.pkg", map_location=torch.device('cpu')))

    for i in range(5, 10):
        X_ = data["actions_" + str(i)] - data["actions_" + str(i-1)]# delta action 

        obs_pos_, obs_force_ = data["traj_pos_" + str(i-1)], data["traj_force_" + str(i-1)]

        # action_new = torch.from_numpy(data["actions_" + str(i)].astype(np.float32))
        # action_prev = torch.from_numpy(X_0.astype(np.float32))

        # y_ = (data["traj_pos_" + str(i)][:,-2:,-1] - y_0) - (model.forward(action_new).detach().numpy() - model.forward(action_prev).detach().numpy())# correction factor = difference between 
        y_ = data["traj_pos_" + str(i)][:,-2:,-1] - data["traj_pos_" + str(i - 1)][:,-2:,-1] # this could possibly be delta goal but that doesnt seem to work as well probably because theres a lot of symettires

        obs_ = np.append(obs_pos_, obs_force_, axis = 1)

        X_train = np.append(X_train, X_[:X_.shape[0] * 3 // 4, :], axis = 0)
        obs_train = np.append(obs_train, obs_[:X_.shape[0] * 3 // 4, :, :], axis = 0)
        y_train = np.append(y_train, y_[:X_.shape[0] * 3 // 4, :], axis = 0)

        X_test = np.append(X_test, X_[X_.shape[0] * 3 // 4:, :], axis = 0)
        obs_test = np.append(obs_test, obs_[X_.shape[0] * 3 // 4:, :, :], axis = 0)
        y_test = np.append(y_test, y_[X_.shape[0] * 3 // 4:, :], axis = 0)

    # add artifical noise (similar to that as we would expect from the sensors)
    # obs_train[:, :40, :] = np.random.normal(obs[:, :40, :], 0.003, obs[:, :40, :].shape) # artificial mocap noise added
    # obs_train[:, 40:, :] = np.random.normal(obs[:, 40:, :], 0.15, obs[:, 40:, :].shape) # artificial force sensor noise added

    return X_train, obs_train, y_train, X_test, obs_test, y_test

def train_lstm(include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size):

    if include_force and include_pos:
        save_file_name = "LSTM_" + "alldata_" + str(lstm_num_layers) + "_" + str(lstm_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_force:
        save_file_name = "LSTM_" + "noforce_" + str(lstm_num_layers) + "_" + str(lstm_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_pos:
        save_file_name = "LSTM_" + "nopose_" + str(lstm_num_layers) + "_" + str(lstm_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)

    model = LSTM_iter(include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size)

    train(model, save_file_name)

def train_rnn(include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size):

    if include_force and include_pos:
        save_file_name = "RNN_" + "alldata_" + str(rnn_num_layers) + "_" + str(rnn_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_force:
        save_file_name = "RNN_" + "noforce_" + str(rnn_num_layers) + "_" + str(rnn_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_pos:
        save_file_name = "RNN_" + "nopose_" + str(rnn_num_layers) + "_" + str(rnn_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)

    model = RNN_iter(include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size)

    train(model, save_file_name)

def train_cnn(include_force, include_pos, mlp_num_layers, mlp_hidden_size):

    if include_force and include_pos:
        save_file_name = "CNN_" + "alldata_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_force:
        save_file_name = "CNN_" + "noforce_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
    elif not include_pos:
        save_file_name = "CNN_" + "nopose_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)

    model = CNN_widden_iter(include_force, include_pos, mlp_num_layers, mlp_hidden_size)

    train(model, save_file_name)

def train(model, save_file_name):

    print("start training")

    X_train, obs_train, y_train, X_test, obs_test, y_test = load_data()
    X_test_, obs_test_, y_test_ = np.copy(X_test), np.copy(obs_test), np.copy(y_test) # these are our normalized values

    # normalize data
    X_mean = np.mean(X_train, axis = 0)
    X_train = X_train - X_mean
    X_std = np.std(X_train, axis = 0)
    X_train = X_train / X_std

    obs_mean_pos = np.array([np.mean(obs_train[:, :40, :])])
    obs_train[:, :40, :] = obs_train[:, :40, :] - obs_mean_pos
    obs_std_pos = np.array([np.std(obs_train[:, :40, :])])
    obs_train[:, :40, :] = obs_train[:, :40, :] / obs_std_pos

    obs_mean_force = np.array([np.mean(obs_train[:, 40:, :])])
    obs_train[:, 40:, :] = obs_train[:, 40:, :] - obs_mean_force
    obs_std_force = np.array([np.std(obs_train[:, 40:, :])])
    obs_train[:, 40:, :] = obs_train[:, 40:, :] / obs_std_force

    y_mean = np.mean(y_train, axis = 0)
    y_train = y_train - y_mean
    y_std = np.std(y_train, axis = 0)
    y_train = y_train / y_std

    model.setNorms(X_mean, X_std, obs_mean_pos, obs_std_pos, obs_mean_force, obs_std_force, y_mean, y_std) # save these for evaluation 

    # begin training
    print(X_train.shape, obs_train.shape, y_train.shape, X_test.shape, obs_test.shape, y_test.shape)

    rand_i = np.linspace(0, X_train.shape[0] - 1, X_train.shape[0], dtype = int)
    np.random.shuffle(rand_i)

    X_train = X_train[rand_i, :]
    obs_train = obs_train[rand_i, :, :]
    y_train = y_train[rand_i, :]

    # setup testing data sets
    # one is normalized and the other isnt
    X_test = torch.from_numpy(X_test.astype(np.float32))
    obs_test = torch.from_numpy(obs_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    X_train = torch.from_numpy(X_train.astype(np.float32))
    obs_train = torch.from_numpy(obs_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    X_test_ = (X_test_ - X_mean) / X_std
    obs_test_[:, :40, :] = (obs_test_[:, :40, :] - obs_mean_pos) / obs_std_pos
    obs_test_[:, 40:, :] = (obs_test_[:, 40:, :] - obs_mean_force) / obs_std_force
    y_test_ = (y_test_ - y_mean) / y_std

    X_test_ = torch.from_numpy(X_test_.astype(np.float32))
    obs_test_ = torch.from_numpy(obs_test_.astype(np.float32))
    y_test_ = torch.from_numpy(y_test_.astype(np.float32))

    learning_rate = 0.05
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 5000
    loss_values_train = []
    loss_values_test = []
    loss_values_test_ = []
    for epoch in range(num_epochs):

        last = 0
        for batch in np.linspace(0, X_train.shape[0], 10, endpoint=False, dtype=np.int32):
            if batch == 0: continue 

            X_train_batched = X_train[last:batch, :]
            obs_train_batched = obs_train[last:batch, :, :]
            y_train_batched = y_train[last:batch, :]

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            pred = model(obs_train_batched, X_train_batched, train=True)
            loss = loss_fn(pred, y_train_batched)
            loss_values_train.append(loss.item())
            # print(loss.item())
            loss.backward()
            optimizer.step()

            last = batch

        pred = model(obs_test, X_test, train=False)
        loss = loss_fn(pred, y_test)
        loss_values_test.append(loss.item())

        pred = model(obs_test_, X_test_, train=True)
        loss = loss_fn(pred, y_test_)
        loss_values_test_.append(loss.item())

        print(epoch, "test:", "{:.2e}".format(loss_values_test[-1]), "test_:", "{:.2e}".format(loss_values_test_[-1]), "train:", "{:.2e}".format(np.mean(np.array(loss_values_train[-10:]))))

        if epoch % 100 == 0: 
            np.savez("models_daction_dgoal/" + save_file_name + "_" + 'iter' + str(epoch), loss_values_test = loss_values_test, loss_values_test_ = loss_values_test_, loss_values_train = loss_values_train)
            torch.save(model.state_dict(), "models_daction_dgoal/" + save_file_name + "_" + 'iter' + str(epoch) + '.pkg')

    np.savez("models_daction_dgoal/" + save_file_name, loss_values_test = loss_values_test, loss_values_test_ = loss_values_test_, loss_values_train = loss_values_train)
    torch.save(model.state_dict(), "models_daction_dgoal/" + save_file_name + '.pkg')

# train_lstm(include_force = True, include_pos = True, lstm_num_layers = 2, lstm_hidden_size = 50, mlp_num_layers = 3, mlp_hidden_size = 500)
train_cnn(include_force = True, include_pos = True, mlp_num_layers = 3, mlp_hidden_size = 500)

# train_lstm(include_force = True, include_pos = True, lstm_num_layers = 2, lstm_hidden_size = 20, mlp_num_layers = 3, mlp_hidden_size = 200)
# train_lstm(include_force = True, include_pos = True, lstm_num_layers = 2, lstm_hidden_size = 100, mlp_num_layers = 3, mlp_hidden_size = 1000)
# train_lstm(include_force = True, include_pos = True, lstm_num_layers = 2, lstm_hidden_size = 100, mlp_num_layers = 4, mlp_hidden_size = 1000)

# for num_layers in [1, 2, 3, 4]:
#     for hidden_size in [10, 20, 50, 100, 200]:
#         for mlp_num_layers in [2, 3, 4, 5]:
#             for mlp_hidden_size in [10, 50, 100, 500, 1000]:

#                 # try:
#                 train_lstm(include_force = True, include_pos = True, lstm_num_layers = num_layers, lstm_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
#                 train_lstm(include_force = True, include_pos = False, lstm_num_layers = num_layers, lstm_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
#                 train_lstm(include_force = False, include_pos = True, lstm_num_layers = num_layers, lstm_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
#                 train_rnn(include_force = True, include_pos = True, rnn_num_layers = num_layers, rnn_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
#                 train_rnn(include_force = True, include_pos = False, rnn_num_layers = num_layers, rnn_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
#                 train_rnn(include_force = False, include_pos = True, rnn_num_layers = num_layers, rnn_hidden_size = hidden_size, mlp_num_layers = mlp_num_layers, mlp_hidden_size = mlp_hidden_size)
                # except: pass

#                 include_pos = True
#                 include_force = True

#                 # try:
#                 train_lstm(include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 include_pos = False
#                 include_force = True

#                 # try:
#                 train_lstm(include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 include_pos = True
#                 include_force = False

#                 # try:
#                 train_lstm(include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 if mlp_num_layers == 1:
#                     break

# for rnn_num_layers in [1, 2]:
#     for rnn_hidden_size in [10, 20, 50, 100, 200]:
#         for mlp_num_layers in [2, 3, 4]:
#             for mlp_hidden_size in [10, 50, 100, 500, 1000]:

#                 include_pos = True
#                 include_force = True

#                 # try:
#                 train_lstm(include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 include_pos = False
#                 include_force = True

#                 # try:
#                 train_lstm(include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 include_pos = True
#                 include_force = False

#                 # try:
#                 train_lstm(include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size)
#                 # except:
#                     # pass

#                 if mlp_num_layers == 1:
#                     break
