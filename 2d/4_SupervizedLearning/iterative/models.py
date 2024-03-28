import torch
from torch import nn
import numpy as np


class LSTM_iter(nn.Module):
    def __init__(self, include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size):
        super(LSTM_iter, self).__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.include_force = include_force 
        self.include_pos = include_pos

        assert self.include_force or self.include_pos

        self.input_dim = 42
        if not self.include_force:
            self.input_dim = 40
        if not self.include_pos:
            self.input_dim = 2 

        # normalization factors which make testing easier
        # setting one value to be the mean and the other to be the std seems arbitrary
        # maybe theres a better solution for storing noramlization values as parameters
        self.X_norm = nn.Linear(3, 3)
        self.X_norm.weight = nn.Parameter(torch.eye(3), False)
        self.X_norm.bias = nn.Parameter(torch.zeros(self.X_norm.bias.shape), False)

        self.obs_pos_norm = nn.Linear(1, 1) # maybe i can replace this later with an actual layer but tht math was confusing
        self.obs_pos_norm.weight = nn.Parameter(torch.tensor([0.0]), False) # holder for mean offset
        self.obs_pos_norm.bias = nn.Parameter(torch.tensor([1.0]), False) # holder for std

        self.obs_force_norm = nn.Linear(1, 1)
        self.obs_force_norm.weight = nn.Parameter(torch.tensor([0.0]), False)
        self.obs_force_norm.bias = nn.Parameter(torch.tensor([1.0]), False)

        self.y_norm = nn.Linear(2, 2)
        self.y_norm.weight = nn.Parameter(torch.eye(2), False)
        self.y_norm.bias = nn.Parameter(torch.zeros(self.y_norm.bias.shape), False)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        self.mlp = nn.ModuleList()

        for i in range(self.mlp_num_layers):
            input_size = self.mlp_hidden_size
            if i == 0:
                input_size = self.lstm_hidden_size + 3

            output_size = self.mlp_hidden_size
            if i == self.mlp_num_layers - 1:
                output_size = 2

            print(input_size, output_size)

            self.mlp.append(nn.Linear(input_size, output_size))

    def setNorms(self, X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std):
        # these are primarily used as containers to save the normalization values with the other weights
        X_mean = torch.from_numpy(X_mean.astype(np.float32))
        X_std = torch.from_numpy(X_std.astype(np.float32))

        obs_pos_mean = torch.from_numpy(obs_pos_mean.astype(np.float32))
        obs_pos_std = torch.from_numpy(obs_pos_std.astype(np.float32)) 

        obs_force_mean = torch.from_numpy(obs_force_mean.astype(np.float32))
        obs_force_std = torch.from_numpy(obs_force_std.astype(np.float32)) 

        y_mean = torch.from_numpy(y_mean.astype(np.float32))
        y_std = torch.from_numpy(y_std.astype(np.float32))

        # print(X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std)

        self.X_norm.weight = nn.Parameter(torch.eye(3) * 1/X_std, False)
        self.X_norm.bias = nn.Parameter(-X_mean/X_std, False)

        self.obs_pos_norm.weight = nn.Parameter(obs_pos_mean, False)
        self.obs_pos_norm.bias = nn.Parameter(obs_pos_std, False)

        self.obs_force_norm.weight = nn.Parameter(obs_force_mean, False)
        self.obs_force_norm.bias = nn.Parameter(obs_force_std, False)

        self.y_norm.weight = nn.Parameter(torch.eye(2) * y_std, False)
        self.y_norm.bias = nn.Parameter(y_mean, False)

        # print(self.X_norm.weight, self.X/_norm.bias, self.obs_pos_norm.weight, self.obs_force_norm.bias, self.obs_force_norm.weight, self.obs_force_norm.bias)

    def forward(self, obs, delta_action, train=False):

        if not train:
            obs = torch.clone(obs)
            obs[:, :40, :] = obs[:, :40, :] - self.obs_pos_norm.weight # subtract out mean 
            obs[:, :40, :] = obs[:, :40, :] / self.obs_pos_norm.bias # divide out std
            obs[:, 40:, :] = obs[:, 40:, :] - self.obs_force_norm.weight # dito 
            obs[:, 40:, :] = obs[:, 40:, :] / self.obs_force_norm.bias # dito 

            delta_action = torch.clone(delta_action)
            delta_action = self.X_norm(delta_action)

        if not self.include_force:
            obs = obs[:, :40, :]

        if not self.include_pos:
            obs = obs[:, 40:, :]

        obs = obs.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(obs)

        mlp_in = torch.cat((lstm_out[:, -1, :], delta_action), axis=1)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        if not train:
            mlp_out = self.y_norm(mlp_out)

        return mlp_out


class RNN_iter(nn.Module):
    def __init__(self, include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size):
        super(RNN_iter, self).__init__()
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.include_force = include_force 
        self.include_pos = include_pos

        assert self.include_force or self.include_pos

        self.input_dim = 42
        if not self.include_force:
            self.input_dim = 40
        if not self.include_pos:
            self.input_dim = 2 

        # normalization factors which make testing easier
        # setting one value to be the mean and the other to be the std seems arbitrary
        # maybe theres a better solution for storing noramlization values as parameters
        self.X_norm = nn.Linear(3, 3)
        self.X_norm.weight = nn.Parameter(torch.eye(3), False)
        self.X_norm.bias = nn.Parameter(torch.zeros(self.X_norm.bias.shape), False)

        self.obs_pos_norm = nn.Linear(1, 1) # maybe i can replace this later with an actual layer but tht math was confusing
        self.obs_pos_norm.weight = nn.Parameter(torch.tensor([0.0]), False) # holder for mean offset
        self.obs_pos_norm.bias = nn.Parameter(torch.tensor([1.0]), False) # holder for std

        self.obs_force_norm = nn.Linear(1, 1)
        self.obs_force_norm.weight = nn.Parameter(torch.tensor([0.0]), False)
        self.obs_force_norm.bias = nn.Parameter(torch.tensor([1.0]), False)

        self.y_norm = nn.Linear(2, 2)
        self.y_norm.weight = nn.Parameter(torch.eye(2), False)
        self.y_norm.bias = nn.Parameter(torch.zeros(self.y_norm.bias.shape), False)

        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers, batch_first=True)

        self.mlp = nn.ModuleList()

        for i in range(self.mlp_num_layers):
            input_size = self.mlp_hidden_size
            if i == 0:
                input_size = self.rnn_hidden_size + 3

            output_size = self.mlp_hidden_size
            if i == self.mlp_num_layers - 1:
                output_size = 2

            print(input_size, output_size)

            self.mlp.append(nn.Linear(input_size, output_size))

    def setNorms(self, X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std):
        # these are primarily used as containers to save the normalization values with the other weights
        X_mean = torch.from_numpy(X_mean.astype(np.float32))
        X_std = torch.from_numpy(X_std.astype(np.float32))

        obs_pos_mean = torch.from_numpy(obs_pos_mean.astype(np.float32))
        obs_pos_std = torch.from_numpy(obs_pos_std.astype(np.float32)) 

        obs_force_mean = torch.from_numpy(obs_force_mean.astype(np.float32))
        obs_force_std = torch.from_numpy(obs_force_std.astype(np.float32)) 

        y_mean = torch.from_numpy(y_mean.astype(np.float32))
        y_std = torch.from_numpy(y_std.astype(np.float32))

        # print(X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std)

        self.X_norm.weight = nn.Parameter(torch.eye(3) * 1/X_std, False)
        self.X_norm.bias = nn.Parameter(-X_mean/X_std, False)

        self.obs_pos_norm.weight = nn.Parameter(obs_pos_mean, False)
        self.obs_pos_norm.bias = nn.Parameter(obs_pos_std, False)

        self.obs_force_norm.weight = nn.Parameter(obs_force_mean, False)
        self.obs_force_norm.bias = nn.Parameter(obs_force_std, False)

        self.y_norm.weight = nn.Parameter(torch.eye(2) * y_std, False)
        self.y_norm.bias = nn.Parameter(y_mean, False)

        # print(self.X_norm.weight, self.X/_norm.bias, self.obs_pos_norm.weight, self.obs_force_norm.bias, self.obs_force_norm.weight, self.obs_force_norm.bias)

    def forward(self, obs, delta_action, train=False):

        if not train:
            obs = torch.clone(obs)
            obs[:, :40, :] = obs[:, :40, :] - self.obs_pos_norm.weight # subtract out mean 
            obs[:, :40, :] = obs[:, :40, :] / self.obs_pos_norm.bias # divide out std
            obs[:, 40:, :] = obs[:, 40:, :] - self.obs_force_norm.weight # dito 
            obs[:, 40:, :] = obs[:, 40:, :] / self.obs_force_norm.bias # dito 

            delta_action = torch.clone(delta_action)
            delta_action = self.X_norm(delta_action)

        if not self.include_force:
            obs = obs[:, :40, :]

        if not self.include_pos:
            obs = obs[:, 40:, :]

        obs = obs.permute(0, 2, 1) 
        rnn_out, _ = self.rnn(obs)

        mlp_in = torch.cat((rnn_out[:, -1, :], delta_action), axis=1)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        if not train:
            mlp_out = self.y_norm(mlp_out)

        return mlp_out

class CNN_widden_iter(nn.Module):
    def __init__(self, include_force, include_pos, mlp_num_layers, mlp_hidden_size): 
        super(CNN_widden_iter, self).__init__()
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.include_force = include_force 
        self.include_pos = include_pos

        assert self.include_force or self.include_pos

        self.cnn_pos = nn.Sequential(
            nn.Conv2d(2, 8, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 4, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.cnn_force = nn.Sequential(
            nn.Conv1d(2, 8, 4, stride=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 4, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        # normalization factors which make testing easier
        # setting one value to be the mean and the other to be the std seems arbitrary
        # maybe theres a better solution for storing noramlization values as parameters
        self.X_norm = nn.Linear(3, 3)
        self.X_norm.weight = nn.Parameter(torch.eye(3), False)
        self.X_norm.bias = nn.Parameter(torch.zeros(self.X_norm.bias.shape), False)

        self.obs_pos_norm = nn.Linear(1, 1) # maybe i can replace this later with an actual layer but tht math was confusing
        self.obs_pos_norm.weight = nn.Parameter(torch.tensor([0.0]), False) # holder for mean offset
        self.obs_pos_norm.bias = nn.Parameter(torch.tensor([1.0]), False) # holder for std

        self.obs_force_norm = nn.Linear(1, 1)
        self.obs_force_norm.weight = nn.Parameter(torch.tensor([0.0]), False)
        self.obs_force_norm.bias = nn.Parameter(torch.tensor([1.0]), False)

        self.y_norm = nn.Linear(2, 2)
        self.y_norm.weight = nn.Parameter(torch.eye(2), False)
        self.y_norm.bias = nn.Parameter(torch.zeros(self.y_norm.bias.shape), False)

        # need to determine these values from experimentation 
        self.input_dim = 88
        if not self.include_force:
            self.input_dim = 44
        if not self.include_pos:
            self.input_dim = 44

        self.mlp = nn.ModuleList()

        for i in range(self.mlp_num_layers):
            input_size = self.mlp_hidden_size
            if i == 0:
                input_size = self.input_dim + 3

            output_size = self.mlp_hidden_size
            if i == self.mlp_num_layers - 1:
                output_size = 2

            print(input_size, output_size)

            self.mlp.append(nn.Linear(input_size, output_size))

    def setNorms(self, X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std):
        # these are primarily used as containers to save the normalization values with the other weights
        X_mean = torch.from_numpy(X_mean.astype(np.float32))
        X_std = torch.from_numpy(X_std.astype(np.float32))

        obs_pos_mean = torch.from_numpy(obs_pos_mean.astype(np.float32))
        obs_pos_std = torch.from_numpy(obs_pos_std.astype(np.float32)) 

        obs_force_mean = torch.from_numpy(obs_force_mean.astype(np.float32))
        obs_force_std = torch.from_numpy(obs_force_std.astype(np.float32)) 

        y_mean = torch.from_numpy(y_mean.astype(np.float32))
        y_std = torch.from_numpy(y_std.astype(np.float32))

        # print(X_mean, X_std, obs_pos_mean, obs_pos_std, obs_force_mean, obs_force_std, y_mean, y_std)

        self.X_norm.weight = nn.Parameter(torch.eye(3) * 1/X_std, False)
        self.X_norm.bias = nn.Parameter(-X_mean/X_std, False)

        self.obs_pos_norm.weight = nn.Parameter(obs_pos_mean, False)
        self.obs_pos_norm.bias = nn.Parameter(obs_pos_std, False)

        self.obs_force_norm.weight = nn.Parameter(obs_force_mean, False)
        self.obs_force_norm.bias = nn.Parameter(obs_force_std, False)

        self.y_norm.weight = nn.Parameter(torch.eye(2) * y_std, False)
        self.y_norm.bias = nn.Parameter(y_mean, False)

        # print(self.X_norm.weight, self.X/_norm.bias, self.obs_pos_norm.weight, self.obs_force_norm.bias, self.obs_force_norm.weight, self.obs_force_norm.bias)


    def forward(self, obs, delta_action, train=False):

        if not train:
            obs = torch.clone(obs)
            obs[:, :40, :] = obs[:, :40, :] - self.obs_pos_norm.weight # subtract out mean 
            obs[:, :40, :] = obs[:, :40, :] / self.obs_pos_norm.bias # divide out std
            obs[:, 40:, :] = obs[:, 40:, :] - self.obs_force_norm.weight # dito 
            obs[:, 40:, :] = obs[:, 40:, :] / self.obs_force_norm.bias # dito 

            delta_action = torch.clone(delta_action)
            delta_action = self.X_norm(delta_action)

        cnn_out = torch.empty((obs.shape[0], 0))
        if self.include_pos:
            obs_pos = obs[:, :40, :]
            obs_pos = torch.unsqueeze(obs_pos, axis = 1)
            obs_pos = torch.cat((obs_pos[:, :, ::2, :], obs_pos[:, :, 1::2, :]), axis = 1)
            cnn_out = torch.cat((cnn_out, self.cnn_pos(obs_pos)), axis = 1)

        # print(cnn_out.shape)

        if self.include_force:
            obs_force = obs[:, 40:, :]
            obs_force = torch.unsqueeze(obs_force, axis = 1)
            obs_force = torch.cat((obs_force[:, :, 1, :], obs_force[:, :, 0, :]), axis = 1)
            cnn_out = torch.cat((cnn_out, self.cnn_force(obs_force)), axis = 1)

        mlp_in = torch.cat((cnn_out, delta_action), axis=1)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        if not train:
            mlp_out = self.y_norm(mlp_out)

        return mlp_out