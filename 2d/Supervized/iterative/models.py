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

            # print(self.obs_pos_norm.weight, self.obs_pos_norm.bias, self.obs_force_norm.weight, self.obs_force_norm.bias)
            # print(self.X_norm.weight, self.X_norm.bias)
            # print(self.y_norm.weight, self.y_norm.bias)

            # print(obs)

            delta_action = torch.clone(delta_action)
            delta_action = self.X_norm(delta_action)

            # print(delta_action)

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

            # print(mlp_out)

        return mlp_out

class RNN_iter(nn.Module):
    def __init__(self, include_force, include_pos, rnn_num_layers, rnn_hidden_size, mlp_num_layers, mlp_hidden_size, train):
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

        self.train = train

        # normalization factors which make testing easier
        self.X_norm = nn.Linear(3, 3)
        self.X_norm.weight = nn.Parameter(torch.eye(3), False)
        self.X_norm.bias = nn.Parameter(torch.zeros(self.X_norm.bias.shape), False)

        self.obs_pos_norm = nn.Linear(1, 1) # maybe i can replace this later with an actual layer but tht math was confusing
        self.obs_pos_norm.weight = nn.Parameter(torch.tensor([0]), False) # holder for mean offset
        self.obs_pos_norm.bias = nn.Parameter(torch.tensor([1]), False) # holder for std

        self.obs_force_norm = nn.Linear(1, 1)
        self.obs_force_norm.weight = nn.Parameter(torch.tensor([0]), False)
        self.obs_force_norm.bias = nn.Parameter(torch.tensor([1]), False)

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

        self.X_norm.weight = nn.Parameter(torch.eye(3) * 1/X_std, False)
        self.X_norm.bias = nn.Parameter(-X_mean/X_std, False)

        self.obs_pos_norm.weight = nn.Parameter(obs_pos_mean, False)
        self.obs_pos_norm.bias = nn.Parameter(obs_pos_std, False)

        self.obs_force_norm.weight = nn.Parameter(obs_force_mean, False)
        self.obs_force_norm.bias = nn.Parameter(obs_force_std, False)

        self.y_norm.weight = nn.Parameter(torch.eye(2) * y_std, False)
        self.y_norm.bias = nn.Parameter(y_mean, False)

    def forward(self, obs, delta_action):

        if not self.train:
            obs = obs[:, :40, :] - self.obs_pos_norm.weight # subtract out mean 
            obs = obs[:, :40, :] / self.obs_pos_norm.bias # divide out std
            obs = obs[:, 40:, :] - self.obs_force_norm.weight # dito 
            obs = obs[:, 40:, :] / self.obs_force_norm.bias # dito 

        if not self.include_force:
            obs = obs[:, :40, :]

        if not self.include_pos:
            obs = obs[:, 40:, :]

        obs = obs.permute(0, 2, 1) 

        rnn_out, _ = self.rnn(obs)
      
        mlp_in = torch.cat((rnn_out[:, -1, :], delta_action), axis=1)

        if not self.train:
            mlp_in = self.X_norm(mlp_in)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        if not self.train:
            mlp_out = self.y_norm(mlp_out)
            print(mlp_out)

        return mlp_out


# class CNN_iter(nn.Module):
#     def __init__(self, include_force, include_pos, mlp_num_layers, mlp_hidden_size, train): 

#         super(CNN, self).__init__()
#         self.include_force = include_force
#         self.include_pos = include_pos

#         self.mlp_num_layers = mlp_num_layers
#         self.mlp_hidden_size = mlp_hidden_size

#         self.train = train

#         self.cnn_pos = nn.Sequential(
#             layer_init(nn.Conv2d(2, 8, 4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(8, 16, 2, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(16, 32, 2, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 4, 2, stride=1)),
#             nn.ReLU(),
#             nn.Flatten(start_dim=1),
#         )

#         self.cnn_force = nn.Sequential(
#             layer_init(nn.Conv1d(2, 8, 4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv1d(8, 16, 2, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv1d(16, 32, 2, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv1d(32, 4, 2, stride=1)),
#             nn.ReLU(),
#             nn.Flatten(start_dim=1),
#         )

#         self.mlp = nn.ModuleList()

#         for i in range(self.mlp_num_layers):
#             input_size = self.mlp_hidden_size
#             if i == 0:
#                 input_size = self.rnn_hidden_size + 3

#             output_size = self.mlp_hidden_size
#             if i == self.mlp_num_layers - 1:
#                 output_size = 2

#             print(input_size, output_size)

#             self.mlp.append(nn.Linear(input_size, output_size))


        
#     def forward(self, obs, delta_action):
#         if 
#         img_force = img_force[:,:,0,:]
#         img_pos = img_pos[:,0,:,:, :]

#         # print(img_pos.shape, img_force.shape, next_action.shape)

#         if self.include_force:
#             x_force = self.cnn_force(img_force)

#             print(x_pos.shape, x_force.shape, next_action.shape)
#             x = torch.cat((x_pos, x_force, next_action), dim=1)
#         else:
#             x = torch.cat((x_pos, next_action), dim=1)


#         x = self.network(x)
#         return x