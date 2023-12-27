import torch
from torch import nn
import numpy as np

class LSTM_iter(nn.Module):
    def __init__(self, include_force, include_pos, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size, train):
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

        self.train = train

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=self.train)

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

    def forward(self, obs, delta_action):

        if not self.include_force:
            obs = obs[:, :40, :]

        if not self.include_pos:
            obs = obs[:, 40:, :]

        obs = obs.permute(0, 2, 1) 

        h0 = torch.zeros(self.lstm_num_layers, obs.size(0), self.lstm_hidden_size)
        c0 = torch.zeros(self.lstm_num_layers, obs.size(0), self.lstm_hidden_size)

        lstm_out, _ = self.lstm(obs, (h0, c0))
      
        if self.train:
            mlp_in = torch.cat((lstm_out[:, -1, :], delta_action), axis=1)
        else:
            mlp_in = torch.cat((lstm_out[-1, :], delta_action), axis=0)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

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

        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers, batch_first=self.train)

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

    def forward(self, obs, delta_action):

        if not self.include_force:
            obs = obs[:, :40, :]

        if not self.include_pos:
            obs = obs[:, 40:, :]

        obs = obs.permute(0, 2, 1) 

        h0 = torch.zeros(self.rnn_num_layers, obs.size(0), self.rnn_hidden_size)

        rnn_out, _ = self.rnn(obs, h0)
      
        if self.train:
            mlp_in = torch.cat((rnn_out[:, -1, :], delta_action), axis=1)
        else:
            mlp_in = torch.cat((rnn_out[-1, :], delta_action), axis=0)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        return mlp_out

# class CNN(nn.Module):
#     def __init__(self, cnn_num_layers, cnn_hidden_size, mlp_num_layers, mlp_hidden_size, train, include_force = True, include_pos = True): 

#         def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#             torch.nn.init.orthogonal_(layer.weight, std)
#             torch.nn.init.constant_(layer.bias, bias_const)
#             return layer

#         super(CNN, self).__init__()
#         self.cnn_num_layers = cnn_num_layers
#         self.cnn_hidden_size = cnn_hidden_size
#         self.include_force = include_force
#         self.mlp_num_layers = mlp_num_layers
#         self.mlp_hidden_size = mlp_hidden_size

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

#         if self.include_force:
#             self.network = nn.Sequential(
#                 layer_init(nn.Linear(90, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 3)),
#                 )
#         else: 
#             self.network = nn.Sequential(
#                 layer_init(nn.Linear(47 + 1, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 512)),
#                 nn.ReLU(),
#                 layer_init(nn.Linear(512, 3)),
#                 )

        
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