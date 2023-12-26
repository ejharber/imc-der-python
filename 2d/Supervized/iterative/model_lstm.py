import torch
from torch import nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, lstm_num_layers, lstm_hidden_size, mlp_num_layers, mlp_hidden_size, train):
        super(LSTM, self).__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.train = train

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=self.train)

        # The linear layer that maps from hidden state space to tag space
        self.mlp = []

        for i in range(self.mlp_num_layers):
            input_size = self.mlp_hidden_size
            if i == 0:
                input_size = self.lstm_hidden_size + 3

            output_size = self.mlp_hidden_size
            if i == self.mlp_num_layers - 1:
                output_size = 2

            self.mlp.append(nn.Linear(input_size, output_size))

    def forward(self, obs, delta_action):

        obs = obs.permute(0, 2, 1) 
        h0 = torch.zeros(self.lstm_num_layers, obs.size(0), self.lstm_hidden_size).to('cpu') 
        c0 = torch.zeros(self.lstm_num_layers, obs.size(0), self.lstm_hidden_size).to('cpu')

        lstm_out, _ = self.lstm(obs, (h0, c0))
      
        if self.train:
            mlp_in = torch.cat((lstm_out[:, -1, :], delta_action), axis=1)
        else:
            mlp_in = torch.cat((lstm_out[-1, :], delta_action), axis=0)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        return mlp_out