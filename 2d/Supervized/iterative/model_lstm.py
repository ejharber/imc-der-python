import torch
from torch import nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, num_lstm_layers, hidden_dim, train):
        super(LSTM, self).__init__()
        self.lstm_hidden_size = 50
        self.hidden_dim = hidden_dim + 3
        self.train = train

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=4, batch_first=self.train)

        # The linear layer that maps from hidden state space to tag space
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim * 4, 2))
                                  # nn.ReLU(),
                                 # nn.Linear(self.hidden_dim * 2, 2))

    def forward(self, obs, delta_action):
        obs = obs.permute(0, 2, 1) 

        h0 = torch.zeros(4, obs.size(0), self.lstm_hidden_size).to('cpu') 
        c0 = torch.zeros(4, obs.size(0), self.lstm_hidden_size).to('cpu')

        lstm_out, _ = self.lstm(obs, (h0, c0))
        # print(lstm_out.shape)
        # print(delta_action.shape)


        if self.train:
            mlp_in = torch.cat((lstm_out[:, -1, :], delta_action), axis=1)
        else:
            mlp_in = torch.append((lstm_out[:, -1, :], delta_action), axis=0)

        # print(mlp_in.shape)
        mlp_out = self.mlp(mlp_in)

        return mlp_out