import torch
import torch.nn as nn
import torch.optim as optim

class LSTMMLPModel(nn.Module):
    def __init__(self, input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp, num_layers_lstm, num_layers_mlp, output_size,
                 delta_actions_mean=None, delta_actions_std=None, delta_goals_mean=None, delta_goals_std=None, traj_pos_mean=None, traj_pos_std=None):
        super(LSTMMLPModel, self).__init__()
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm

        self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers_lstm, batch_first=True)
        
        self.input_size_mlp = hidden_size_lstm + input_size_classic

        self.hidden_layers_mlp = nn.ModuleList()
        for _ in range(num_layers_mlp):
            self.hidden_layers_mlp.append(nn.Linear(self.input_size_mlp, hidden_size_mlp))
            self.input_size_mlp = hidden_size_mlp
        
        self.output_layer = nn.Linear(hidden_size_mlp, output_size)
        self.relu = nn.ReLU()

        self.delta_actions_mean = delta_actions_mean 
        self.delta_actions_std = delta_actions_std
        self.delta_goals_mean = delta_goals_mean
        self.delta_goals_std = delta_goals_std
        self.traj_pos_mean = traj_pos_mean
        self.traj_pos_std = traj_pos_std

    def forward(self, x_lstm, x_classic, test=False):
        if test:
            # if self.data_mean is None or self.data_std is None or self.labels_mean is None or self.labels_std:
                # print("failed to set noramlization params")
                # return

            x_lstm = x_lstm - self.traj_pos_mean
            x_lstm = x_lstm / self.traj_pos_std 

            x_classic = x_classic - self.delta_actions_mean
            x_classic = x_classic / self.delta_actions_std

        h0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        c0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm[:, -1, :]  # Take the output from the last time step
                
        # Concatenate LSTM output with classic input
        combined_input = torch.cat((out_lstm, x_classic), dim=1)

        # MLP layers
        for layer in self.hidden_layers_mlp:
            combined_input = layer(combined_input)
            combined_input = self.relu(combined_input)
        
        out = self.output_layer(combined_input)

        if test:
            out = out * self.delta_goals_std 
            out = out + self.delta_goals_mean
        
        return out