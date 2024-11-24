import torch
import torch.nn as nn
import torch.optim as optim

class LSTMMLPModel(nn.Module):
    def __init__(self, input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp, num_layers_lstm, num_layers_mlp, output_size,
                 x_lstm_type=0, delta_actions_mean=None, delta_actions_std=None, delta_goals_mean=None, delta_goals_std=None, traj_pos_mean=None, traj_pos_std=None):
        super(LSTMMLPModel, self).__init__()
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm
        self.x_lstm_type = x_lstm_type

        if x_lstm_type == 0:
            self.input_size_lstm = input_size_lstm
        elif x_lstm_type == 1:
            self.input_size_lstm = 2  # First two dimensions
        elif x_lstm_type == 2:
            self.input_size_lstm = 1  # Third dimension

        self.lstm = nn.LSTM(self.input_size_lstm, hidden_size_lstm, num_layers_lstm, batch_first=True)
        
        self.input_size_mlp = hidden_size_lstm + input_size_classic

        self.hidden_layers_mlp = nn.ModuleList()
        for _ in range(num_layers_mlp):
            self.hidden_layers_mlp.append(nn.Linear(self.input_size_mlp, hidden_size_mlp))
            self.input_size_mlp = hidden_size_mlp
        
        self.output_layer = nn.Linear(hidden_size_mlp, output_size)
        self.relu = nn.ReLU()

        # Store the means and stds as tensors if they exist
        self.delta_actions_mean = delta_actions_mean if delta_actions_mean is None else torch.tensor(delta_actions_mean, dtype=torch.float32)
        self.delta_actions_std = delta_actions_std if delta_actions_std is None else torch.tensor(delta_actions_std, dtype=torch.float32)
        self.delta_goals_mean = delta_goals_mean if delta_goals_mean is None else torch.tensor(delta_goals_mean, dtype=torch.float32)
        self.delta_goals_std = delta_goals_std if delta_goals_std is None else torch.tensor(delta_goals_std, dtype=torch.float32)
        self.traj_pos_mean = traj_pos_mean if traj_pos_mean is None else torch.tensor(traj_pos_mean, dtype=torch.float32)
        self.traj_pos_std = traj_pos_std if traj_pos_std is None else torch.tensor(traj_pos_std, dtype=torch.float32)

    def forward(self, x_lstm, x_classic, test=False, run_time=False):

        if test:
            x_lstm = x_lstm - self.traj_pos_mean
            x_lstm = x_lstm / self.traj_pos_std 

            x_classic = x_classic - self.delta_actions_mean
            x_classic = x_classic / self.delta_actions_std

        if self.x_lstm_type == 0:
            x_lstm = x_lstm
        elif self.x_lstm_type == 1:
            x_lstm = x_lstm[:, :, :2]  # Take the first two dimensions
        elif self.x_lstm_type == 2:
            x_lstm = x_lstm[:, :, 2:3]  # Take the third dimension

        h0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        c0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm[:, -1, :]  # Take the output from the last time step

        if run_time:
            out_lstm = out_lstm.repeat(x_classic.shape[0], 1, 1)
                
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

    def to(self, device):
        model = super(LSTMMLPModel, self).to(device)
        
        # Move additional tensors to the specified device
        if self.delta_actions_mean is not None:
            self.delta_actions_mean = self.delta_actions_mean.to(device)
        if self.delta_actions_std is not None:
            self.delta_actions_std = self.delta_actions_std.to(device)
        if self.delta_goals_mean is not None:
            self.delta_goals_mean = self.delta_goals_mean.to(device)
        if self.delta_goals_std is not None:
            self.delta_goals_std = self.delta_goals_std.to(device)
        if self.traj_pos_mean is not None:
            self.traj_pos_mean = self.traj_pos_mean.to(device)
        if self.traj_pos_std is not None:
            self.traj_pos_std = self.traj_pos_std.to(device)
        return model

