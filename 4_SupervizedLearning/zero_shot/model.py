import torch
from torch import nn
import numpy as np

class MLP_zeroshot(nn.Module):
    def __init__(self, mlp_num_layers, mlp_hidden_size, train=True):
        super(MLP_zeroshot, self).__init__()
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.input_dim = 3
        self.output_dim = 2

        self.train = train

        self.X_norm = nn.Linear(3, 3)
        self.X_norm.weight = nn.Parameter(torch.eye(3), False)
        self.X_norm.bias = nn.Parameter(torch.zeros(self.X_norm.bias.shape), False)

        self.y_norm = nn.Linear(2, 2)
        self.y_norm.weight = nn.Parameter(torch.eye(2), False)
        self.y_norm.bias = nn.Parameter(torch.zeros(self.y_norm.bias.shape), False)

        self.mlp = nn.ModuleList()

        for i in range(self.mlp_num_layers):
            input_size = self.mlp_hidden_size
            if i == 0:
                input_size = self.input_dim

            output_size = self.mlp_hidden_size
            if i == self.mlp_num_layers - 1:
                output_size = self.output_dim

            print(input_size, output_size)

            self.mlp.append(nn.Linear(input_size, output_size))

    def setNorms(self, X_mean, X_std, y_mean, y_std):
        X_mean = torch.from_numpy(X_mean.astype(np.float32))
        X_std = torch.from_numpy(X_std.astype(np.float32))
        y_mean = torch.from_numpy(y_mean.astype(np.float32))
        y_std = torch.from_numpy(y_std.astype(np.float32))

        self.X_norm.weight = nn.Parameter(torch.eye(3) * 1/X_std, False)
        self.X_norm.bias = nn.Parameter(-X_mean/X_std, False)

        self.y_norm.weight = nn.Parameter(torch.eye(2) * y_std, False)
        self.y_norm.bias = nn.Parameter(y_mean, False)

    def forward(self, mlp_in):

        if not self.train:
            mlp_in = self.X_norm(mlp_in)

        for i in range(self.mlp_num_layers):
            mlp_out = self.mlp[i](mlp_in)
            mlp_in = nn.functional.relu(mlp_out) # we don't want to apply relu on the last layers

        if not self.train:
            mlp_out = self.y_norm(mlp_out)

        return mlp_out