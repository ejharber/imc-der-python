import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, data_mean=None, data_std=None, labels_mean=None, labels_std=None):
        super(SimpleMLP, self).__init__()
        # self.layer_norm = nn.LayerNorm(input_size)
        self.hidden_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = nn.Linear(input_size if _ == 0 else hidden_size, hidden_size)
            nn.init.xavier_uniform_(layer.weight)  # Initialize weights with Xavier uniform initialization
            nn.init.zeros_(layer.bias)  # Initialize biases to zeros
            self.hidden_layers.append(layer)
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)  # Initialize output layer weights
        nn.init.zeros_(self.output_layer.bias)  # Initialize output layer biases to zeros
        
        self.relu = nn.ReLU()

        self.data_mean = data_mean 
        self.data_std = data_std
        self.labels_mean = labels_mean
        self.labels_std = labels_std

    def forward(self, x, test=False):
        if test:
            # if self.data_mean is None or self.data_std is None or self.labels_mean is None or self.labels_std:
                # print("failed to set noramlization params")
                # return

            x = x - self.data_mean
            x = x / self.data_std 

        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)

        if test:
            x = x * self.labels_std 
            x = x + self.labels_mean

        return x

if __name__ == '__main__':
    # Example usage:
    input_size = 10
    hidden_size = 100
    num_layers = 3
    output_size = 5
    
    model = SimpleMLP(input_size, hidden_size, num_layers, output_size)
    print(model)  # Print the model architecture with initialized weights
