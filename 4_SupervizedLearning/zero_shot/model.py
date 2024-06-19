import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleMLP, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(input_size if _ == 0 else hidden_size, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

# Model parameters
input_size = 28 * 28  # Example input size (e.g., for MNIST dataset)
hidden_size = 500
output_size = 10
num_layers = 3  # Example number of hidden layers

# Initialize the model, loss function, and optimizer
model = SimpleMLP(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Dummy training loop (replace with actual training loop)
for epoch in range(2):
    # Dummy inputs and targets (replace with actual data)
    inputs = torch.randn(32, input_size)
    targets = torch.randint(0, 10, (32,))
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed.")