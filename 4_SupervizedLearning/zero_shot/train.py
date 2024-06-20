import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex  # Import IPEX

import os
import sys
sys.path.append("..")
from load_data import load_data_zeroshot  # Assuming load_data is correctly defined

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleMLP, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, elementwise_affine=False)
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

def plot_curves(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

# Load data using the function
train_data, train_labels, test_data, test_labels = load_data_zeroshot("../../3_ExpandDataSet/raw_data")
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)  # Use torch.float32 for continuous labels
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)  # Use torch.float32 for continuous labels

# Determine input size and output size from data
input_size = train_data.shape[1]
output_size = train_labels.shape[1]

# Create DataLoader for batching
batch_size = 32
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = SimpleMLP(input_size, hidden_size=500, num_layers=5, output_size=output_size).to(device)
criterion = nn.MSELoss()  # Use Mean Squared Error for regression
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training and evaluation loop
num_epochs = 500
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)  # Ensure inputs are float32
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluation on test set
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)  # Ensure inputs are float32
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# Plotting and saving curves
plot_curves(train_losses, test_losses)

# Save model parameters and training curves
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses
}, 'model_checkpoint.pth')

print("Training completed.")
