import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from load_data import load_data_zeroshot, load_realworlddata_zeroshot  # Assuming load_data is correctly defined
from model_zeroshot import SimpleMLP

# Define model parameters at the beginning
input_size = None  # To be determined after loading data
hidden_size = 2000
num_layers = 5
output_size = None  # To be determined after loading data
batch_size = 512
num_epochs = 1000
learning_rate = 0.0001
momentum = 0.9
checkpoint_freq = 50  # Frequency to save checkpoints

def plot_curves(train_losses, test_losses, valid_losses, folder):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(valid_losses, label='Validation Loss')  # Add validation losses to the plot
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Testing, and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'loss_curves.png'))
    plt.show()

# Load train/test data (the filename will be used for the checkpoint folder)
dataset_name = "N2_all"
validation_dataset_name = "N3"
validation_dataset_params_name = "N3"

train_data, train_labels, test_data, test_labels, data_mean, data_std, labels_mean, labels_std = load_data_zeroshot(dataset_name)

# Create a folder for saving checkpoints and final models based on dataset name
checkpoint_folder = f"checkpoints_{dataset_name}"
os.makedirs(checkpoint_folder, exist_ok=True)

# Convert data and labels to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Convert means and stds to PyTorch tensors
data_mean = torch.tensor(data_mean, dtype=torch.float32)
data_std = torch.tensor(data_std, dtype=torch.float32)
labels_mean = torch.tensor(labels_mean, dtype=torch.float32)
labels_std = torch.tensor(labels_std, dtype=torch.float32)

# Load validation data
valid_actions, valid_goals = load_realworlddata_zeroshot(validation_dataset_name, validation_dataset_params_name)
valid_data = torch.tensor(valid_actions, dtype=torch.float32)
valid_labels = torch.tensor(valid_goals, dtype=torch.float32)

# Determine input size and output size from data
input_size = train_data.shape[1]
output_size = train_labels.shape[1]

# Create DataLoader for batching
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = TensorDataset(valid_data, valid_labels)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMLP(input_size, hidden_size, num_layers, output_size, data_mean, data_std, labels_mean, labels_std).to(device)
criterion = nn.MSELoss()  # Use Mean Squared Error for regression
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training and evaluation loop
train_losses = []
test_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Evaluation on validation set
    total_valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, test=True)
            loss = criterion(outputs, targets)
            total_valid_loss += loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    
    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}')
    
    # Save checkpoint every checkpoint_freq epochs
    if (epoch + 1) % checkpoint_freq == 0:
        checkpoint_path = os.path.join(checkpoint_folder, f'model_checkpoint_{epoch+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'valid_losses': valid_losses,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'learning_rate': learning_rate,
            'momentum': momentum,
            "data_mean": data_mean,
            "data_std": data_std,
            "labels_mean": labels_mean,
            "labels_std": labels_std
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

# Plotting and saving curves
plot_curves(train_losses, test_losses, valid_losses, checkpoint_folder)

# Save final model parameters and training curves
final_model_path = os.path.join(checkpoint_folder, 'final_model_checkpoint.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'valid_losses': valid_losses,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'output_size': output_size,
    'learning_rate': learning_rate,
    'momentum': momentum,
    "data_mean": data_mean,
    "data_std": data_std,
    "labels_mean": labels_mean,
    "labels_std": labels_std
}, final_model_path)

print(f"Training completed. Final model saved to: {final_model_path}")
