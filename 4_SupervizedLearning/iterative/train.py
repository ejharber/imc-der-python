import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from load_data import *
from model_iterative import LSTMMLPModel

# Define model parameters at the beginning
input_size_lstm = None  # To be determined after loading data
input_size_classic = None  # To be determined after loading data
hidden_size_lstm = 200
hidden_size_mlp = 500
num_layers_lstm = 2
num_layers_mlp = 5
output_size = None  # To be determined after loading data
batch_size = 512
num_epochs = 1000
learning_rate = 0.00001
momentum = 0.9
checkpoint_freq = 50  # Frequency to save checkpoints

# Define x_lstm_type
x_lstm_type = 0  # 0: all dimensions, 1: first two dimensions, 2: third dimension

def plot_curves(train_losses, test_losses, valid_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Testing, and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

def plot_combined_data_distribution(train_data, test_data, valid_data, title):
    plt.figure(figsize=(10, 5))
    plt.hist(train_data.flatten(), bins=100, alpha=0.5, label='Train Data')
    plt.hist(test_data.flatten(), bins=100, alpha=0.5, label='Test Data')
    plt.hist(valid_data.flatten(), bins=100, alpha=0.5, label='Validation Data')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

# Load data using the function
(train_data, train_labels, test_data, test_labels,
 delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
 traj_pos_mean, traj_pos_std) = load_data_iterative("../../3_ExpandDataSet/raw_data")

train_data_time_series = torch.tensor(train_data['time_series'], dtype=torch.float32)  # Use float32 for LSTM input
train_data_classic = torch.tensor(train_data['classic'], dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

test_data_time_series = torch.tensor(test_data['time_series'], dtype=torch.float32)
test_data_classic = torch.tensor(test_data['classic'], dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Load validation data using the function
(valid_data, valid_labels) = load_realworlddata_iterative()

traj_pos_mean = torch.tensor(traj_pos_mean, dtype=torch.float32)
traj_pos_std = torch.tensor(traj_pos_std, dtype=torch.float32)
delta_actions_mean = torch.tensor(delta_actions_mean, dtype=torch.float32)
delta_actions_std = torch.tensor(delta_actions_std, dtype=torch.float32)
delta_goals_mean = torch.tensor(delta_goals_mean, dtype=torch.float32)
delta_goals_std = torch.tensor(delta_goals_std, dtype=torch.float32)

valid_data_time_series = torch.tensor(valid_data['time_series'], dtype=torch.float32)  # Use float32 for LSTM input
valid_data_time_series = (valid_data_time_series - traj_pos_mean) / traj_pos_std
valid_data_classic = torch.tensor(valid_data['classic'], dtype=torch.float32)
valid_data_classic = (valid_data_classic - delta_actions_mean) / delta_actions_std
valid_labels = torch.tensor(valid_labels, dtype=torch.float32)
valid_labels = (valid_labels - delta_goals_mean) / delta_goals_std

# Plot combined data distributions
plot_combined_data_distribution(train_data_time_series.numpy(), test_data_time_series.numpy(), valid_data_time_series.numpy(), 'Time Series Data Distribution')
plot_combined_data_distribution(train_data_classic.numpy(), test_data_classic.numpy(), valid_data_classic.numpy(), 'Classic Data Distribution')
plot_combined_data_distribution(train_labels.numpy(), test_labels.numpy(), valid_labels.numpy(), 'Labels Distribution')
plt.show()

# Determine input sizes and output size from data
input_size_lstm = train_data_time_series.shape[2]  # Assuming the time series data is in (samples, timesteps, features) format
input_size_classic = train_data_classic.shape[1]  # Assuming classic data is in (samples, features) format
output_size = train_labels.shape[1]

# Initialize the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = LSTMMLPModel(input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp, num_layers_lstm, num_layers_mlp, output_size, x_lstm_type=x_lstm_type).to(device)
criterion = nn.MSELoss()  # Use Mean Squared Error for regression
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Create DataLoader for batching
train_dataset = TensorDataset(train_data_time_series, train_data_classic, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(test_data_time_series, test_data_classic, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = TensorDataset(valid_data_time_series, valid_data_classic, valid_labels)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Training and evaluation loop
train_losses = []
test_losses = []
valid_losses = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch_idx, (inputs_lstm, inputs_classic, targets) in enumerate(train_loader):
        inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs_lstm, inputs_classic)
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
        for inputs_lstm, inputs_classic, targets in test_loader:
            inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
            outputs = model(inputs_lstm, inputs_classic)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Evaluation on valid set
    model.eval()  # Set the model to evaluation mode
    total_valid_loss = 0
    with torch.no_grad():
        for inputs_lstm, inputs_classic, targets in valid_loader:
            inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
            outputs = model(inputs_lstm, inputs_classic)
            loss = criterion(outputs, targets)
            total_valid_loss += loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}')
    
    # Save checkpoint every checkpoint_freq epochs
    if (epoch + 1) % checkpoint_freq == 0:
        checkpoint_path = f'model_checkpoint_{epoch+1}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'valid_losses': valid_losses,
            'input_size_lstm': input_size_lstm,
            'input_size_classic': input_size_classic,
            'hidden_size_lstm': hidden_size_lstm,
            'hidden_size_mlp': hidden_size_mlp,
            'num_layers_lstm': num_layers_lstm,
            'num_layers_mlp': num_layers_mlp,
            'output_size': output_size,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'delta_actions_mean': torch.tensor(delta_actions_mean, dtype=torch.float32),
            'delta_actions_std': torch.tensor(delta_actions_std, dtype=torch.float32),
            'delta_goals_mean': torch.tensor(delta_goals_mean, dtype=torch.float32),
            'delta_goals_std': torch.tensor(delta_goals_std, dtype=torch.float32),
            'traj_pos_mean': torch.tensor(traj_pos_mean, dtype=torch.float32),
            'traj_pos_std': torch.tensor(traj_pos_std, dtype=torch.float32),
            'x_lstm_type': x_lstm_type
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

# Plotting and saving curves
plot_curves(train_losses, test_losses, valid_losses)

# Save final model parameters and training curves
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'valid_losses': valid_losses,
    'input_size_lstm': input_size_lstm,
    'input_size_classic': input_size_classic,
    'hidden_size_lstm': hidden_size_lstm,
    'hidden_size_mlp': hidden_size_mlp,
    'num_layers_lstm': num_layers_lstm,
    'num_layers_mlp': num_layers_mlp,
    'output_size': output_size,
    'learning_rate': learning_rate,
    'momentum': momentum,
    'delta_actions_mean': torch.tensor(delta_actions_mean, dtype=torch.float32),
    'delta_actions_std': torch.tensor(delta_actions_std, dtype=torch.float32),
    'delta_goals_mean': torch.tensor(delta_goals_mean, dtype=torch.float32),
    'delta_goals_std': torch.tensor(delta_goals_std, dtype=torch.float32),
    'traj_pos_mean': torch.tensor(traj_pos_mean, dtype=torch.float32),
    'traj_pos_std': torch.tensor(traj_pos_std, dtype=torch.float32),
    'x_lstm_type': x_lstm_type
}, 'final_model_checkpoint.pth')

print("Training completed.")
