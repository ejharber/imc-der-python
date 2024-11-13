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
train_dataset_name = "N2_pose"  # Specify your dataset name
train_params_file_name = "N2_pose"
validation_dataset_name = "N3"
validation_params_file_name = "N3"

# Define x_lstm_type
x_lstm_type = 1  # 0: all dimensions, 1: first two dimensions, 2: third dimension

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

# Create a folder for saving checkpoints and final models based on dataset name
checkpoint_folder = f"checkpoints_nonoise_{train_dataset_name}"
os.makedirs(checkpoint_folder, exist_ok=True)

def plot_curves(train_losses, test_losses, valid_losses, folder):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Testing, and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'loss_curves.png'))
    plt.show()

# Load data using the function
(train_data, train_labels, test_data, test_labels,
 delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
 traj_pos_mean, traj_pos_std) = load_data_iterative(train_dataset_name, train_params_file_name)

# Prepare data for PyTorch
train_data_time_series = torch.tensor(train_data['time_series'], dtype=torch.float32)
train_data_classic = torch.tensor(train_data['classic'], dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

test_data_time_series = torch.tensor(test_data['time_series'], dtype=torch.float32)
test_data_classic = torch.tensor(test_data['classic'], dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Load and normalize validation data
valid_data, valid_labels = load_realworlddata_iterative(validation_dataset_name, validation_params_file_name)
valid_data_time_series = torch.tensor(valid_data['time_series'], dtype=torch.float32)
valid_data_classic = torch.tensor(valid_data['classic'], dtype=torch.float32)
valid_labels = torch.tensor(valid_labels, dtype=torch.float32)

# Determine input sizes and output size
input_size_lstm = train_data_time_series.shape[2]
input_size_classic = train_data_classic.shape[1]
output_size = train_labels.shape[1]

# Initialize the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMMLPModel(input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp,
                     num_layers_lstm, num_layers_mlp, output_size, 
                     delta_actions_mean=delta_actions_mean, delta_actions_std=delta_actions_std, 
                     delta_goals_mean=delta_goals_mean, delta_goals_std=delta_goals_std, 
                     traj_pos_mean=traj_pos_mean, traj_pos_std=traj_pos_std, x_lstm_type=x_lstm_type).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# DataLoader for batching
train_loader = DataLoader(TensorDataset(train_data_time_series, train_data_classic, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data_time_series, test_data_classic, test_labels), batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(TensorDataset(valid_data_time_series, valid_data_classic, valid_labels), batch_size=batch_size, shuffle=False)

# Training and evaluation loop
train_losses, test_losses, valid_losses = [], [], []
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for inputs_lstm, inputs_classic, targets in train_loader:
        inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs_lstm, inputs_classic)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    # Evaluate on test and validation sets
    model.eval()
    total_test_loss, total_valid_loss = 0, 0
    with torch.no_grad():
        for inputs_lstm, inputs_classic, targets in test_loader:
            inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
            outputs = model(inputs_lstm, inputs_classic)
            total_test_loss += criterion(outputs, targets).item()

        for inputs_lstm, inputs_classic, targets in valid_loader:
            inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
            outputs = model(inputs_lstm, inputs_classic, test=True)
            total_valid_loss += criterion(outputs, targets).item()

    test_losses.append(total_test_loss / len(test_loader))
    valid_losses.append(total_valid_loss / len(valid_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}, Validation Loss: {valid_losses[-1]:.6f}")

    # Save checkpoint every checkpoint_freq epochs
    if (epoch + 1) % checkpoint_freq == 0:
        checkpoint_path = os.path.join(checkpoint_folder, f'model_checkpoint_{epoch+1}.pth')
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
            'delta_actions_mean': delta_actions_mean,
            'delta_actions_std': delta_actions_std,
            'delta_goals_mean': delta_goals_mean,
            'delta_goals_std': delta_goals_std
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

# Plot and save final curves
plot_curves(train_losses, test_losses, valid_losses, checkpoint_folder)

# Save final model
final_model_path = os.path.join(checkpoint_folder, 'final_model_checkpoint.pth')
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
    'delta_actions_mean': delta_actions_mean,
    'delta_actions_std': delta_actions_std,
    'delta_goals_mean': delta_goals_mean,
    'delta_goals_std': delta_goals_std
}, final_model_path)

print(f"Training completed. Final model saved to: {final_model_path}")
