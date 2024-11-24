import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from load_data import load_data_iterative, load_realworlddata_iterative
from model_iterative import LSTMMLPModel

# Define model parameters at the beginning
train_dataset_name = "N2_all"  # Specify your dataset name
train_params_file_name = "N2_all"
validation_dataset_name = "N3"
validation_params_file_name = "N3"

checkpoint_folder = f"checkpoints_{train_dataset_name}"
checkpoint_file = "final_model_checkpoint.pth"  # Final model checkpoint
checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Extract parameters from checkpoint
input_size_lstm = checkpoint['input_size_lstm']
input_size_classic = checkpoint['input_size_classic']
hidden_size_lstm = checkpoint['hidden_size_lstm']
hidden_size_mlp = checkpoint['hidden_size_mlp']
num_layers_lstm = checkpoint['num_layers_lstm']
num_layers_mlp = checkpoint['num_layers_mlp']
output_size = checkpoint['output_size']
delta_actions_mean = checkpoint['delta_actions_mean']
delta_actions_std = checkpoint['delta_actions_std']
delta_goals_mean = checkpoint['delta_goals_mean']
delta_goals_std = checkpoint['delta_goals_std']
traj_pos_mean = checkpoint['traj_pos_mean']
traj_pos_std = checkpoint['traj_pos_std']
x_lstm_type = checkpoint["x_lstm_type"]

# Load test and validation data
# Load data using the function
_, _, test_data, test_labels, _, _, _, _, _, _ = load_data_iterative(train_dataset_name, train_params_file_name, normalize=False)
valid_data, valid_labels = load_realworlddata_iterative(validation_dataset_name, validation_dataset_name)

# Convert data to PyTorch tensors
test_data_time_series = torch.tensor(test_data['time_series'], dtype=torch.float32)
test_data_classic = torch.tensor(test_data['classic'], dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

valid_data_time_series = torch.tensor(valid_data['time_series'], dtype=torch.float32)
valid_data_classic = torch.tensor(valid_data['classic'], dtype=torch.float32)
valid_labels = torch.tensor(valid_labels, dtype=torch.float32)

# DataLoader for batching
test_loader = DataLoader(TensorDataset(test_data_time_series, test_data_classic, test_labels), batch_size=512, shuffle=False)
valid_loader = DataLoader(TensorDataset(valid_data_time_series, valid_data_classic, valid_labels), batch_size=512, shuffle=False)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMMLPModel(input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp,
                     num_layers_lstm, num_layers_mlp, output_size,
                     delta_actions_mean=delta_actions_mean, delta_actions_std=delta_actions_std,
                     delta_goals_mean=delta_goals_mean, delta_goals_std=delta_goals_std,
                     traj_pos_mean=traj_pos_mean, traj_pos_std=traj_pos_std, x_lstm_type=x_lstm_type).to(device)

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])

# Define loss function
criterion = nn.MSELoss()

# Evaluate on test and validation datasets
model.eval()
def evaluate(loader):
    total_loss = 0
    with torch.no_grad():
        for inputs_lstm, inputs_classic, targets in loader:
            inputs_lstm, inputs_classic, targets = inputs_lstm.to(device), inputs_classic.to(device), targets.to(device)
            outputs = model(inputs_lstm, inputs_classic, test=True)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

test_loss = evaluate(test_loader)
valid_loss = evaluate(valid_loader)

print(f"Test Loss: {test_loss:.6f}")
print(f"Validation Loss: {valid_loss:.6f}")
