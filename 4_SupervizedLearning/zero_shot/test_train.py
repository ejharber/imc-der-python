import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from load_data import load_data_zeroshot, load_realworlddata_zeroshot  # Assuming load_data is correctly defined
from model_zeroshot import SimpleMLP

# Define parameters for loading the model
dataset_name = "N2_all"  # Must match the dataset name used during training
checkpoint_folder = f"checkpoints_{dataset_name}"
checkpoint_file = "final_model_checkpoint.pth"  # The final model or a specific checkpoint file
checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)

# Load the saved model checkpoint
checkpoint = torch.load(checkpoint_path)

# Extract necessary parameters from the checkpoint
input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']
output_size = checkpoint['output_size']
data_mean = checkpoint['data_mean']
data_std = checkpoint['data_std']
labels_mean = checkpoint['labels_mean']
labels_std = checkpoint['labels_std']

# Load test data (must be identical to how it was done during training)
train_data, train_labels, test_data, test_labels, _, _, _, _ = load_data_zeroshot(dataset_name)

# Convert test data and labels to PyTorch tensors
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Create DataLoader for batching
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Initialize the model and load state dict from checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMLP(input_size, hidden_size, num_layers, output_size, data_mean, data_std, labels_mean, labels_std).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Define the loss function (MSE used during training)
criterion = nn.MSELoss()

# Evaluation on the test set
model.eval()  # Set the model to evaluation mode
total_test_loss = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_test_loss += loss.item()

# Compute average test loss
avg_test_loss = total_test_loss / len(test_loader)

print(f"Expected Test Loss: {avg_test_loss:.6f}")
