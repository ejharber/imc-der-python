import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_iterative

sys.path.append("../4_SupervizedLearning/iterative")
from model import LSTMMLPModel

# Load data using the function
(train_data, train_labels, test_data, test_labels,
 delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
 traj_pos_mean, traj_pos_std) = load_data_iterative("../3_ExpandDataSet/raw_data", normalize=False, subset=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# To load the model
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = LSTMMLPModel(
        input_size_lstm=checkpoint['input_size_lstm'], 
        input_size_classic=checkpoint['input_size_classic'], 
        hidden_size_lstm=checkpoint['hidden_size_lstm'], 
        hidden_size_mlp=checkpoint['hidden_size_mlp'], 
        num_layers_lstm=checkpoint['num_layers_lstm'], 
        num_layers_mlp=checkpoint['num_layers_mlp'], 
        output_size=checkpoint['output_size'],
        delta_actions_mean=checkpoint['delta_actions_mean'].to(device), 
        delta_actions_std=checkpoint['delta_actions_std'].to(device), 
        delta_goals_mean=checkpoint['delta_goals_mean'].to(device), 
        delta_goals_std=checkpoint['delta_goals_std'].to(device), 
        traj_pos_mean=checkpoint['traj_pos_mean'].to(device), 
        traj_pos_std=checkpoint['traj_pos_std'].to(device)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=checkpoint['learning_rate'], momentum=checkpoint['momentum'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['train_losses'], checkpoint['test_losses']

# Example of loading the model
loaded_model, loaded_optimizer, loaded_train_losses, loaded_test_losses = load_model('../4_SupervizedLearning/iterative/model_checkpoint_90.pth')
loaded_model = loaded_model.to(device)

# Convert numpy arrays to torch tensors
train_data_time_series = torch.tensor(train_data['time_series'], dtype=torch.float32).to(device)
train_data_classic = torch.tensor(train_data['classic'], dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

test_data_time_series = torch.tensor(test_data['time_series'], dtype=torch.float32).to(device)
test_data_classic = torch.tensor(test_data['classic'], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)

# Define a function to calculate MSE and get predictions
def calculate_mse_and_predictions(model, time_series_data, classic_data, targets):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(time_series_data, classic_data, test=True)
        mse = nn.MSELoss()(predictions, targets)
    return mse.item(), predictions.cpu().numpy()

# Calculate training and testing MSE and get predictions
train_mse, predicted_train_goals = calculate_mse_and_predictions(loaded_model, train_data_time_series, train_data_classic, train_labels)
test_mse, predicted_test_goals = calculate_mse_and_predictions(loaded_model, test_data_time_series, test_data_classic, test_labels)

print(f'Training MSE: {train_mse}')
print(f'Testing MSE: {test_mse}')

# Plot goals and predicted goals for a subset of points
num_points = 100

# Get the subset of data (first 100 points)
goals_train_subset = train_labels[:num_points].cpu().numpy()
goals_test_subset = test_labels[:num_points].cpu().numpy()

# Plot the actual and predicted goals
plt.figure(figsize=(16, 6))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(goals_train_subset[:, 0], goals_train_subset[:, 1], c='blue', s=10, label='Training Goals')
plt.scatter(predicted_train_goals[:num_points, 0], predicted_train_goals[:num_points, 1], c='cyan', s=10, label='Predicted Training Goals')
for i in range(num_points):
    plt.plot([goals_train_subset[i, 0], predicted_train_goals[i, 0]], 
             [goals_train_subset[i, 1], predicted_train_goals[i, 1]], 
             'k-', linewidth=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Training Goals and Predictions')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Testing data
plt.subplot(1, 2, 2)
plt.scatter(goals_test_subset[:, 0], goals_test_subset[:, 1], c='red', s=10, label='Testing Goals')
plt.scatter(predicted_test_goals[:num_points, 0], predicted_test_goals[:num_points, 1], c='orange', s=10, label='Predicted Testing Goals')
for i in range(num_points):
    plt.plot([goals_test_subset[i, 0], predicted_test_goals[i, 0]], 
             [goals_test_subset[i, 1], predicted_test_goals[i, 1]], 
             'k-', linewidth=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Testing Goals and Predictions')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot distributions of actions_train and actions_test
plt.figure(figsize=(16, 6))

# Plot histograms for each dimension of actions_train
for i in range(train_data['classic'].shape[1]):
    plt.subplot(1, train_data['classic'].shape[1], i + 1)
    plt.hist(train_data['classic'][:, i], bins=50, alpha=0.7, color='blue')
    plt.xlabel(f'Value of Dimension {i}')
    plt.ylabel('Frequency')
    plt.title(f'Actions Train Dimension {i}')
    plt.grid(True)

# Plot distributions of actions_test
plt.figure(figsize=(16, 6))

# Plot histograms for each dimension of actions_test
for i in range(test_data['classic'].shape[1]):
    plt.subplot(1, test_data['classic'].shape[1], i + 1)
    plt.hist(test_data['classic'][:, i], bins=50, alpha=0.7, color='red')
    plt.xlabel(f'Value of Dimension {i}')
    plt.ylabel('Frequency')
    plt.title(f'Actions Test Dimension {i}')
    plt.grid(True)

plt.tight_layout()
plt.show()
