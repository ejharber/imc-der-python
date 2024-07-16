import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

sys.path.append("../4_SupervizedLearning/zero_shot")
from model import SimpleMLP

# Load data using the function
actions_train, goals_train, actions_test, goals_test, data_mean, data_std, labels_mean, labels_std = load_data_zeroshot("../3_ExpandDataSet/raw_data", noramlize=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To load the model
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = SimpleMLP(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=checkpoint['output_size'],
        data_mean=checkpoint['data_mean'].to(device), 
        data_std=checkpoint['data_std'].to(device), 
        labels_mean=checkpoint['labels_mean'].to(device), 
        labels_std=checkpoint['labels_std'].to(device)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=checkpoint['learning_rate'], momentum=checkpoint['momentum'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['train_losses'], checkpoint['test_losses']

# Example of loading the model
loaded_model, loaded_optimizer, loaded_train_losses, loaded_test_losses = load_model('../4_SupervizedLearning/zero_shot/model_checkpoint_1000.pth')
loaded_model = loaded_model.to(device)

# Convert numpy arrays to torch tensors
actions_train_tensor = torch.tensor(actions_train, dtype=torch.float32).to(device)
goals_train_tensor = torch.tensor(goals_train, dtype=torch.float32).to(device)
actions_test_tensor = torch.tensor(actions_test, dtype=torch.float32).to(device)
goals_test_tensor = torch.tensor(goals_test, dtype=torch.float32).to(device)

# Define a function to calculate MSE and get predictions
def calculate_mse_and_predictions(model, inputs, targets):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(inputs, test=True)
        mse = nn.MSELoss()(predictions, targets)
    return mse.item(), predictions.cpu().numpy()

# Calculate training and testing MSE and get predictions
train_mse, predicted_train_goals = calculate_mse_and_predictions(loaded_model, actions_train_tensor, goals_train_tensor)
test_mse, predicted_test_goals = calculate_mse_and_predictions(loaded_model, actions_test_tensor, goals_test_tensor)

print(f'Training MSE: {train_mse}')
print(f'Testing MSE: {test_mse}')

# Plot goals and predicted goals for a subset of points
num_points = 100

# Get the subset of data (first 100 points)
goals_train_subset = goals_train[:num_points]
goals_test_subset = goals_test[:num_points]

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
for i in range(actions_train.shape[1]):
    plt.subplot(1, actions_train.shape[1], i + 1)
    plt.hist(actions_train[:, i], bins=50, alpha=0.7, color='blue')
    plt.xlabel(f'Value of Dimension {i}')
    plt.ylabel('Frequency')
    plt.title(f'Actions Train Dimension {i}')
    plt.grid(True)

# Plot distributions of actions_train and actions_test
plt.figure(figsize=(16, 6))

# Plot histograms for each dimension of actions_train
for i in range(actions_test.shape[1]):
    plt.subplot(1, actions_test.shape[1], i + 1)
    plt.hist(actions_test[:, i], bins=50, alpha=0.7, color='blue')
    plt.xlabel(f'Value of Dimension {i}')
    plt.ylabel('Frequency')
    plt.title(f'Actions Train Dimension {i}')
    plt.grid(True)


plt.tight_layout()
plt.show()
