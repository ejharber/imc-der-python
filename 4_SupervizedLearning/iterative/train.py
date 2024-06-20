import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Assuming load_data is correctly defined
import os
import sys
sys.path.append("..")
from load_data import load_data_iterative  # Assuming load_data is correctly defined

class LSTMMLPModel(nn.Module):
    def __init__(self, input_size_lstm, input_size_classic, hidden_size_lstm, hidden_size_mlp, num_layers_lstm, num_layers_mlp, output_size):
        super(LSTMMLPModel, self).__init__()
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm
        self.layer_norm_lstm = nn.LayerNorm(input_size_lstm)
        self.lstm = nn.LSTM(input_size_lstm, hidden_size_lstm, num_layers_lstm, batch_first=True)
        
        self.input_size_mlp = hidden_size_lstm + input_size_classic
        self.layer_norm_classic = nn.LayerNorm(input_size_classic)
        self.hidden_layers_mlp = nn.ModuleList()
        for _ in range(num_layers_mlp):
            self.hidden_layers_mlp.append(nn.Linear(self.input_size_mlp, hidden_size_mlp))
            self.input_size_mlp = hidden_size_mlp
        
        self.output_layer = nn.Linear(hidden_size_mlp, output_size)
        self.relu = nn.ReLU()

    def forward(self, x_lstm, x_classic):
        x_lstm = self.layer_norm_lstm(x_lstm)
        h0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        c0 = torch.zeros(self.num_layers_lstm, x_lstm.size(0), self.hidden_size_lstm).to(x_lstm.device)
        
        out_lstm, _ = self.lstm(x_lstm, (h0, c0))
        out_lstm = out_lstm[:, -1, :]  # Take the output from the last time step
        
        # Layer normalization for classic input
        x_classic = self.layer_norm_classic(x_classic)
        
        # Concatenate LSTM output with classic input
        combined_input = torch.cat((out_lstm, x_classic), dim=1)

        # MLP layers
        for layer in self.hidden_layers_mlp:
            combined_input = layer(combined_input)
            combined_input = self.relu(combined_input)
        
        out = self.output_layer(combined_input)
        
        return out

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
train_data, train_labels, test_data, test_labels = load_data_iterative("../../3_ExpandDataSet/raw_data")
train_data_time_series = torch.tensor(train_data['time_series'], dtype=torch.float32)  # Use float32 for LSTM input
train_data_classic = torch.tensor(train_data['classic'], dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

test_data_time_series = torch.tensor(test_data['time_series'], dtype=torch.float32)
test_data_classic = torch.tensor(test_data['classic'], dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Determine input sizes and output size from data
input_size_lstm = train_data_time_series.shape[2]  # Assuming the time series data is in (samples, timesteps, features) format
input_size_classic = train_data_classic.shape[1]  # Assuming classic data is in (samples, features) format
output_size = train_labels.shape[1]

# Create DataLoader for batching
batch_size = 1000
train_dataset = TensorDataset(train_data_time_series, train_data_classic, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_data_time_series, test_data_classic, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = LSTMMLPModel(input_size_lstm, input_size_classic, hidden_size_lstm=100, hidden_size_mlp=500, num_layers_lstm=2, num_layers_mlp=3, output_size=output_size).to(device)
criterion = nn.MSELoss()  # Use Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training and evaluation loop
num_epochs = 200
train_losses = []
test_losses = []
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
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# Plot the training and testing loss curves
plot_curves(train_losses, test_losses)

# Save the trained model
torch.save(model.state_dict(), 'lstm_mlp_model.pth')
