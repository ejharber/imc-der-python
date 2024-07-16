import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

sys.path.append("../4_SupervizedLearning/zero_shot")
from model import SimpleMLP

import sys
sys.path.append("../gym/")
from rope import Rope

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
    return model, optimizer

# Example of loading the model
loaded_model, loaded_optimizer = load_model('../4_SupervizedLearning/zero_shot/model_checkpoint_50.pth')
loaded_model = loaded_model.to(device)

# Convert numpy arrays to torch tensors
actions_test_tensor = torch.tensor(actions_test, dtype=torch.float32).to(device)

# Function to sample 10,000 actions
def sample_actions(num_samples=10000):
    qf = np.array([-90, 100, -180], dtype=np.float32)
    random_actions = np.tile(qf, (num_samples, 1))
    random_actions[:, 0] += np.random.rand(num_samples) * 20 - 10
    random_actions[:, 1] += np.random.rand(num_samples) * 20 - 10
    random_actions[:, 2] += np.random.rand(num_samples) * 24 - 12
    return random_actions

# Evaluate model on randomly sampled goal and actions
def evaluate_model_on_random_goal(model, actions_tensor, goals_tensor, num_samples=10000):
    model.eval()
    with torch.no_grad():
        # Sample a random goal from the test set
        random_goal_idx = np.random.randint(len(goals_tensor))
        random_goal = goals_tensor[random_goal_idx]

        # Sample 10000 random actions
        random_actions = sample_actions(num_samples)
        random_actions = torch.tensor(random_actions, dtype=torch.float32).to(device)

        # Predict goals for each random action
        predicted_goals = model(random_actions, test=True)

        # Calculate distances between each predicted goal and the random goal
        distances = torch.norm(predicted_goals - random_goal, dim=1)

        # Find the index of the minimum distance
        min_distance_idx = torch.argmin(distances)

        # Get the best action
        best_action = random_actions[min_distance_idx]

        # Calculate the error between the desired goal and the one produced by the brute force sample algorithm
        best_predicted_goal = predicted_goals[min_distance_idx]
        error = torch.norm(best_predicted_goal - random_goal).item()

        return best_action.cpu().numpy(), random_goal.cpu().numpy(), error

# Initialize rope and load parameters
params = np.load("../2_SysID/res_all_noise.npz")["x"]
rope = Rope(params)

# Loop to evaluate the model 100 times
all_errors = []
for i in range(100):
    best_action, random_goal, error = evaluate_model_on_random_goal(loaded_model, actions_test_tensor, torch.tensor(goals_test, dtype=torch.float32).to(device))
    all_errors.append(error)
    
    print(f"Iteration {i+1}:")
    print(f"  Best Action Predicted for Random Goal: {best_action}")
    print(f"  Error between desired goal and predicted goal: {error}")

    q0 = [180, -53.25, 134.66, -171.28, -90, 0]
    qf = [180, -90, 100, -180, -90, 0]
    qf[1] = best_action[0]         
    qf[2] = best_action[1]         
    qf[3] = best_action[2]         

    success, traj_pos_sim, traj_force_sim, q_save, f_save = rope.run_sim(q0, qf)

    # Render pose
    print(random_goal)
    rope.render(q_save, goal=random_goal)

# Calculate average error
average_error = np.mean(all_errors)
print(f"Average error over 100 evaluations: {average_error}")
