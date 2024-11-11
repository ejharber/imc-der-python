import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) +  "/../UR5e")
from CustomRobots import *

import matplotlib.pyplot as plt

def load_data_zeroshot(folder_name, seed=0, noramlize=True):
    np.random.seed(seed)  # Set the random seed for reproducibility

    actions = np.empty((0, 3))  # Predefine empty array for actions
    goals = np.empty((0, 2))    # Predefine empty array for goals

    folder_path = os.path.dirname(os.path.realpath(__file__)) + "/../3_ExpandDataSet/" + folder_name

    for file in os.listdir(folder_path):
        if not file.endswith(".npz"):
            continue 

        data = np.load(os.path.join(folder_path, file))

        # Append actions and goals using np.append
        actions = np.append(actions, data["qf_save"][:, [1, 2, 3]], axis=0)
        goals = np.append(goals, data["traj_pos_save"][:, -1, :], axis=0)

    print(actions.shape, goals.shape)
    
    # Calculate mean and standard deviation of actions
    actions_mean = np.mean(actions, axis=0)
    actions_std = np.std(actions, axis=0)
    
    # Normalize actions
    if noramlize:
        actions = (actions - actions_mean) / actions_std

    # Calculate mean and standard deviation of goals
    goals_mean = np.mean(goals, axis=0)
    goals_std = np.std(goals, axis=0)
    
    # Normalize goals
    if noramlize:
        goals = (goals - goals_mean) / goals_std

    # Create a list of indices
    indices = np.arange(actions.shape[0])

    # Randomize the list of indices
    np.random.shuffle(indices)

    # Sample the first 2/3 and last 1/3
    split = int(len(indices) * 2 / 3)
    train_indices = indices[:split]
    test_indices = indices[split:]

    # Return normalized actions, goals, means, and stds for train and test sets
    return (actions[train_indices], goals[train_indices],
            actions[test_indices], goals[test_indices],
            actions_mean, actions_std, goals_mean, goals_std)

def load_realworlddata_zeroshot(file_name, seed=0):

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/filtered_data/"
    data = np.load(os.path.join(file_path, file_name) + ".npz")

    np.random.seed(seed)

    actions = data["qf_save"][:, [1, 2, 3]]
    goals = data["traj_rope_tip_save"][:, -1, :]
    
    # Return normalized data, means, and stds for train and test sets
    return (actions, goals)

####################################################################################################

def load_data_iterative(folder_name, seed=0, normalize=True, subset=False):
    np.random.seed(seed)

    delta_actions = np.empty((0, 3))
    delta_goals = np.empty((0, 2))
    traj = np.empty((0, 500, 3))  # Assuming traj has shape based on data contents

    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../3_ExpandDataSet/", folder_name)

    for file in os.listdir(folder_path):
        if not file.endswith(".npz"):
            continue 

        data = np.load(os.path.join(folder_path, file))

        actions = data["qf_save"][:400, [1, 2, 3]]
        goals = data["traj_pos_save"][:400, -1, :]

        split = 200

        # Compute deltas and append
        delta_actions = np.append(delta_actions, actions[:split, :] - actions[split:, :], axis=0)
        delta_goals = np.append(delta_goals, goals[:split, :] - goals[split:, :], axis=0)

        # Combine position and force trajectories and add noise
        traj_data = np.append(data["traj_pos_save"][:split, :, :], data["traj_force_save"][:split, :, :], axis=2)

        # print(traj.shape, traj_data.shape,data["traj_pos_save"][:split, :, :].shape, data["traj_force_save"][:split, :, :].shape)
        traj = np.append(traj, traj_data, axis=0)

        if subset:
            break

    # Calculate mean and standard deviation for normalization
    delta_actions_mean, delta_actions_std = np.mean(delta_actions, axis=0), np.std(delta_actions, axis=0)
    delta_goals_mean, delta_goals_std = np.mean(delta_goals, axis=0), np.std(delta_goals, axis=0)
    traj_mean, traj_std = np.mean(traj, axis=(0, 1)), np.std(traj, axis=(0, 1))

    if normalize:
        delta_actions = (delta_actions - delta_actions_mean) / delta_actions_std
        delta_goals = (delta_goals - delta_goals_mean) / delta_goals_std
        traj = (traj - traj_mean) / traj_std

    # Randomize and split data into train and test sets
    indices = np.arange(delta_actions.shape[0])
    np.random.shuffle(indices)
    split = int(len(indices) * 2 / 3)
    train_indices, test_indices = indices[:split], indices[split:]

    train_data = {"time_series": traj[train_indices], "classic": delta_actions[train_indices]}
    test_data = {"time_series": traj[test_indices], "classic": delta_actions[test_indices]}
    
    train_labels, test_labels = delta_goals[train_indices], delta_goals[test_indices]

    print(delta_actions_mean.shape, delta_actions_std.shape, delta_goals_mean.shape, delta_goals_std.shape,
          traj_mean.shape, traj_std.shape)

    return (train_data, train_labels, test_data, test_labels,
            delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
            traj_mean, traj_std)

def load_realworlddata_iterative(file_name, seed=0):
    np.random.seed(seed)

    # Define file path and load data
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../2_SysID/filtered_data/")
    data = np.load(os.path.join(file_path, file_name) + ".npz")

    # Extract actions, goals, and trajectories
    actions = data["qf_save"][:, [1, 2, 3]]
    goals = data["traj_rope_tip_save"][:, -1, :]
    traj_pos = data["traj_rope_tip_save"]
    traj_force = data["traj_force_save"]
    traj = np.append(traj_pos, traj_force, axis=2)

    # Compute delta actions, goals, and trajectories
    delta_actions = actions[:, np.newaxis, :] - actions[np.newaxis, :, :]
    delta_goals = goals[:, np.newaxis, :] - goals[np.newaxis, :, :]
    delta_traj = traj[:, np.newaxis, :, :] - traj[np.newaxis, :, :, :]

    # Reshape deltas for concatenation
    delta_actions = delta_actions.reshape(-1, delta_actions.shape[-1])
    delta_goals = delta_goals.reshape(-1, delta_goals.shape[-1])
    delta_traj = delta_traj.reshape(-1, delta_traj.shape[2], delta_traj.shape[3])

    # Organize data into a dictionary
    valid_data = {
        "time_series": delta_traj,
        "classic": delta_actions
    }
    valid_labels = delta_goals

    return valid_data, valid_labels

def load_realworlddata_iterative_check(file_name, seed=0):
    np.random.seed(seed)

    # Define file path and load data
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../2_SysID/filtered_data/")
    data = np.load(os.path.join(file_path, file_name) + ".npz")

    # Extract actions, goals, and trajectories
    actions = data["qf_save"][:, [1, 2, 3]]
    goals = data["traj_rope_tip_save"][:, -1, :]
    traj_pos = data["traj_rope_tip_save"]
    traj_force = data["traj_force_save"]
    traj = np.append(traj_pos, traj_force, axis=2)

    # Find the halfway index
    half_index = len(actions) // 2

    # Compute differences between the first half and second half
    delta_actions = actions[half_index:] - actions[:half_index]
    delta_goals = goals[half_index:] - goals[:half_index]
    delta_traj = traj[half_index:] - traj[:half_index]

    # Reshape deltas for concatenation
    delta_actions = delta_actions.reshape(-1, delta_actions.shape[-1])
    delta_goals = delta_goals.reshape(-1, delta_goals.shape[-1])
    delta_traj = delta_traj.reshape(-1, delta_traj.shape[2], delta_traj.shape[3])

    # Organize data into a dictionary
    valid_data = {
        "time_series": delta_traj,
        "classic": delta_actions
    }
    valid_labels = delta_goals

    return valid_data, valid_labels

def validate_realworld_data(file_name, seed=0, atol=1e-8, rtol=1e-5):
    # Load data using both functions
    data_full, labels_full = load_realworlddata_iterative(file_name, seed=seed)
    data_subset, labels_subset = load_realworlddata_iterative_check(file_name, seed=seed)
    
    # Extract delta_actions, delta_traj, and delta_goals from both datasets
    delta_actions_full = data_full["classic"]
    delta_traj_full = data_full["time_series"]
    delta_goals_full = labels_full

    delta_actions_subset = data_subset["classic"]
    delta_traj_subset = data_subset["time_series"]
    delta_goals_subset = labels_subset

    # Initialize a flag to track if all matching subsets have the same delta_goals
    all_goals_match = True

    # Iterate over each point in the subset data
    for i in range(len(delta_actions_subset)):
        actions_point = delta_actions_subset[i]
        traj_point = delta_traj_subset[i]
        goals_point = delta_goals_subset[i]

        print(actions_point.shape, actions_point)
        # Find approximate matches for delta_actions in the full dataset
        matches = [
            j for j in range(len(delta_actions_full))
            if np.allclose(delta_actions_full[j], actions_point, atol=atol, rtol=rtol)
        ]

        print(matches)

        # If matches are found, check delta_traj and delta_goals consistency
        if not matches:
            print(f"No approximate match found for subset index {i}")
            all_goals_match = False
        else:
            print(delta_actions_full[matches[0]], actions_point)
            # # Verify delta_traj and delta_goals consistency for all matches
            # traj_match = any(np.allclose(delta_traj_full[j], traj_point, atol=atol, rtol=rtol) for j in matches)
            # goals_match = all(np.allclose(delta_goals_full[j], goals_point, atol=atol, rtol=rtol) for j in matches)

            # if not traj_match or not goals_match:
            #     print(f"Mismatch in delta_traj or delta_goals for subset index {i}")
            #     all_goals_match = False

    if all_goals_match:
        print("All matching subset points have consistent delta_traj and delta_goals in the full dataset.")
    else:
        print("There were inconsistencies in delta_traj or delta_goals for some matching points.")

    return all_goals_match

if __name__ == '__main__':
    # load_data_zeroshot("N2_all")
    # load_realworlddata_zeroshot("N2")
    # load_data_iterative("N2_all")
    # load_realworlddata_iterative_check()
    # out = load_realworlddata_iterative("N2")
    # print(out)
    # out = load_realworlddata_iterative_check("N2")
    # print(out)
    validate_realworld_data("N2")