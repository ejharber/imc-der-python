import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) +  "/../UR5e")
from CustomRobots import *

import matplotlib.pyplot as plt

def load_data_zeroshot(folder_name, seed=0, normalize=True):
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
    if normalize:
        actions = (actions - actions_mean) / actions_std

    # Calculate mean and standard deviation of goals
    goals_mean = np.mean(goals, axis=0)
    goals_std = np.std(goals, axis=0)
    
    # Normalize goals
    if normalize:
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

def load_realworlddata_zeroshot(filter_data_file_name, params_file_name, seed=0):

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/filtered_data/"
    filter_data = np.load(os.path.join(file_path, filter_data_file_name) + ".npz")

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/params/"
    params = np.load(os.path.join(file_path, params_file_name) + ".npz")["params"]

    np.random.seed(seed)

    actions = filter_data["qf_save"][:, [1, 2, 3]]
    goals = filter_data["traj_rope_tip_save"][:, round(params[-1] + 500), :]
    
    # Return normalized data, means, and stds for train and test sets
    return (actions, goals)

####################################################################################################

def load_data_iterative(filter_data_file_name, params_file_name, seed=0, normalize=True, subset=1000):
    np.random.seed(seed)

    # Initialize containers
    all_delta_actions = []
    all_delta_goals = []
    all_delta_traj = []

    # Paths for data files
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../3_ExpandDataSet/", filter_data_file_name)
    params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../2_SysID/params/", params_file_name + ".npz")

    # Load parameters and standard deviations
    params = np.load(params_path)["params"]
    std_x = np.load(params_path)["mocap_x_costs_std"][-1]
    std_y = np.load(params_path)["mocap_y_costs_std"][-1]
    std_ati = np.load(params_path)["ati_costs_std"][-1]

    # Iterate through the data files
    count = 0
    for file in os.listdir(folder_path):
        if not file.endswith(".npz"):
            continue

        data = np.load(os.path.join(folder_path, file))

        actions = data["qf_save"][:100, [1, 2, 3]]
        goals = data["traj_pos_save"][:100, -1, :]
        traj_pos = data["traj_pos_save"][:100, :, :]
        traj_force = data["traj_force_save"][:100, :, :]

        # Combine position and force trajectories
        traj_data = np.append(traj_pos, traj_force, axis=2)

        # Add noise to trajectories
        traj_data[:, :, 0] += np.random.normal(0, std_x, traj_data[:, :, 0].shape)
        traj_data[:, :, 1] += np.random.normal(0, std_y, traj_data[:, :, 1].shape)
        traj_data[:, :, 2] += np.random.normal(0, std_ati, traj_data[:, :, 2].shape)

        # Compute pairwise differences for actions, goals, and trajectories
        delta_actions = actions[np.newaxis, :, :] - actions[:, np.newaxis, :]
        delta_goals = goals[np.newaxis, :, :] - goals[:, np.newaxis, :]
        delta_traj = 0*traj_data[np.newaxis, :, :, :] + traj_data[:, np.newaxis, :, :]

        # Reshape and append
        all_delta_actions.append(delta_actions.reshape(-1, delta_actions.shape[-1]))
        all_delta_goals.append(delta_goals.reshape(-1, delta_goals.shape[-1]))
        all_delta_traj.append(delta_traj.reshape(-1, delta_traj.shape[2], delta_traj.shape[3]))

        count += 1
        print(count)
        if subset == count:
            break

    # Concatenate all collected data
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    all_delta_goals = np.concatenate(all_delta_goals, axis=0)
    all_delta_traj = np.concatenate(all_delta_traj, axis=0)

    print(all_delta_actions.shape, all_delta_goals.shape, all_delta_traj.shape)

    # Normalize data if required
    if normalize:
        delta_actions_mean, delta_actions_std = np.mean(all_delta_actions, axis=0), np.std(all_delta_actions, axis=0)
        delta_goals_mean, delta_goals_std = np.mean(all_delta_goals, axis=0), np.std(all_delta_goals, axis=0)
        traj_mean, traj_std = np.mean(all_delta_traj, axis=(0, 1)), np.std(all_delta_traj, axis=(0, 1))

        all_delta_actions = (all_delta_actions - delta_actions_mean) / delta_actions_std
        all_delta_goals = (all_delta_goals - delta_goals_mean) / delta_goals_std
        all_delta_traj = (all_delta_traj - traj_mean) / traj_std
    else:
        delta_actions_mean = delta_actions_std = delta_goals_mean = delta_goals_std = traj_mean = traj_std = None

    # Randomize and split data into train and test sets
    indices = np.arange(all_delta_actions.shape[0])
    np.random.shuffle(indices)
    split = int(len(indices) * 2 / 3)
    train_indices, test_indices = indices[:split], indices[split:]

    train_data = {"time_series": all_delta_traj[train_indices], "classic": all_delta_actions[train_indices]}
    test_data = {"time_series": all_delta_traj[test_indices], "classic": all_delta_actions[test_indices]}
    train_labels, test_labels = all_delta_goals[train_indices], all_delta_goals[test_indices]

    return (train_data, train_labels, test_data, test_labels,
            delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
            traj_mean, traj_std)

def load_realworlddata_iterative(filter_data_file_name, params_file_name, seed=0, subset=1000):
    np.random.seed(seed)

    # Initialize containers
    all_delta_actions = []
    all_delta_goals = []
    all_delta_traj = []

    # Paths for data files
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../2_SysID/filtered_data_iter/", filter_data_file_name)
    params_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../2_SysID/params/", params_file_name + ".npz")

    # Load parameters and standard deviations
    params = np.load(params_path)["params"]

    # Iterate through the data files
    count = 0
    for file in os.listdir(folder_path):
        if not file.endswith(".npz"):
            continue

        data = np.load(os.path.join(folder_path, file))

        actions = data["qf_save"][:, [1, 2, 3]]
        goals = data["traj_rope_tip_save"][:, round(params[-1] + 500), :]
        traj_pos = data["traj_rope_tip_save"][:, round(params[-1]):round(params[-1] + 500), :]
        traj_force = data["traj_force_save"][:, round(params[-2]):round(params[-2] + 500), :]

        # Combine position and force trajectories
        traj_data = np.append(traj_pos, traj_force, axis=2)

        # Compute pairwise differences for actions, goals, and trajectories
        delta_actions = actions[np.newaxis, :, :] - actions[:, np.newaxis, :]
        delta_goals = goals[np.newaxis, :, :] - goals[:, np.newaxis, :]
        delta_traj = 0*traj_data[np.newaxis, :, :, :] + traj_data[:, np.newaxis, :, :]

        # Reshape and append
        all_delta_actions.append(delta_actions.reshape(-1, delta_actions.shape[-1]))
        all_delta_goals.append(delta_goals.reshape(-1, delta_goals.shape[-1]))
        all_delta_traj.append(delta_traj.reshape(-1, delta_traj.shape[2], delta_traj.shape[3]))

        count += 1
        print(count)
        if subset == count:
            break

    # Concatenate all collected data
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    all_delta_goals = np.concatenate(all_delta_goals, axis=0)
    all_delta_traj = np.concatenate(all_delta_traj, axis=0)

    print(all_delta_actions.shape, all_delta_goals.shape, all_delta_traj.shape)

    # Organize data into a dictionary
    valid_data = {
        "time_series": all_delta_traj,
        "classic": all_delta_actions
    }
    valid_labels = all_delta_goals

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
    traj = traj[half_index:]

    print(delta_actions.shape, delta_goals.shape, delta_traj.shape)

    # Organize data into a dictionary
    valid_data = {
        "time_series": traj,
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
    delta_goals_full = labels_full

    delta_actions_subset = data_subset["classic"]
    delta_goals_subset = labels_subset

    # Initialize a flag to track if all matching subsets have the same delta_goals
    all_goals_match = True

    # Iterate over each point in the subset data
    for i in range(len(delta_actions_subset)):
        print(i)
        actions_point = delta_actions_subset[i]
        goals_point = delta_goals_subset[i]

        # Find approximate matches for delta_actions in the full dataset
        matches = [
            j for j in range(len(delta_actions_full))
            if np.allclose(delta_actions_full[j], actions_point, atol=atol, rtol=rtol)
        ]

        # If matches are found, check delta_traj and delta_goals consistency
        if not matches:
            print(f"No approximate match found for subset index {i}")
            all_goals_match = False
        else:
            # Verify delta_traj and delta_goals consistency for all matches
            # traj_match = any(np.allclose(delta_traj_full[j], traj_point, atol=atol, rtol=rtol) for j in matches)
            goals_match = any(np.allclose(delta_goals_full[j], goals_point, atol=atol, rtol=rtol) for j in matches)

            for j in matches: print(delta_goals_full[j], goals_point)
            # print(delta_actions_full)


            if not goals_match:
                print(f"Mismatch in delta_traj or delta_goals for subset index {i}")
                all_goals_match = False

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