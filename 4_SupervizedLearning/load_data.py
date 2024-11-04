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

def load_realworlddata_iterative(file_name, seed=0):

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/filtered_data/"
    data = np.load(os.path.join(file_path, file_name))

    np.random.seed(seed)

    actions = data["qf_save"][:, [1, 2, 3]]
    goals = data["traj_pos_save"][:, -1, :]
    traj_pos = data["traj_pos_save"][:, :, :]
    traj_force = data["traj_force_save"][:, :, :]
    traj = np.append(traj_pos, traj_force, axis=2)

    # Calculate delta actions
    delta_actions = actions[:, np.newaxis, :] - actions[np.newaxis, :, :]

    # Calculate delta goals
    delta_goals = goals[:, np.newaxis, :] - goals[np.newaxis, :, :]
    
    # Calculate delta trajectory positions
    delta_traj = traj[:, np.newaxis, :, :] - traj[np.newaxis, :, :, :] * 0

    delta_actions = np.reshape(delta_actions, (delta_actions.shape[0]*delta_actions.shape[1], delta_actions.shape[2]))
    delta_goals = np.reshape(delta_goals, (delta_goals.shape[0]*delta_goals.shape[1], delta_goals.shape[2]))
    delta_traj = np.reshape(delta_traj, (delta_traj.shape[0]*delta_traj.shape[1], delta_traj.shape[2], delta_traj.shape[3]))

    valid_data = dict()
    valid_data["time_series"] = delta_traj
    valid_data["classic"] = delta_actions

    valid_labels = delta_goals

    # Return normalized data, means, and stds for train and test sets
    return (valid_data, valid_labels)

def load_realworlddata_iterative_check(seed=0):
    # Assuming UR5eCustom is defined somewhere in your code
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../3_ExpandDataSet/N2_all_expanded_data"

    np.random.seed(seed)

    actions = []
    goals = []
    traj_pos = []

    for file in os.listdir(file_path):
        if not file.endswith(".npz"): continue 

        data = np.load(os.path.join(file_path, file))

        actions = data["qf_save"][:500, [1, 2, 3]]
        goals = data["traj_pos_save"][:500, -1, :]
        traj_pos = data["traj_pos_save"][:500, :, :]
        traj_force = data["traj_force_save"][:500, :, :]
        traj = np.append(traj_pos, traj_force, axis=2)

        break

    actions = np.array(actions)
    goals = np.array(goals)
    traj = np.array(traj)

    print(actions.shape, goals.shape, traj_pos.shape)

    # Calculate delta actions
    delta_actions = actions[:, np.newaxis, :] - actions[np.newaxis, :, :]

    # Calculate delta goals
    delta_goals = goals[:, np.newaxis, :] - goals[np.newaxis, :, :]
    
    # Calculate delta trajectory positions
    delta_traj = traj[:, np.newaxis, :, :] - traj[np.newaxis, :, :, :] * 0

    delta_actions = np.reshape(delta_actions, (delta_actions.shape[0]*delta_actions.shape[1], delta_actions.shape[2]))
    delta_goals = np.reshape(delta_goals, (delta_goals.shape[0]*delta_goals.shape[1], delta_goals.shape[2]))
    delta_traj = np.reshape(delta_traj, (delta_traj.shape[0]*delta_traj.shape[1], delta_traj.shape[2], delta_traj.shape[3]))

    valid_data = dict()
    valid_data["time_series"] = delta_traj
    valid_data["classic"] = delta_actions

    valid_labels = delta_goals

    # Return normalized data, means, and stds for train and test sets
    return (valid_data, valid_labels)

def load_data_iterative(folder_name, seed=0, normalize=True, subset=False):
    np.random.seed(seed)

    delta_actions = []
    delta_goals = []
    traj = []

    for file in os.listdir(folder_name):
        if not file.endswith(".npz"): continue 

        data = np.load(os.path.join(folder_name, file))

        actions = data["qf_save"][:, [1, 2, 3]]
        goals = data["traj_pos_save"][:, -1, :]

        split = int(actions.shape[0] / 2)

        delta_actions.append(actions[:split, :] - actions[split:, :])
        delta_goals.append(goals[:split, :] - goals[split:, :])
        # delta_goals.append(goals[split:, :])

        traj_pos = data["traj_pos_save"][:split, :, :]
        traj_force = data["traj_force_save"][:split, :, :]
        traj.append(np.append(traj_pos, traj_force, axis=2))

        if subset:
            break

    delta_actions = np.array(delta_actions)
    delta_actions = np.reshape(delta_actions, (delta_actions.shape[0]*delta_actions.shape[1], delta_actions.shape[2]))
    # purge = np.where(np.max(delta_actions, axis=1) < 2.0)[0]
    # print(delta_actions.shape)
    # print((np.where(np.max(delta_actions, axis=1) < 2.0)[0]).shape)
    # exit()
    # delta_actions = delta_actions[purge, :]

    # Calculate mean and standard deviation of delta_actions
    delta_actions_mean = np.mean(delta_actions, axis=0)
    delta_actions_std = np.std(delta_actions, axis=0)
    
    # Normalize delta_actions
    if normalize:
        delta_actions = (delta_actions - delta_actions_mean) / delta_actions_std

    delta_goals = np.array(delta_goals)
    delta_goals = np.reshape(delta_goals, (delta_goals.shape[0]*delta_goals.shape[1], delta_goals.shape[2]))

    # delta_goals = delta_goals[purge, :]

    # Calculate mean and standard deviation of delta_goals
    delta_goals_mean = np.mean(delta_goals, axis=0)
    delta_goals_std = np.std(delta_goals, axis=0)
    
    # Normalize delta_goals
    if normalize:
        delta_goals = (delta_goals - delta_goals_mean) / delta_goals_std

    traj = np.array(traj)
    traj = np.reshape(traj, (traj.shape[0]*traj.shape[1], traj.shape[2], traj.shape[3]))
    traj = np.random.normal(loc=traj, scale=0.1, size=traj.shape)

    # traj_pos = traj_pos[purge, :, :]

    # Calculate mean and standard deviation of traj_pos along the third dimension
    traj_mean = np.mean(traj, axis=(0, 1))
    traj_std = np.std(traj, axis=(0, 1))

    # Normalize traj_pos
    if normalize:
        traj = (traj - traj_mean) / traj_std

    # Create a list of indices and shuffle them
    indices = np.arange(delta_actions.shape[0])
    np.random.shuffle(indices)

    # Split the shuffled indices into training and testing sets
    split = int(delta_actions.shape[0] * 2 / 3)
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_data = dict()
    train_data["time_series"] = traj[train_indices, :, :]
    train_data["classic"] = delta_actions[train_indices, :]

    train_labels = delta_goals[train_indices, :]

    test_data = dict()
    test_data["time_series"] = traj[test_indices, :, :]
    test_data["classic"] = delta_actions[test_indices, :]

    test_labels = delta_goals[test_indices, :]

    print(delta_actions_mean.shape, delta_actions_std.shape, delta_goals_mean.shape, delta_goals_std.shape,
            traj_mean.shape, traj_std.shape)

    # Return normalized data, means, and stds for train and test sets
    return (train_data, train_labels, test_data, test_labels,
            delta_actions_mean, delta_actions_std, delta_goals_mean, delta_goals_std,
            traj_mean, traj_std)

if __name__ == '__main__':
    load_data_zeroshot("N2_all")
    load_realworlddata_zeroshot("N2")
    # load_data_iterative("../3_ExpandDataSet/raw_data")
    # load_realworlddata_iterative_check()
    # load_realworlddata_iterative()