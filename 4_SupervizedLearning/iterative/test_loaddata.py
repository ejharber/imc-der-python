import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("..")
# Assume load_data and load_realworlddata_iterative are already imported
from load_data import load_data_iterative
from load_data import load_realworlddata_iterative
from load_data import load_data_zeroshot


import numpy as np
import matplotlib.pyplot as plt

def plot_actions_and_goals(actions_train, goals_train, actions_test, goals_test):
    """
    Plot actions and goals from training and testing datasets.

    Parameters:
    - actions_train, actions_test: Action arrays of shape (N, 3).
    - goals_train, goals_test: Goal arrays of shape (N, 2).
    """
    # Plot training and testing goals
    plt.figure(figsize=(10, 6))
    plt.scatter(goals_train[:, 0], goals_train[:, 1], label="Training Goals", alpha=0.7, c='blue', s=10)
    plt.scatter(goals_test[:, 0], goals_test[:, 1], label="Testing Goals", alpha=0.7, c='orange', s=10)
    plt.title("2D Scatter Plot of Goals (Train vs. Test)", fontsize=14)
    plt.xlabel("Goal X", fontsize=12)
    plt.ylabel("Goal Y", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot actions in 3 separate subplots for Training
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle("Training Actions (X, Y, Z)", fontsize=16)

    axs[0].scatter(np.arange(len(actions_train)), actions_train[:, 0], label="Action X", alpha=0.7, c='red', s=5)
    axs[0].set_title("Training Action X")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Normalized Value")
    axs[0].grid(True)

    axs[1].scatter(np.arange(len(actions_train)), actions_train[:, 1], label="Action Y", alpha=0.7, c='green', s=5)
    axs[1].set_title("Training Action Y")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Normalized Value")
    axs[1].grid(True)

    axs[2].scatter(np.arange(len(actions_train)), actions_train[:, 2], label="Action Z", alpha=0.7, c='blue', s=5)
    axs[2].set_title("Training Action Z")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel("Normalized Value")
    axs[2].grid(True)

    plt.show()

    # Plot actions in 3 separate subplots for Testing
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle("Testing Actions (X, Y, Z)", fontsize=16)

    axs[0].scatter(np.arange(len(actions_test)), actions_test[:, 0], label="Action X", alpha=0.7, c='red', s=5)
    axs[0].set_title("Testing Action X")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Normalized Value")
    axs[0].grid(True)

    axs[1].scatter(np.arange(len(actions_test)), actions_test[:, 1], label="Action Y", alpha=0.7, c='green', s=5)
    axs[1].set_title("Testing Action Y")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Normalized Value")
    axs[1].grid(True)

    axs[2].scatter(np.arange(len(actions_test)), actions_test[:, 2], label="Action Z", alpha=0.7, c='blue', s=5)
    axs[2].set_title("Testing Action Z")
    axs[2].set_xlabel("Sample Index")
    axs[2].set_ylabel("Normalized Value")
    axs[2].grid(True)

    plt.show()

# Example usage with both zeroshot and iterative loading methods
if __name__ == "__main__":
    # Test case for zeroshot loading
    folder_name_zeroshot = "N2_all"  # Adjust to your folder structure
    actions_train, goals_train, actions_test, goals_test, _, _, _, _ = load_data_zeroshot(folder_name_zeroshot, normalize=False)
    print("Testing with zeroshot data...")
    plot_actions_and_goals(actions_train, goals_train, actions_test, goals_test)

    # Test case for iterative loading
    folder_name_iterative = "N2_all"  # Adjust to your folder structure
    actions_train, goals_train, actions_test, goals_test, _, _, _, _, _, _ = load_data_iterative(folder_name_iterative, "N2_all", normalize=False)
    print("Testing with iterative data...")
    plot_actions_and_goals(actions_train['classic'], goals_train, actions_test['classic'], goals_test)
