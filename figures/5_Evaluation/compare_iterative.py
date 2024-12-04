import os
import numpy as np
import matplotlib.pyplot as plt

def filter_files_by_j(directory, j_values):
    """
    Filter .npz files in a directory based on the desired set of `j` values in filenames of format `i_j.npz`.

    :param directory: Directory to search for files.
    :param j_values: List of desired `j` values to filter.
    :return: Dictionary mapping `j` values to lists of files with those `j` values.
    """
    filtered_files = {j: [] for j in j_values}
    for file in os.listdir(directory):
        if file.endswith('.npz'):
            if int(file.split("_")[0]) > 60: continue
            file_j = int(file.split("_")[1].split(".")[0])
            if file_j in j_values:
                filtered_files[file_j].append(os.path.join(directory, file))
    return filtered_files

def plot_goals_and_mocap(data_files, params_file_name, j_value):
    """
    Plot all goal and mocap data from multiple .npz files for a specific `j` value in a single Matplotlib figure.
    Also, calculate and return the MSE for each file.

    :param data_files: List of paths to .npz data files containing goal and mocap data.
    :param params_file_name: Name of the parameter file (without extension).
    :param j_value: The `j` value corresponding to the current plot.
    :return: The average MSE and standard deviation for this `j` value.
    """
    mse_values = []

    plt.figure(figsize=(10, 10))
    for data_file in data_files:
        data = np.load(data_file)
        params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

        main_goal_idx = 1000 + round(params[-1])
        main_goal_position = data["mocap_data_robot_save"][main_goal_idx, :]
        secondary_goal_position = data["goal_robot_save"]

        mse_values.append(np.linalg.norm(main_goal_position - secondary_goal_position))

        plt.plot(
            secondary_goal_position[0],
            secondary_goal_position[1],
            color="green",
            marker="+",
            markersize=12,
            label="Secondary Goal (Green +)" if data_file == data_files[0] else None,
        )
        plt.plot(
            main_goal_position[0],
            main_goal_position[1],
            color="gold",
            marker="+",
            markersize=12,
            label="Main Goal (Gold +)" if data_file == data_files[0] else None,
        )
        plt.plot(
            [secondary_goal_position[0], main_goal_position[0]],
            [secondary_goal_position[1], main_goal_position[1]],
            linestyle="--",
            color="gray",
            alpha=0.7,
        )

    plt.axis("equal")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.title(f"Goals and Mocap Data for j = {j_value}")
    plt.legend()
    plt.grid(True)

    return np.mean(mse_values), np.std(mse_values)

import matplotlib.ticker as ticker

def plot_average_mse(mse_values_dict, j_values):
    """
    Plot the average MSE from two datasets with comparison.

    :param mse_values_dict: Dictionary containing average MSE and standard deviation for each input directory.
    :param j_values: List of corresponding j values.
    """
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'orange']
    for idx, (label, mse_data) in enumerate(mse_values_dict.items()):
        mse_values, std_values = mse_data
        mse_values_mm = [mse * 1000 for mse in mse_values]  # Convert MSE to mm
        std_values_mm = [std * 1000 for std in std_values]  # Convert Std Dev to mm
        
        plt.plot(j_values, mse_values_mm, marker='o', color=colors[idx], label=f"{label} - Average MSE")
        lower_bound = [mse - std for mse, std in zip(mse_values_mm, std_values_mm)]
        upper_bound = [mse + std for mse, std in zip(mse_values_mm, std_values_mm)]
        plt.fill_between(j_values, lower_bound, upper_bound, color=colors[idx], alpha=0.2, label=f"{label} - 1 Std. Dev.")
    
    plt.xlabel("Number of Iterations")  # Change X-axis label
    plt.ylabel("Average MSE (mm)")  # Update Y-axis label to mm
    plt.title("Comparison of Average MSE Between Main and Secondary Goals")
    plt.grid(True)
    plt.legend()

    # Set the Y-axis to integer ticks
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Set the X-axis ticks to every 1
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()


def main():
    input_dirs = ['../../5_Evaluation/N2_all_iterative/', '../../5_Evaluation/N2_pose_iterative/']
    desired_j_values = [0, 1, 2, 3, 4]
    params_file_name = "N3"

    mse_values_dict = {}

    for input_dir in input_dirs:
        label = os.path.basename(os.path.normpath(input_dir))
        data_files_by_j = filter_files_by_j(input_dir, desired_j_values)

        average_mse_values = []
        average_std_values = []

        for j_value in desired_j_values:
            files = data_files_by_j.get(j_value, [])
            if files:
                print(f"Found {len(files)} files for j = {j_value} in {label}.")
                avg_mse, avg_std = plot_goals_and_mocap(files, params_file_name, j_value)
                average_mse_values.append(avg_mse)
                average_std_values.append(avg_std)
            else:
                print(f"No files found for j = {j_value} in {label}.")

        mse_values_dict[label] = (average_mse_values, average_std_values)

    plot_average_mse(mse_values_dict, desired_j_values)

if __name__ == "__main__":
    main()
