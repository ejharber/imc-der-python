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
    filtered_files = {j: [] for j in j_values}  # Initialize a dictionary for each desired `j`
    for file in os.listdir(directory):
        if file.endswith('.npz'):
            if int(file.split("_")[0]) >60: continue
            file_j = int(file.split("_")[1].split(".")[0])  # Extract `j` from `i_j.npz`
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
    :return: The average MSE for this `j` value.
    """
    mse_values = []

    plt.figure(figsize=(10, 10))

    for data_file in data_files:
        # Load data and parameters
        data = np.load(data_file)
        params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

        # Main goal (gold `+`) position
        main_goal_idx = 1000 + round(params[-1])
        main_goal_position = data["mocap_data_robot_save"][main_goal_idx, :]  # (x, y)

        # Secondary goal (green `+`) position
        secondary_goal_position = data["goal_robot_save"]  # (x, y)

        # Calculate the MSE between the main and secondary goal positions
        mse_values.append(np.linalg.norm(main_goal_position - secondary_goal_position))

        # Plot secondary goal (green `+`)
        plt.plot(
            secondary_goal_position[0],
            secondary_goal_position[1],
            color="green",
            marker="+",
            markersize=12,
            label="Secondary Goal (Green +)" if data_file == data_files[0] else None,
        )

        # Plot main goal (gold `+`)
        plt.plot(
            main_goal_position[0],
            main_goal_position[1],
            color="gold",
            marker="+",
            markersize=12,
            label="Main Goal (Gold +)" if data_file == data_files[0] else None,
        )

        # Connect the goals with a dashed line
        plt.plot(
            [secondary_goal_position[0], main_goal_position[0]],
            [secondary_goal_position[1], main_goal_position[1]],
            linestyle="--",
            color="gray",
            alpha=0.7,
        )

    # Equal axis for proper scaling
    plt.axis("equal")

    # Add labels, legend, and title
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.title(f"Goals and Mocap Data for j = {j_value}")
    plt.legend()
    plt.grid(True)

    # Return the average MSE for this j value
    return np.mean(mse_values)

def plot_average_mse(average_mse_values, j_values):
    """
    Plot the average MSE vs. the j values.

    :param average_mse_values: List of average MSE values for each j.
    :param j_values: List of corresponding j values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(j_values, average_mse_values, marker='o', color='blue', label="Average MSE")
    plt.xlabel("j Value")
    plt.ylabel("Average MSE")
    plt.title("Average MSE Between Main and Secondary Goal vs j")
    plt.grid(True)
    plt.legend()

def main():
    input_dir = '../../5_Evaluation/N2_pose_iterative/'  # Specify the directory to search
    desired_j_values = [0, 1, 2, 3, 4]  # Specify the desired `j` values

    # Find the files matching the desired `j` values
    data_files_by_j = filter_files_by_j(input_dir, desired_j_values)

    average_mse_values = []

    # Calculate the average MSE for each j value and plot the goals
    for j_value in desired_j_values:
        files = data_files_by_j.get(j_value, [])
        if files:
            print(f"Found {len(files)} files for j = {j_value}.")
            # Plot the goals for each j value
            avg_mse = plot_goals_and_mocap(files, params_file_name="N3", j_value=j_value)
            average_mse_values.append(avg_mse)
        else:
            print(f"No files found for j = {j_value} in {input_dir}.")

    # Plot the average MSE vs j values
    plot_average_mse(average_mse_values, desired_j_values)

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
