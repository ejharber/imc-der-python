import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_video_and_data_files(directories):
    """
    Get a list of .mp4 video files and their corresponding .npz data files from multiple directories.

    :param directories: List of directory paths to search for files.
    :return: Tuple containing two lists:
             - List of .mp4 video file paths.
             - List of corresponding .npz data file paths.
    """
    video_files = []
    data_files = []
    # To track which directory a file belongs to
    directory_map = []

    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith('.mp4'):
                base_name = os.path.splitext(file)[0]
                data_file = f"{base_name}.npz"
                data_path = os.path.join(directory, data_file)
                if os.path.exists(data_path):
                    video_files.append(os.path.join(directory, file))
                    data_files.append(data_path)
                    directory_map.append(directory)  # Store which directory the data file is from

    return video_files, data_files, directory_map


def get_plus_positions_from_video(data_file, params_file_name):
    """
    Get the positions where the '+' symbols should be overlaid in each frame of the video.

    :param data_file: Path to the data file.
    :param params_file_name: Name of the parameter file (without extension).
    :return: List of (x, y) positions for the '+' symbol for each frame.
    """
    data = np.load(data_file)
    params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

    goals = data["mocap_data_camera_save"][1000 + round(params[-1]), 2, :]  # (x, y) in pixels
    goal_time = data["ros_time_save"][1000 + round(params[-1])]
    camera_time = data["ros_time_camera_save"]
    goal_frame_idx = np.argmin(abs(camera_time - goal_time))

    return goals, goal_frame_idx


def plot_all_goals_and_mocap(data_files, directory_map, params_file_name):
    """
    Plot all goal and mocap data from multiple .npz files in a single Matplotlib figure.

    :param data_files: List of paths to .npz data files containing goal and mocap data.
    :param directory_map: List mapping data files to their respective directories.
    :param params_file_name: Name of the parameter file (without extension).
    """
    plt.figure(figsize=(10, 10))

    # To track distances for each directory
    directory_distances = {"N2_pose": [], "N2_all": []}

    for idx, (data_file, directory) in enumerate(zip(data_files, directory_map)):
        # Extract the directory name from the full path
        directory_name = os.path.basename(directory)

        # Load data and parameters
        data = np.load(data_file)
        params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

        # Main goal (gold `+`) position
        main_goal_idx = 1000 + round(params[-1])

        main_goal_position = data["mocap_data_robot_save"][main_goal_idx, :]  # (x, y)
        # Secondary goal (green `+`) position
        secondary_goal_position = data["goal_robot_save"]  # (x, y)

        # Compute the Euclidean distance between the green and final goal positions
        distance = np.linalg.norm(secondary_goal_position - main_goal_position)

        # Append the distance to the correct directory's list
        directory_distances[directory_name].append(distance)

        # Determine color based on whether directory ends with "N2_pose"
        color = "gold" if directory_name == "N2_pose" else "blue"

        # Plot secondary goal (green `+`)
        plt.plot(
            secondary_goal_position[0],
            secondary_goal_position[1],
            color="green",
            marker="+",
            markersize=6,
            label="Secondary Goal (Green +)" if idx == 0 else None,
        )

        # Plot main goal with color based on the directory
        plt.plot(
            main_goal_position[0],
            main_goal_position[1],
            color=color,
            marker="+",
            markersize=6,
            label=f"Main Goal ({color.capitalize()} +)" if idx == 0 else None,
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
    plt.title("Goals and Mocap Data from All Files")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    # Plot the Euclidean distance distribution as a violin plot
    distances = [directory_distances["N2_pose"], directory_distances["N2_all"]]
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=distances, inner="quart", cut=0)  # Use seaborn violinplot without xticklabels
    plt.xticks(ticks=[0, 1], labels=["N2_pose", "N2_all"])  # Set x-axis labels manually
    plt.title("Distribution of Euclidean Distances Between Green and Final Goals")
    plt.ylabel("Euclidean Distance (pixels)")
    plt.xlabel("Directory")
    plt.show()



def main():
    # Define multiple directories to process
    directories = ["N2_pose", "N2_all"]

    input_dirs = [os.path.join('../../5_Evaluation', directory) for directory in directories]

    # Get video and data files from all directories
    video_files, data_files, directory_map = get_video_and_data_files(input_dirs)

    # Plot all goals and mocap data together
    print("Plotting all goals and mocap data together.")
    plot_all_goals_and_mocap(data_files, directory_map, params_file_name="N3")

if __name__ == "__main__":
    main()
