import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def get_video_and_data_files(directory):
    """
    Get a list of .mp4 video files and their corresponding .npz data files in a directory.

    :param directory: Path to the directory to search for files.
    :return: Tuple containing two lists:
             - List of .mp4 video file paths.
             - List of corresponding .npz data file paths.
    """
    video_files = []
    data_files = []

    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            base_name = os.path.splitext(file)[0]
            data_file = f"{base_name}.npz"
            data_path = os.path.join(directory, data_file)
            if os.path.exists(data_path):
                video_files.append(os.path.join(directory, file))
                data_files.append(data_path)

    return video_files, data_files


def get_plus_positions_from_video(data_file, params_file_name):
    """
    Get the positions where the '+' symbols should be overlaid in each frame of the video.

    :param data_file: Path to the data file.
    :param params_file_name: Name of the parameter file (without extension).
    :return: List of (x, y) positions for the '+' symbol for each frame.
    """
    data = np.load(data_file)
    params = np.load(os.path.join(os.path.dirname(__file__), f"../2_SysID/params/{params_file_name}.npz"))["params"]

    goals = data["mocap_data_camera_save"][1000 + round(params[-1]), 2, :]  # (x, y) in pixels
    goal_time = data["ros_time_save"][1000 + round(params[-1])]
    camera_time = data["ros_time_camera_save"]
    goal_frame_idx = np.argmin(abs(camera_time - goal_time))

    return goals, goal_frame_idx


def overlay_plus_with_matplotlib(video_file, data_file, params_file_name, output_file):
    """
    Overlay '+' symbols on a video using Matplotlib.

    :param video_file: Path to the video file.
    :param data_file: Path to the data file.
    :param params_file_name: Name of the parameter file (without extension).
    :param output_file: Path to save the processed video.
    """
    import cv2

    # Load the data file
    data = np.load(data_file)

    # Get the primary goal positions and frame index for the gold '+' symbol
    goals, goal_frame_idx = get_plus_positions_from_video(data_file, params_file_name)
    x_goal, y_goal = goals

    # Get the secondary goal position (data["goal_camera_save"])
    goal_camera_save = data["goal_camera_save"][0]
    x_goal_camera, y_goal_camera = goal_camera_save

    # Open the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open the video: {video_file}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figure size for less white space
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust padding

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # Invert y-axis for proper image alignment
    ax.set_aspect('equal')

    # Remove axis ticks, labels, and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_frame_on(False)
    ax.grid(False)

    # Prepare video writer
    writer = FFMpegWriter(fps=fps, metadata={"title": "Overlay Video"})

    # Plot initial elements
    with writer.saving(fig, output_file, dpi=500):
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            ax.clear()
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw the green '+' symbol (secondary goal)
            ax.plot(x_goal_camera, y_goal_camera, color="green", marker="+", markersize=12, label="Goal")

            # Draw the gold '+' symbol (main goal) after the specified frame index
            if frame_idx >= goal_frame_idx:
                ax.plot(x_goal, y_goal, color="gold", marker="+", markersize=12, label="Mocap")

                # Connect the two goals with a dashed line
                ax.plot([x_goal_camera, x_goal], [y_goal_camera, y_goal], linestyle="--", color="gray", alpha=0.7)

            ax.legend()
            # Add the legend inside the plot area
            legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, fontsize=10)
            legend.get_frame().set_alpha(0.8)  # Make the legend background slightly transparent

            writer.grab_frame()

    cap.release()
    print(f"Processed video saved: {output_file}")


def plot_all_goals_and_mocap(data_files, params_file_name):
    """
    Plot all goal and mocap data from multiple .npz files in a single Matplotlib figure.

    :param data_files: List of paths to .npz data files containing goal and mocap data.
    :param params_file_name: Name of the parameter file (without extension).
    """
    plt.figure(figsize=(10, 10))

    for data_file in data_files:
        # Load data and parameters
        data = np.load(data_file)
        params = np.load(os.path.join(os.path.dirname(__file__), f"../2_SysID/params/{params_file_name}.npz"))["params"]

        # Main goal (gold `+`) position
        main_goal_idx = 1000 + round(params[-1])
        main_goal_position = data["mocap_data_robot_save"][main_goal_idx, :]  # (x, y)

        # Secondary goal (green `+`) position
        secondary_goal_position = data["goal_robot_save"]  # (x, y)

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
    plt.title("Goals and Mocap Data from All Files")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def main():
    directory = "N2_pose"
    input_dir = '../5_Evaluation/' + directory
    output_dir = './5_Evaluation_' + directory
    os.makedirs(output_dir, exist_ok=True)

    video_files, data_files = get_video_and_data_files(input_dir)

    # Plot all goals and mocap data together
    print("Plotting all goals and mocap data together.")
    plot_all_goals_and_mocap(data_files, params_file_name="N3")

    # Overlay '+' symbols on videos using Matplotlib
    for i in range(len(video_files)):
        video_file = video_files[i]
        data_file = data_files[i]
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(video_file)}")

        overlay_plus_with_matplotlib(video_file, data_file, params_file_name="N3", output_file=output_file)


if __name__ == "__main__":
    main()
