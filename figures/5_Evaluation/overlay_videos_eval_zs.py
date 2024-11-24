import os
import cv2
import numpy as np


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
    params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

    goals = data["mocap_data_camera_save"][1000 + round(params[-1]), 2, :]  # (x, y) in pixels
    goal_time = data["ros_time_save"][1000 + round(params[-1])]
    camera_time = data["ros_time_camera_save"]
    goal_frame_idx = np.argmin(abs(camera_time - goal_time))

    return goals, goal_frame_idx

def overlay_plus_on_video(video_file, data_file, params_file_name, output_file):
    """
    Overlay two '+' symbols and a dashed line on a video:
    - A gold '+' for the main goal position, added partway through the video.
    - A green '+' for the secondary position specified by data["goal_camera_save"], visible throughout the video.
    - A dashed line connecting the green '+' and the gold '+' after the green '+' is added.

    :param video_file: Path to the video file.
    :param data_file: Path to the data file.
    :param params_file_name: Name of the parameter file (without extension).
    :param output_file: Path to save the processed video.
    """
    if not os.path.exists(video_file) or not os.path.exists(data_file):
        print(f"File not found: {video_file} or {data_file}")
        return

    # Load the data file
    data = np.load(data_file)

    # Get the primary goal positions and frame index for the gold '+' symbol
    goals, goal_frame_idx = get_plus_positions_from_video(data_file, params_file_name)
    x_goal, y_goal = int(goals[0]), int(goals[1])

    # Get the secondary goal position (data["goal_camera_save"])
    goal_camera_save = data["goal_camera_save"][0]
    x_goal_camera, y_goal_camera = int(goal_camera_save[0]), int(goal_camera_save[1])

    # Open the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open the video: {video_file}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the gold '+' symbol (main goal) after the specified frame index
        if frame_idx >= goal_frame_idx:
            if 0 <= x_goal < frame_width and 0 <= y_goal < frame_height:
                color_gold = (0, 215, 255)  # Gold in BGR
                thickness = 1
                size = 5
                cv2.line(frame, (x_goal - size, y_goal), (x_goal + size, y_goal), color_gold, thickness)
                cv2.line(frame, (x_goal, y_goal - size), (x_goal, y_goal + size), color_gold, thickness)

        # Draw the green '+' symbol (secondary goal) from the start
        if 0 <= x_goal_camera < frame_width and 0 <= y_goal_camera < frame_height:
            color_green = (0, 255, 0)  # Green in BGR
            thickness = 1
            size = 5
            cv2.line(frame, (x_goal_camera - size, y_goal_camera), (x_goal_camera + size, y_goal_camera), color_green, thickness)
            cv2.line(frame, (x_goal_camera, y_goal_camera - size), (x_goal_camera, y_goal_camera + size), color_green, thickness)

        # Draw a dashed line connecting the green '+' to the gold '+' after the green '+' is added
        if frame_idx >= goal_frame_idx:
            if 0 <= x_goal < frame_width and 0 <= y_goal < frame_height and 0 <= x_goal_camera < frame_width and 0 <= y_goal_camera < frame_height:
                color_dash = (255, 255, 255)  # White dashed line
                thickness = 1
                line_length = 5  # Length of dashes
                gap_length = 3  # Gap between dashes

                # Calculate the vector components and the distance
                dx = x_goal - x_goal_camera
                dy = y_goal - y_goal_camera
                distance = int(np.sqrt(dx**2 + dy**2))
                num_dashes = distance // (line_length + gap_length)

                # Draw the dashed line
                for i in range(num_dashes):
                    start_x = x_goal_camera + (i * (line_length + gap_length)) * dx // distance
                    start_y = y_goal_camera + (i * (line_length + gap_length)) * dy // distance
                    end_x = start_x + line_length * dx // distance
                    end_y = start_y + line_length * dy // distance
                    cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color_dash, thickness)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved: {output_file}")


def overlay_videos(video_files, data_files, output_path, params_file_name):
    """
    Overlay multiple videos into a single video without opacity blending. 
    Directly overlay the '+' symbols from each video, stopping at the shortest video length.

    :param video_files: List of video file paths to overlay.
    :param data_files: List of data file paths corresponding to the video files.
    :param output_path: Path to save the combined video.
    :param params_file_name: Name of the parameter file (without extension).
    """
    if not video_files:
        print("No video files to process.")
        return

    # Open all videos and determine the minimum frame count
    caps = [cv2.VideoCapture(file) for file in video_files]
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    min_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Prepare positions for '+' overlay
    plus_positions = [get_plus_positions_from_video(data, params_file_name) for data in data_files]

    # Process frames up to the minimum frame count
    for frame_idx in range(min_frames):
        combined_frame = None
        for video_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                # Draw the '+' symbol for the current video
                goals, goal_frame_idx = plus_positions[video_idx]
                if frame_idx >= goal_frame_idx:
                    x, y = int(goals[0]), int(goals[1])
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        color = (0, 215, 255)  # Gold in BGR
                        thickness = 1
                        size = 5
                        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
                        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

                # Accumulate frame into the combined video
                if combined_frame is None:
                    combined_frame = frame.astype(np.float32)
                else:
                    combined_frame = np.maximum(combined_frame, frame.astype(np.float32))

        if combined_frame is not None:
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            out.write(combined_frame)
        else:
            print(f"Frame {frame_idx}: No valid frames.")
            break

    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    print(f"Combined video saved: {output_path}")

import matplotlib.pyplot as plt

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
        params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

        # Main goal (gold `+`) position
        main_goal_idx = 1000 + round(params[-1])
        # print(data["mocap_data_robot_save"].shape)
        # print(data["goal_robot_save"].shape)

        # plt.plot(data["mocap_data_robot_save"][:, 0], data["mocap_data_robot_save"][:, 1])

        main_goal_position = data["mocap_data_robot_save"][main_goal_idx, :]  # (x, y)
        # Secondary goal (green `+`) position
        secondary_goal_position = data["goal_robot_save"]  # (x, y)
        print(data_file, secondary_goal_position)
        # print(secondary_goal_position.shape)

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
    input_dir = '../../5_Evaluation/' + directory
    output_dir = './' + directory
    os.makedirs(output_dir, exist_ok=True)

    video_files, data_files = get_video_and_data_files(input_dir)

    # Plot all goals and mocap data together
    print("Plotting all goals and mocap data together.")
    plot_all_goals_and_mocap(data_files, params_file_name="N3")

    processed_videos = []
    for i in range(len(video_files)):
        video_file = video_files[i]
        data_file = data_files[i]
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(video_file)}")

        # Process the video
        overlay_plus_on_video(video_file, data_file, params_file_name="N3", output_file=output_file)
        processed_videos.append(output_file)

    # Combine all processed videos
    combined_output = os.path.join(output_dir, "combined_video.mp4")
    overlay_videos(processed_videos, data_files, combined_output, params_file_name="N3")


if __name__ == "__main__":
    main()
