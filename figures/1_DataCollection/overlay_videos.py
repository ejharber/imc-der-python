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
    Overlay a gold '+' symbol on a video at the goal position extracted from the data file.

    :param video_file: Path to the video file.
    :param data_file: Path to the data file.
    :param params_file_name: Name of the parameter file (without extension).
    :param output_file: Path to save the processed video.
    """
    if not os.path.exists(video_file) or not os.path.exists(data_file):
        print(f"File not found: {video_file} or {data_file}")
        return

    # Get the goal positions and frame index for the '+' symbol
    goals, goal_frame_idx = get_plus_positions_from_video(data_file, params_file_name)

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

        if frame_idx >= goal_frame_idx:
            x, y = int(goals[0]), int(goals[1])
            if 0 <= x < frame_width and 0 <= y < frame_height:
                color = (0, 215, 255)  # Gold in BGR
                thickness = 1
                size = 5
                cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
                cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

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

# Main Workflow
def main():
    directory = "raw_data_N4_final (copy)"
    input_dir = '../../1_DataCollection/' + directory
    output_dir = './' + directory
    os.makedirs(output_dir, exist_ok=True)

    video_files, data_files = get_video_and_data_files(input_dir)

    processed_videos = []
    for i in range(len(video_files)):
        video_file = video_files[i]
        data_file = data_files[i]
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(video_file)}")
        overlay_plus_on_video(video_file, data_file, params_file_name="N3", output_file=output_file)
        processed_videos.append(output_file)

    # Combine all processed videos
    combined_output = os.path.join(output_dir, "combined_video.mp4")
    overlay_videos(processed_videos, data_files, combined_output, params_file_name="N3")


if __name__ == "__main__":
    main()
