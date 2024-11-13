import os
import cv2
import numpy as np

import os

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
            base_name = os.path.splitext(file)[0]  # Get the file name without the extension
            data_file = f"{base_name}.npz"        # Construct the expected .npz file name
            data_path = os.path.join(directory, data_file)
            
            # Check if the .npz file exists
            if os.path.exists(data_path):
                video_files.append(os.path.join(directory, file))
                data_files.append(data_path)

    return video_files, data_files


def overlay_data_on_videos(video_file, data_file, params_file_name, output_path=""):

    if not os.path.exists(video_file):
        print(f"File not found: {video_file}")
        return
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return

    # Open the first video to determine properties
    cap = cv2.VideoCapture(video_files[0])
    if not cap.isOpened():
        print(f"Cannot open the video: {video_files[0]}")
        return

    data = np.load(data_file)

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/../2_SysID/params/"
    params = np.load(os.path.join(file_path, params_file_name) + ".npz")["params"]

    goals = filter_data["traj_rope_tip_save"][:, round(params[-1] + 500), :]
    traj_pos = filter_data["traj_rope_tip_save"][:, round(params[-1]):round(params[-1] + 500), :]

    print(data)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Initialize the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Prepare the video captures
    caps = [cv2.VideoCapture(file) for file in video_files]
    
    # Process frame by frame
    for frame_idx in range(total_frames):
        overlay_frame = None
        valid_frame_count = 0
        
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (frame_width, frame_height))
                if overlay_frame is None:
                    overlay_frame = np.zeros_like(frame, dtype=np.float32)
                overlay_frame += frame.astype(np.float32)
                valid_frame_count += 1
        
        if valid_frame_count > 0:
            # Normalize by the number of valid frames and apply opacity
            overlay_frame = (overlay_frame / valid_frame_count) * opacity
            overlay_frame = np.clip(overlay_frame, 0, 255).astype(np.uint8)
            out.write(overlay_frame)
        else:
            print(f"Frame {frame_idx}: No valid frames to overlay.")
            break
    
    # Release all resources
    for cap in caps:
        cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")

# Example Usage
directory_path = '../../1_DataCollection/raw_data_N3_ns'  # Replace with your directory path
video_files, data_files = get_video_and_data_files(directory_path)

for i in range(len(video_files)):
    video_file = video_files[i]
    data_file = data_files[i]

    print(video_file, data_file, params_file_name="N3")
    overlay_data_on_videos(video_file, data_file, )
    break