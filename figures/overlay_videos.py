import os
import cv2
import numpy as np

def get_video_files(directory):
    """
    Get a list of all .mp4 files in a given directory.

    :param directory: Path to the directory to search for video files.
    :return: List of file paths to .mp4 files.
    """
    video_files = []
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            video_files.append(os.path.join(directory, file))
    return video_files

def overlay_videos(video_files, output_path, opacity=1.0):
    """
    Overlay multiple videos into a single video with adjustable opacity.

    :param video_files: List of video file paths to overlay.
    :param output_path: Path to save the output video.
    :param opacity: Opacity multiplier for each video, relative to the number of videos.
    """
    # Check all video paths
    for video in video_files:
        if not os.path.exists(video):
            print(f"File not found: {video}")
            return
    
    # Open the first video to determine properties
    cap = cv2.VideoCapture(video_files[0])
    if not cap.isOpened():
        print(f"Cannot open the video: {video_files[0]}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
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
video_files = get_video_files(directory_path)

if video_files:
    output_path = 'raw_data_N3_ns.mp4'
    overlay_videos(video_files, output_path, opacity=1.0)
else:
    print(f"No .mp4 files found in the directory: {directory_path}")
