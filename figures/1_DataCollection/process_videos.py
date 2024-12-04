import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import VideoFileClip

# Ensure the Agg backend is used for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

def get_video_and_data_files(directory):
    """
    Get a list of .mp4 video files and their corresponding .npz data files in a directory.
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
    """
    data = np.load(data_file)
    params = np.load(os.path.join(os.path.dirname(__file__), f"../../2_SysID/params/{params_file_name}.npz"))["params"]

    goals = data["mocap_data_camera_save"][500 + round(params[-1]), 2, :]  # (x, y) in pixels
    goal_time = data["ros_time_save"][500 + round(params[-1])]
    camera_time = data["ros_time_camera_save"]
    goal_frame_idx = np.argmin(abs(camera_time - goal_time))

    return goals, goal_frame_idx

def overlay_plus_on_video_with_matplotlib(video_file, data_file, params_file_name, output_file):
    """
    Overlay a '+' symbol on a video at the goal position extracted from the data file using Matplotlib.
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

    # Set up the figure for Matplotlib
    fig, ax = plt.subplots(figsize=(frame_width / 200, frame_height / 200), dpi=200)

    # Video Writer Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ax.clear()
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)

        # Plot the frame and add the '+' marker at the goal position
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_frame)

        # Explicitly add the '+' symbol with a label
        if frame_idx >= goal_frame_idx:
            ax.plot(goals[0], goals[1], marker='+', color='gold', markersize=12, linestyle='', label='Measured')  # Plot '+' marker
        else:
            ax.plot(-100, -100, marker='+', color='gold', markersize=12, linestyle='', label='Measured')  # Plot '+' marker

        # Set the legend
        ax.legend(fontsize=16, loc='upper left')  # Ensure the legend font size remains large

        # Convert the plot to an image
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_frame = plt_frame.reshape(frame_height, frame_width, 3)
        plt_frame = cv2.cvtColor(plt_frame, cv2.COLOR_RGB2BGR)
        out.write(plt_frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved: {output_file}")

def overlay_videos_with_matplotlib(video_files, data_files, output_path, params_file_name):
    """
    Overlay multiple videos into a single video by combining frames using Matplotlib.
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

    # Prepare positions for '+' overlay
    plus_positions = [get_plus_positions_from_video(data, params_file_name) for data in data_files]

    # Set up Matplotlib figure
    fig, ax = plt.subplots(figsize=(frame_width / 200, frame_height / 200), dpi=200)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame_idx in range(min_frames):
        combined_image = np.zeros((frame_height, frame_width, 3), dtype=np.float32)
        ax.clear()
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)  # Invert y-axis for image coordinates

        for video_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to RGB for Matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.1)

            # Overlay '+' symbol
            goals, goal_frame_idx = plus_positions[video_idx]

            if video_idx == 0:
                # Explicitly add the '+' symbol with a label
                if frame_idx >= goal_frame_idx:
                    ax.plot(goals[0], goals[1], marker='+', color='gold', markersize=12, linestyle='', label='Measured')  # Plot '+' marker
                else:
                    ax.plot(-100, -100, marker='+', color='gold', markersize=12, linestyle='', label='Measured')  # Plot '+' marker
            else:
                # Explicitly add the '+' symbol with a label
                if frame_idx >= goal_frame_idx:
                    ax.plot(goals[0], goals[1], marker='+', color='gold', markersize=12, linestyle='')  # Plot '+' marker
                else:
                    ax.plot(-100, -100, marker='+', color='gold', markersize=12, linestyle='')  # Plot '+' marker

        ax.legend(fontsize=16, loc='upper left')  # Ensure the legend font size remains large

        # Draw the frame and convert it to a NumPy array
        plt.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        plt_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_frame = plt_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_frame = cv2.cvtColor(plt_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        # Write the frame to the output video
        out.write(plt_frame)

    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    plt.close(fig)
    print(f"Combined video saved: {output_path}")

def append_videos_with_matplotlib(video_files, data_files, output_path, params_file_name, speed_factor):
    """
    Append multiple videos one after another with measured goals staying visible.
    Display the speed factor in the bottom-left corner and adjust the video playback speed.
    """
    if not video_files:
        print("No video files to process.")
        return

    # Open all videos to check properties
    caps = [cv2.VideoCapture(file) for file in video_files]
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    new_fps = int(original_fps * speed_factor)

    # Prepare positions for '+' overlay
    plus_positions = [get_plus_positions_from_video(data, params_file_name) for data in data_files]

    # Set up Matplotlib figure
    fig, ax = plt.subplots(figsize=(frame_width / 200, frame_height / 200), dpi=200)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))

    # Loop through each video sequentially
    goals = np.zeros((0, 2))
    for video_idx, cap in enumerate(caps):
        goal, goal_frame_idx = plus_positions[video_idx]
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to adjust speed
            if frame_idx % speed_factor != 0:
                frame_idx += 1
                continue

            if frame_idx > 60:
                continue 

            ax.clear()
            ax.set_xlim(0, frame_width)
            ax.set_ylim(frame_height, 0)  # Invert y-axis for image coordinates

            # Convert frame to RGB for Matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_frame)

            # Add '+' marker when the goal frame is reached
            if frame_idx >= goal_frame_idx:
                goals = np.append(goals, [goal], axis=0)
                goal_frame_idx = np.inf

            if goals.shape[0] > 0:
                ax.plot(goals[:, 0], goals[:, 1], marker='+', color='gold', markersize=12, linestyle='', label='Measured')
            else:
                ax.plot(-100, -100, marker='+', color='gold', markersize=12, linestyle='', label='Measured')

            # Add legend (appears only once)
            ax.legend(fontsize=16, loc='upper left')

            # Add speed factor text
            ax.text(
                20, frame_height - 50,  # Position in pixels (bottom-left corner)
                f"{int(speed_factor)}x", color="white", fontsize=32, 
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
            )

            # Convert the Matplotlib figure to an image
            plt.axis('off')

            plt.tight_layout()

            fig.canvas.draw()
            plt_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plt_frame = plt_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt_frame = cv2.cvtColor(plt_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

            # print(plt_frame.shape, frame_width, frame_height)
            # Crop the plt_frame to the central region
            # cropped_frame = plt_frame[100:-100, 100:-100, :]
            # print(cropped_frame.shape, frame_width, frame_height)

            # Write the frame to the output video
            out.write(plt_frame)

            frame_idx += 1

    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    plt.close(fig)
    print(f"Appended video saved: {output_path}")

def convert_mp4_to_gif(input_folder, output_folder, start_time=0, end_time=10):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            # Full path to the input video file
            input_path = os.path.join(input_folder, filename)
            
            # Define the output subclip file path
            temp_subclip_path = os.path.join(output_folder, f"temp_{filename}")
            
            # Extract the subclip from the video (start_time to end_time)
            ffmpeg_extract_subclip(input_path, start_time, end_time, temp_subclip_path)
            print(f"Extracted subclip from {filename} between {start_time} and {end_time} seconds.")
            
            # Load the extracted subclip
            subclip = VideoFileClip(temp_subclip_path)
            
            # Downsample the video by half
            downsampled_subclip = subclip.resized(0.5)  # Resize to 50% of original dimensions
            
            # Define the output gif file path
            output_filename = filename.replace(".mp4", ".gif")
            output_path = os.path.join(output_folder, output_filename)
            
            # Convert the resized subclip to gif and save
            downsampled_subclip.write_gif(output_path)
            print(f"Converted and downsampled subclip to {output_filename}")
            
            # Clean up resources
            subclip.close()
            downsampled_subclip.close()
            
            # Optionally, delete the temporary subclip file
            os.remove(temp_subclip_path)

# Main Workflow
def main():
    directory = "N4"
    input_dir = '../../1_DataCollection/' + directory
    output_dir = './' + directory
    os.makedirs(output_dir, exist_ok=True)

    video_files, data_files = get_video_and_data_files(input_dir)

    processed_videos = []
    for i in range(len(video_files)):
        video_file = video_files[i]
        data_file = data_files[i]
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(video_file)}")
        overlay_plus_on_video_with_matplotlib(video_file, data_file, params_file_name="N2", output_file=output_file)
        processed_videos.append(output_file)

    if video_files and data_files:
        output_file = os.path.join(output_dir, "overlayed_video.mp4")
        overlay_videos_with_matplotlib(video_files, data_files, output_file, params_file_name="N2")

    if video_files and data_files:
        for speed_factor in [1, 2, 4, 8, 16]:
            output_file = os.path.join(output_dir, f"appended_video_{speed_factor}.mp4")
            append_videos_with_matplotlib(video_files, data_files, output_file, params_file_name="N2", speed_factor=speed_factor)

    convert_mp4_to_gif(directory, directory, start_time=0, end_time=1e6)  # Convert 0-10 seconds of each video

if __name__ == "__main__":
    main()
