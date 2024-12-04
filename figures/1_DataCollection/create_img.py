import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the Agg backend is used for non-interactive plotting
import matplotlib
matplotlib.use('Agg')


def overlay_every_nth_frame(video_path, n, output_image_path):
    """
    Overlays every Nth frame of a video on top of each other and saves the result as an image.

    Parameters:
        video_path (str): Path to the video file.
        n (int): Interval of frames to overlay.
        output_image_path (str): Path to save the final image.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare Matplotlib figure
    fig, ax = plt.subplots(figsize=(frame_width / 100, frame_height / 100), dpi=100)

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # Invert y-axis for image coordinates
    ax.axis('off')  # Turn off axes for a clean image

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(frame_idx)

        if frame_idx % n == 0 and frame_idx > 10 and frame_idx < 52:
            # Convert frame to RGB for Matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.18)

        if frame_idx == 52:
            # Convert frame to RGB for Matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.4)

        frame_idx += 1

    # Save the combined image
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    cap.release()
    print(f"Overlay saved to {output_image_path}")


# Example Usage
def main():
    video_path = "../../1_DataCollection/N1/0.mp4"  # Replace with your video file path
    output_image_path = "overlay_output.png"  # Output path for the saved image
    n = 2  # Interval to select every Nth frame

    overlay_every_nth_frame(video_path, n, output_image_path)


if __name__ == "__main__":
    main()
