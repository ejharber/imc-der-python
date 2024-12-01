import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import VideoFileClip

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
            
            # Now convert the subclip to a gif
            output_filename = filename.replace(".mp4", ".gif")
            output_path = os.path.join(output_folder, output_filename)
            
            # Load the extracted subclip
            subclip = VideoFileClip(temp_subclip_path)
            
            # Convert to gif and save
            subclip.write_gif(output_path)
            print(f"Converted subclip to {output_filename}")
            
            # Optionally, delete the temporary subclip file
            os.remove(temp_subclip_path)

# Example usage
input_folder = "raw_data_N2_final"
output_folder = "raw_data_N2_final"
convert_mp4_to_gif(input_folder, output_folder, start_time=0, end_time=10)  # Convert 0-5 seconds of each video
