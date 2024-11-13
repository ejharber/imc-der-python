import os
import cv2

def check_video_files(directory_path):
    corrupted_files = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            file_path = os.path.join(directory_path, filename)
            
            # Try opening the video file using OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"{filename} - CORRUPTED or cannot be opened")
                corrupted_files.append(filename)
            else:
                # Try reading one frame to ensure the video is readable
                ret, frame = cap.read()
                if not ret:
                    print(f"{filename} - Cannot read frames")
                    corrupted_files.append(filename)
                else:
                    print(f"{filename} - OK")
                cap.release()
    
    if corrupted_files:
        print("\nCorrupted files found:")
        for f in corrupted_files:
            print(f)
    else:
        print("\nAll video files are playable.")

# Usage
directory_path = '../../1_DataCollection/raw_data_N2_ns'
check_video_files(directory_path)
