import imageio
import numpy as np
import matplotlib.pyplot as plt

def gif_to_numpy_array(gif_path):
    # Open the GIF file with imageio
    reader = imageio.get_reader(gif_path)

    # Extract frames and store them in a list
    frames = []
    for frame in reader:
        print(np.array(frame).shape)
        frames.append(frame[::2,::2,:3])

    # Convert the list of frames to a numpy array
    frames = np.array(frames)

    return frames

# Example usage:

count = 0

results_final = None
for k in range(5):
    results = None

    for i in range(3):

        results_ = None
        for j in range(3):
            gif_path = f"results/{count}.gif"
            count += 1
            gif_frames = gif_to_numpy_array(gif_path)
            if j == 0:
                results_ = gif_frames
            else:
                results_ = np.append(results_, gif_frames, axis=2)

        if i == 0:
            results = results_
        else:
            results = np.append(results, results_, axis=1)
    if k == 0:
        results_final = results
    else:
        results_final = np.append(results_final, results, axis=0)
        

# Create the GIF
# output_file = filename
with imageio.get_writer("combined.gif", mode='I', duration=0.01, loop=0) as writer:
    for i in range(results_final.shape[0]):
        image = results_final[i, :, :, :]
        writer.append_data(image)

    print("Shape of the numpy array:", gif_frames.shape)