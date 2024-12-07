import numpy as np
import os

# Define the new array
data = np.array([
    3.36672471e-03, 3.17003743e-02, 7.69110384e-01, 2.51707586e-03,
    2.18143855e+00, 2.07349739e+02, 1.18899931e+02, 5.06594913e-02,
    8.48618192e-01, 3.43221102e-05, 5.89321834e-02, 2.93649099e-03,
    6.68435806e-02, 1.18266073e+02, 1.33889501e+02
])

# Ensure the directory exists
output_dir = "params"
os.makedirs(output_dir, exist_ok=True)

# Save the array to an npz file
output_file = os.path.join(output_dir, "N2_fake.npz")
np.savez(output_file, params=data)

print(f"Data saved to {output_file} under the key 'params'")
