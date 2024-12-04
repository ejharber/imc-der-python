import numpy as np
import os

# Define the new array
data = np.array([
    2.42954407e-04, 5.51194816e-02, 7.68815834e-01, 5.48419216e-02,
    6.46894271e+00, 2.21026767e+02, 7.18248586e+03, 1.53457088e-03,
    4.26410648e-02, 2.38844994e+00, 1.89942664e-01, 8.89857051e-03,
    7.72389844e-02, 5.00259080e+02, 5.16039651e+02
])

# Ensure the directory exists
output_dir = "params"
os.makedirs(output_dir, exist_ok=True)

# Save the array to an npz file
output_file = os.path.join(output_dir, "N2_all.npz")
np.savez(output_file, params=data)

print(f"Data saved to {output_file} under the key 'params'")
