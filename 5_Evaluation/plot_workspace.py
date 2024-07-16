import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../4_SupervizedLearning")
from load_data import load_data_zeroshot

# Load data using the function
actions_train, goals_train, actions_test, goals_test = load_data_zeroshot("../3_ExpandDataSet/raw_data")

# Check if goals_train and goals_test are in the correct shape (N, 2)
if goals_train.shape[1] != 2 or goals_test.shape[1] != 2:
    raise ValueError("Goals should be a (N, 2) array where N is the number of points")

# Plot goals
plt.figure(figsize=(8, 6))
plt.scatter(goals_train[:, 0], goals_train[:, 1], c='blue', s=0.1, label='Training Goals')  # Smaller points
plt.scatter(goals_test[:, 0], goals_test[:, 1], c='red', s=0.1, label='Testing Goals')  # Smaller points
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Goals Points Plot')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Set axis to be equal
plt.show()
