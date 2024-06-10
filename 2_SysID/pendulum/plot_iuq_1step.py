from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process


data = np.load("IUQ_1step_realistic.npz")

min_x_dist = data["min_x_dist"]

print(min_x_dist.shape)

plt.figure(1)
plt.hist(min_x_dist[:, 0], alpha=0.5, bins=40, density=True)

# plt.hist(min_x_dist[:, 0] / np.mean(min_x_dist[:, 0]), alpha=0.5, bins=40, density=True)
plt.axvline(x = 1, color='k', linestyle='--')
plt.title("Probability Distribution of L")
plt.xlabel("Length (m)")
plt.ylabel("Probability")

plt.figure(2)
plt.hist(min_x_dist[:, 1], alpha=0.5, bins=40, density=True)

# plt.hist(min_x_dist[:, 1] / np.mean(min_x_dist[:, 1]) * 0.5, alpha=0.5, bins=40, density=True)
plt.axvline(x = 0.5, color='k', linestyle='--')
plt.title("Probability Distribution of b")
plt.xlabel("dampening (Ns/m)")
plt.ylabel("Probability")

plt.figure(3)
plt.hist(min_x_dist[:, 2], alpha=0.5, bins=40, density=True)
# plt.hist(min_x_dist[:, 2] / np.mean(min_x_dist[:, 2]), alpha=0.5, bins=40, density=True)
plt.axvline(x = 1, color='k', linestyle='--')
plt.title("Probability Distribution of m")
plt.xlabel("mass (kg)")
plt.ylabel("Probability")

plt.show()