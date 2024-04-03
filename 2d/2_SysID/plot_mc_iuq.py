from pendulum import *

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.stats import wasserstein_distance
from scipy.stats import skewnorm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

data = np.load("MC_IUQ.npz")
min_x = data["min_x"]
min_x_dist = data["min_x_dist"]

print(min_x, min_x_dist.shape)
min_x_dist = min_x_dist[:,:,0]

costs_real = []
# for _ in range(5_000):
    # costs_real.append(cost_fun_noise(min_x))

# plt.figure(1)
# plt.hist(costs_real, alpha=0.5, bins=40, density=True)

# plt.figure(2)
##
costs_est = []

for i in range(3):
    plt.figure(i)
    plt.hist(min_x_dist[i, :], alpha=0.5, bins=40, density=True)
plt.show()
# for i in range(5_000):
#     index = 0
#     min_x_sample = np.copy(min_x)
#     min_x_sample[index] = min_x_dist[index, i]
#     costs_est.append(cost_fun(min_x_sample))

# plt.hist(costs_est, alpha=0.5, bins=40, density=True)
# plt.show()
