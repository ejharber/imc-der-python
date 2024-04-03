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

data = np.load("DE_uniform_e.npz")
min_x = data["min_x"]
min_x_std = data["min_x_std"]

print(min_x, min_x_std)

costs_real = []
for _ in range(5_000):
    costs_real.append(cost_fun_noise(min_x))

plt.figure(1)
plt.hist(costs_real, alpha=0.5, bins=40, density=True)

# plt.figure(2)
##
costs_est = []
for _ in range(5_000):
    i = 0
    min_x_sample = np.copy(min_x)
    # min_x_sample[i] = np.random.gamma(min_x[i]/min_x_std[i], min_x_std[i])
    # min_x_sample[i] = np.random.gumbel(min_x[i], min_x_std[i])
    print(min_x_std)
    # min_x_sample[i] = np.random.uniform(min_x[i] - min_x_std[i, 0], min_x[i] + min_x_std[i, 1])
    min_x_sample[i] = np.random.uniform(min_x[i] - min_x_std[i, 0], min_x[i] + min_x_std[i, 1])
    # min_x_sample[i] = np.random.triangular(min_x[i]-min_x_std[i, 0], min_x[i], min_x[i]+min_x_std[i, 1])
    # print(min_x_sample, min_x)
    # min_x[i] = 0.978
    # std_1 = 4.5905
    # std_2 = 0.03996634
    # min_x_sample[i] = skewnorm.rvs(a=std_1, loc=min_x[i], scale=std_2, size=1)
    costs_est.append(cost_fun_2(min_x, min_x_sample))

plt.hist(costs_est, alpha=0.5, bins=40, density=True)
plt.show()
