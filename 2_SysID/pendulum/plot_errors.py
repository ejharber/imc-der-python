from pendulum import *

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

plt.figure('no error')
for std in [0, 0.05, 0.01, 0.02, 0.05]:
    costs = []
    for _ in range(5_000):
        costs.append(cost_fun_noise([1, 0.5, 1], stdx=std, stdy=std))

    plt.hist(costs, bins = 40, label=str(std))

plt.legend()

plt.figure('error')
for std in [0, 0.05, 0.01, 0.02, 0.05]:
    costs = []
    for _ in range(5_000):
        costs.append(cost_fun_noise([1.1, 0.6, 1.1], stdx=std, stdy=std))

    plt.hist(costs, bins = 40, label=str(std))

plt.legend()
plt.show()

# plt.show()
