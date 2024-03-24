from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process

# tune the model of the pendulum to the real world data by fitting its 3 physics parameters
cost = []
cost_1 = []
for i in range(10000):

    x_sim, y_sim = pendulum_sim(L = 2)
    x_measured, y_measured = pendulum_real()

    # L2 cost between sim and measured values
    # if you add or subtract values you square the variance, add then squaroot
    # if you multiply a value by a scalar you do the same to the std
    cost_1.append((x_sim[0] - x_measured[0]))

    cost.append(np.sum(abs(x_sim - x_measured)) + np.sum(abs(y_sim - y_measured)))
    # cost has a std of about 0.3
    # variance cant be caluclated in closed form
                            

# plt.hist(cost, bins=40)
# plt.show()

noise = np.sqrt(2*len(x_measured)*0.05**2)
print(noise)

noise = np.sqrt(2*len(x_measured)*0.05)
print(noise)

print(np.std(cost))

print(np.std(cost_1))

plt.hist(cost_1, bins=40)
plt.show()
